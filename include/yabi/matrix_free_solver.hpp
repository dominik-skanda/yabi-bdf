/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_MATRIX_FREE_SOLVER_HPP
#define YABI_MATRIX_FREE_SOLVER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/linear_solver.hpp>
#include <yabi/matrix_free_operator.hpp>

namespace yabi {

// Krylov method used by the matrix-free Newton solve.
enum class KrylovMethod { Gmres, Fgmres, Bicgstab };

// Preconditioner applied by the Krylov iteration.  Jacobi uses the diagonal of
// the scaled Newton matrix passed to factorizeCsc(); Direct factorizes that
// matrix with an inner direct solver (DenseLUSolver by default).  A matrix-free
// state solve still evaluates J only through the MatrixFreeOperator; the CSC
// matrix is used solely to build/scale the preconditioner.
enum class KrylovPreconditioner { None, Jacobi, Direct };

// Shared implementation for the three matrix-free LinearSolver front-ends.
//
// Interface contract:
//   * The Newton solver installs a MatrixFreeOperator for P = D1*(I - c*J)*D2.
//   * factorizeCsc(P_csc, ...) prepares the preconditioner from the scaled P.
//   * solveCsc(P_csc, ..., b, ...) runs the Krylov iteration on the operator.
// The dense factorize()/solve() interface is deliberately a direct LU fallback:
// it is used by the unchanged dense path (and by sensitivity solves, which
// still need an explicit J/H).  This keeps a matrix-free solver a drop-in
// replacement without changing the sensitivity contract.
template <typename Scalar> class KrylovSolverBase : public LinearSolver<Scalar> {
  public:
    explicit KrylovSolverBase(KrylovMethod method) : method_(method) {
    }
    virtual ~KrylovSolverBase() {
    }

    bool initialize(unsigned int n) override {
        n_ = n;
        diagonal_.clear();
        jacobi_valid_ = false;
        preconditioner_matrix_ = nullptr;
        ip_.clear();
        scaling_D1_.clear();
        scaling_D2_.clear();
        last_iterations_ = 0;
        last_converged_ = false;
        if (fallback_)
            fallback_->initialize(n);
        return true;
    }

    bool supportsMatrixFree() const override {
        return true;
    }
    bool prefersSparseFactorization() const override {
        return false;
    }

    void setMatrixFreeOperator(const MatrixFreeOperator<Scalar> *op) override {
        operator_ = op;
    }

    KrylovMethod method() const {
        return method_;
    }

    void setPreconditioner(KrylovPreconditioner preconditioner) {
        preconditioner_ = preconditioner;
    }
    KrylovPreconditioner preconditioner() const {
        return preconditioner_;
    }

    // Optional explicit direct preconditioner; when unset, Direct uses an
    // owned DenseLUSolver.
    void setDirectPreconditioner(std::unique_ptr<LinearSolver<Scalar>> solver) {
        direct_ = std::move(solver);
    }

    void setTolerance(Scalar relative, Scalar absolute, unsigned int max_iterations,
                      unsigned int restart) {
        relative_tolerance_ = relative;
        absolute_tolerance_ = absolute;
        max_iterations_ = max_iterations == 0 ? 1 : max_iterations;
        restart_ = restart == 0 ? 1 : restart;
    }

    Scalar relativeTolerance() const {
        return relative_tolerance_;
    }
    unsigned int maxIterations() const {
        return max_iterations_;
    }
    unsigned int restarts() const {
        return restart_;
    }
    unsigned int lastIterations() const {
        return last_iterations_;
    }
    bool lastConverged() const {
        return last_converged_;
    }
    unsigned int fallbackCount() const {
        return fallbacks_;
    }

    // Dense fallback: a matrix-free solver can still be installed for an
    // integrator that later requests sensitivities.  Those solves must use the
    // unchanged explicit-J path, so they delegate to a direct factorization.
    bool factorize(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
                   unsigned int n) override {
        ensureFallback();
        fallback_->initialize(n);
        return fallback_->factorize(P, ipiv, scalingD1, scalingD2, n);
    }

    int solve(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2, Scalar *b, unsigned int n,
              unsigned int nrhs) override {
        ensureFallback();
        fallback_->initialize(n);
        return fallback_->solve(P, ipiv, scalingD1, scalingD2, b, n, nrhs);
    }

    // The Krylov path is activated explicitly by the Newton solver; it is not
    // advertised as a CSC direct solver so the T31 sparse path does not claim
    // it.
    bool supportsCsc() const override {
        return false;
    }

    bool factorizeCsc(const CscMatrix<Scalar> &P, int *ipiv, long *scalingD1, long *scalingD2,
                      unsigned int n) override {
        n_ = n;
        preconditioner_matrix_ = &P;

        if (preconditioner_ == KrylovPreconditioner::Jacobi) {
            buildJacobi(P);
        } else if (preconditioner_ == KrylovPreconditioner::Direct) {
            ensureDirect();
            direct_->initialize(n);
            if (!direct_->factorizeCsc(P, ipiv, scalingD1, scalingD2, n))
                return false;
        } else {
            jacobi_valid_ = false;
        }

        // Copy the pivots only after the inner direct factorization has filled
        // them; the direct preconditioner applies them on every iteration.
        ip_.assign(ipiv, ipiv + n);
        scaling_D1_.assign(scalingD1, scalingD1 + n);
        scaling_D2_.assign(scalingD2, scalingD2 + n);
        return true;
    }

    int solveCsc(const CscMatrix<Scalar> &P, int *ipiv, long *scalingD1, long *scalingD2, Scalar *b,
                 unsigned int n, unsigned int nrhs) override {
        n_ = n;
        if (nrhs == 0)
            return 0;

        if (!operator_) {
            // No matrix-free product was installed: be a safe direct solver.
            ensureFallback();
            fallback_->initialize(n);
            if (!fallback_->factorizeCsc(P, ipiv, scalingD1, scalingD2, n))
                return -1;
            return fallback_->solveCsc(P, ipiv, scalingD1, scalingD2, b, n, nrhs);
        }

        for (unsigned int rhs = 0; rhs < nrhs; ++rhs) {
            Scalar *b_rhs = b + (std::size_t)rhs * n;
            std::vector<Scalar> rhs_vector(b_rhs, b_rhs + n);
            std::vector<Scalar> solution(n, Scalar(0));

            bool ok = false;
            switch (method_) {
            case KrylovMethod::Gmres:
                ok = gmres(rhs_vector, solution, /*flexible=*/false);
                break;
            case KrylovMethod::Fgmres:
                ok = gmres(rhs_vector, solution, /*flexible=*/true);
                break;
            case KrylovMethod::Bicgstab:
                ok = bicgstab(rhs_vector, solution);
                break;
            }

            last_converged_ = ok;
            if (!ok) {
                // Robustness net: P is available as the scaled CSC matrix, so a
                // failed/breakdown Krylov solve can still produce the correct
                // Newton step.  The counter makes the fallback observable.
                ++fallbacks_;
                ensureFallback();
                fallback_->initialize(n);
                if (!fallback_->factorizeCsc(P, ipiv, scalingD1, scalingD2, n))
                    return -1;
                if (fallback_->solveCsc(P, ipiv, scalingD1, scalingD2, b_rhs, n, 1) != 0)
                    return -1;
            } else {
                std::copy(solution.begin(), solution.end(), b_rhs);
            }
        }
        return 0;
    }

  protected:
    Scalar dot(const std::vector<Scalar> &a, const std::vector<Scalar> &b) const {
        Scalar sum = Scalar(0);
        for (std::size_t i = 0; i < a.size(); ++i)
            sum += a[i] * b[i];
        return sum;
    }

    Scalar norm(const std::vector<Scalar> &a) const {
        return std::sqrt(dot(a, a));
    }

    bool applyOperator(const Scalar *x, Scalar *y) const {
        if (!operator_)
            return false;
        if (!operator_->apply(x, y))
            return false;
        for (unsigned int i = 0; i < n_; ++i)
            if (!std::isfinite(y[i]))
                return false;
        return true;
    }

    bool applyPreconditioner(const Scalar *input, Scalar *output) {
        if (preconditioner_ == KrylovPreconditioner::Jacobi && jacobi_valid_) {
            for (unsigned int i = 0; i < n_; ++i)
                output[i] = input[i] / diagonal_[i];
            return true;
        }
        if (preconditioner_ == KrylovPreconditioner::Direct && preconditioner_matrix_) {
            ensureDirect();
            std::copy(input, input + n_, output);
            return direct_->solveCsc(*preconditioner_matrix_, ip_.data(), scaling_D1_.data(),
                                     scaling_D2_.data(), output, n_, 1) == 0;
        }
        std::copy(input, input + n_, output);
        return true;
    }

    bool gmres(const std::vector<Scalar> &b, std::vector<Scalar> &x, bool flexible) {
        const unsigned int n = n_;
        x.assign(n, Scalar(0));

        const Scalar bnorm = norm(b);
        if (!std::isfinite(bnorm))
            return false;
        const Scalar tolerance = absolute_tolerance_ + relative_tolerance_ * bnorm;
        if (bnorm <= tolerance) {
            last_iterations_ = 0;
            return true;
        }

        std::vector<Scalar> r = b;
        unsigned int total = 0;

        while (total < max_iterations_) {
            const Scalar beta = norm(r);
            if (!std::isfinite(beta))
                return false;
            if (beta <= tolerance) {
                last_iterations_ = total;
                return true;
            }

            const unsigned int m = std::min(restart_, max_iterations_ - total);
            if (m == 0)
                break;

            std::vector<Scalar> V((std::size_t)(m + 1) * n, Scalar(0));
            std::vector<Scalar> Z(flexible ? (std::size_t)(m + 1) * n : 0, Scalar(0));
            std::vector<Scalar> H((std::size_t)(m + 1) * m, Scalar(0));
            std::vector<Scalar> cs(m, Scalar(0));
            std::vector<Scalar> sn(m, Scalar(0));
            std::vector<Scalar> g(m + 1, Scalar(0));

            for (unsigned int i = 0; i < n; ++i)
                V[i] = r[i] / beta;
            g[0] = beta;

            unsigned int j = 0;
            for (; j < m && total < max_iterations_; ++j) {
                std::vector<Scalar> z(n, Scalar(0));
                if (!applyPreconditioner(&V[(std::size_t)j * n], z.data()))
                    return false;
                if (flexible)
                    std::copy(z.begin(), z.end(), Z.begin() + (std::size_t)j * n);

                std::vector<Scalar> w(n, Scalar(0));
                if (!applyOperator(z.data(), w.data()))
                    return false;

                for (unsigned int i = 0; i <= j; ++i) {
                    Scalar h = Scalar(0);
                    for (unsigned int k = 0; k < n; ++k)
                        h += w[k] * V[(std::size_t)i * n + k];
                    H[(std::size_t)i * m + j] = h;
                    for (unsigned int k = 0; k < n; ++k)
                        w[k] -= h * V[(std::size_t)i * n + k];
                }

                const Scalar hnext = norm(w);
                H[(std::size_t)(j + 1) * m + j] = hnext;
                if (hnext > std::numeric_limits<Scalar>::min()) {
                    for (unsigned int k = 0; k < n; ++k)
                        V[(std::size_t)(j + 1) * n + k] = w[k] / hnext;
                }

                for (unsigned int i = 0; i < j; ++i) {
                    const Scalar upper = H[(std::size_t)i * m + j];
                    const Scalar lower = H[(std::size_t)(i + 1) * m + j];
                    H[(std::size_t)i * m + j] = cs[i] * upper + sn[i] * lower;
                    H[(std::size_t)(i + 1) * m + j] = -sn[i] * upper + cs[i] * lower;
                }

                const Scalar a = H[(std::size_t)j * m + j];
                const Scalar c = H[(std::size_t)(j + 1) * m + j];
                const Scalar rr = std::sqrt(a * a + c * c);
                if (rr > std::numeric_limits<Scalar>::min()) {
                    cs[j] = a / rr;
                    sn[j] = c / rr;
                } else {
                    cs[j] = Scalar(1);
                    sn[j] = Scalar(0);
                }
                H[(std::size_t)j * m + j] = rr;
                H[(std::size_t)(j + 1) * m + j] = Scalar(0);
                g[j + 1] = -sn[j] * g[j];
                g[j] = cs[j] * g[j];

                ++total;
                if (std::fabs(g[j + 1]) <= tolerance) {
                    ++j;
                    break;
                }
            }

            if (j == 0)
                return false;

            std::vector<Scalar> y(j, Scalar(0));
            for (int i = (int)j - 1; i >= 0; --i) {
                Scalar s = g[(unsigned int)i];
                for (unsigned int k = (unsigned int)i + 1; k < j; ++k)
                    s -= H[(std::size_t)i * m + k] * y[k];
                const Scalar diagonal = H[(std::size_t)i * m + (std::size_t)i];
                if (diagonal == Scalar(0))
                    return false;
                y[(unsigned int)i] = s / diagonal;
            }

            if (flexible) {
                for (unsigned int i = 0; i < j; ++i)
                    for (unsigned int k = 0; k < n; ++k)
                        x[k] += y[i] * Z[(std::size_t)i * n + k];
            } else {
                std::vector<Scalar> u(n, Scalar(0));
                for (unsigned int i = 0; i < j; ++i)
                    for (unsigned int k = 0; k < n; ++k)
                        u[k] += y[i] * V[(std::size_t)i * n + k];
                std::vector<Scalar> z(n, Scalar(0));
                if (!applyPreconditioner(u.data(), z.data()))
                    return false;
                for (unsigned int k = 0; k < n; ++k)
                    x[k] += z[k];
            }

            std::vector<Scalar> Ax(n, Scalar(0));
            if (!applyOperator(x.data(), Ax.data()))
                return false;
            for (unsigned int k = 0; k < n; ++k)
                r[k] = b[k] - Ax[k];

            const Scalar residual = norm(r);
            if (!std::isfinite(residual))
                return false;
            if (residual <= tolerance) {
                last_iterations_ = total;
                return true;
            }
        }

        last_iterations_ = total;
        return false;
    }

    bool bicgstab(const std::vector<Scalar> &b, std::vector<Scalar> &x) {
        const unsigned int n = n_;
        x.assign(n, Scalar(0));

        const Scalar bnorm = norm(b);
        if (!std::isfinite(bnorm))
            return false;
        const Scalar tolerance = absolute_tolerance_ + relative_tolerance_ * bnorm;
        if (bnorm <= tolerance) {
            last_iterations_ = 0;
            return true;
        }

        std::vector<Scalar> r = b;
        std::vector<Scalar> rhat = r;
        std::vector<Scalar> v(n, Scalar(0));
        std::vector<Scalar> p(n, Scalar(0));
        std::vector<Scalar> phat(n, Scalar(0));
        std::vector<Scalar> s(n, Scalar(0));
        std::vector<Scalar> shat(n, Scalar(0));
        std::vector<Scalar> t(n, Scalar(0));

        Scalar rho = Scalar(1);
        Scalar alpha = Scalar(1);
        Scalar omega = Scalar(1);
        const Scalar tiny = std::numeric_limits<Scalar>::min();

        unsigned int total = 0;
        while (total < max_iterations_) {
            const Scalar rho_new = dot(rhat, r);
            if (!std::isfinite(rho_new) || std::fabs(rho_new) <= tiny)
                break;
            const Scalar beta = (rho_new / rho) * (alpha / omega);
            for (unsigned int i = 0; i < n; ++i)
                p[i] = r[i] + beta * (p[i] - omega * v[i]);

            if (!applyPreconditioner(p.data(), phat.data()))
                return false;
            if (!applyOperator(phat.data(), v.data()))
                return false;

            const Scalar denominator = dot(rhat, v);
            if (!std::isfinite(denominator) || std::fabs(denominator) <= tiny)
                break;
            alpha = rho_new / denominator;

            for (unsigned int i = 0; i < n; ++i)
                s[i] = r[i] - alpha * v[i];
            if (norm(s) <= tolerance) {
                for (unsigned int i = 0; i < n; ++i)
                    x[i] += alpha * phat[i];
                last_iterations_ = total + 1;
                return true;
            }

            if (!applyPreconditioner(s.data(), shat.data()))
                return false;
            if (!applyOperator(shat.data(), t.data()))
                return false;

            const Scalar tt = dot(t, t);
            if (!std::isfinite(tt) || tt <= tiny)
                break;
            omega = dot(t, s) / tt;

            for (unsigned int i = 0; i < n; ++i) {
                x[i] += alpha * phat[i] + omega * shat[i];
                r[i] = s[i] - omega * t[i];
            }
            rho = rho_new;
            ++total;

            if (norm(r) <= tolerance) {
                last_iterations_ = total;
                return true;
            }
        }

        last_iterations_ = total;
        return false;
    }

    void buildJacobi(const CscMatrix<Scalar> &P) {
        diagonal_.assign(n_, Scalar(0));
        for (unsigned int j = 0; j < n_; ++j) {
            const int begin = P.col_ptr[j];
            const int end = P.col_ptr[j + 1];
            for (int k = begin; k < end; ++k)
                if (P.row_idx[(std::size_t)k] == (int)j) {
                    diagonal_[j] = P.val[(std::size_t)k];
                    break;
                }
        }
        jacobi_valid_ = true;
        for (unsigned int i = 0; i < n_; ++i)
            if (!std::isfinite(diagonal_[i]) || diagonal_[i] == Scalar(0))
                jacobi_valid_ = false;
    }

    void ensureFallback() {
        if (!fallback_)
            fallback_.reset(new DenseLUSolver<Scalar>());
    }

    void ensureDirect() {
        if (!direct_)
            direct_.reset(new DenseLUSolver<Scalar>());
    }

    KrylovMethod method_;
    KrylovPreconditioner preconditioner_ = KrylovPreconditioner::Jacobi;
    Scalar relative_tolerance_ = Scalar(1e-12);
    Scalar absolute_tolerance_ = Scalar(0);
    unsigned int max_iterations_ = 200;
    unsigned int restart_ = 30;
    unsigned int n_ = 0;

    const MatrixFreeOperator<Scalar> *operator_ = nullptr;
    const CscMatrix<Scalar> *preconditioner_matrix_ = nullptr;
    std::vector<Scalar> diagonal_;
    bool jacobi_valid_ = false;
    std::vector<int> ip_;
    std::vector<long> scaling_D1_;
    std::vector<long> scaling_D2_;

    std::unique_ptr<LinearSolver<Scalar>> fallback_;
    std::unique_ptr<LinearSolver<Scalar>> direct_;

    unsigned int last_iterations_ = 0;
    bool last_converged_ = false;
    unsigned int fallbacks_ = 0;
};

template <typename Scalar> class GmresLinearSolver : public KrylovSolverBase<Scalar> {
  public:
    GmresLinearSolver() : KrylovSolverBase<Scalar>(KrylovMethod::Gmres) {
    }
};

template <typename Scalar> class FgmresLinearSolver : public KrylovSolverBase<Scalar> {
  public:
    FgmresLinearSolver() : KrylovSolverBase<Scalar>(KrylovMethod::Fgmres) {
    }
};

template <typename Scalar> class BicgstabLinearSolver : public KrylovSolverBase<Scalar> {
  public:
    BicgstabLinearSolver() : KrylovSolverBase<Scalar>(KrylovMethod::Bicgstab) {
    }
};

} // namespace yabi

#endif
