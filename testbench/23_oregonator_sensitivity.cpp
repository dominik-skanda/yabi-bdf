// T12d — Oregonator sensitivity (fwd/rev/sim) — gate verification (t=1.0; t=5/10 blocked by T11d)
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <limits>

int main() {
    std::cout << "=== T12d Oregonator sensitivity (fwd/rev/sim) ===" << std::endl;
    constexpr unsigned N = 3;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(N);
    x[0] = 1.0; x[1] = 1.0; x[2] = 1.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(N);
    f[0] = 77.27*(x[1] + x[0]*(1.0 - 8.375e-6*x[0] - x[1]));
    f[1] = (x[2] - (1.0 + x[0])*x[1])/77.27;
    f[2] = 0.161*(x[0] - x[2]);
    rhs.Dependent(x, f); rhs.optimize();

    double y0[N] = {1.0, 1.0, 1.0};
    double abs_err[N] = {1e-10, 1e-10, 1e-10};
    const double T = 1.0;
    bool ok = true;

    // Forward
    {
        yabi::BDFIntegrator<double> it(rhs);
        double seed[N] = {1.0, 0.0, 0.0};
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-10, abs_err, true, false, NULL,NULL,NULL,NULL,0) ||
            !it.performIntegration(T, 1000000) || !it.forward(1, seed, NULL, false)) { std::cerr << "FAIL fwd\n"; ok=false; }
        else { double dir[N]; it.getForwardSolutionVector(dir);
               for (unsigned i=0;i<N;i++) if (!std::isfinite(dir[i])) ok=false;
               std::cout << "FWD dir0=" << dir[0] << " dir2=" << dir[2] << std::endl; }
    }
    // Reverse
    {
        yabi::BDFIntegrator<double> it(rhs);
        double w[N] = {1.0, 0.0, 0.0};
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-10, abs_err, true, false, NULL,NULL,NULL,NULL,0) ||
            !it.performIntegration(T, 1000000) || !it.reverse(1, w)) { std::cerr << "FAIL rev\n"; ok=false; }
        else { double* adj[N]; for (unsigned i=0;i<N;i++) adj[i]=new double[1];
               if (!it.getReverseSolutionVector(adj, NULL)) ok=false;
               else { for (unsigned i=0;i<N;i++) if (!std::isfinite(adj[i][0])) ok=false;
                      std::cout << "REV adj0=" << adj[0][0] << " adj2=" << adj[2][0] << std::endl; }
               for (unsigned i=0;i<N;i++) delete[] adj[i]; }
    }
    // Simultaneous (ndir=N)
    {
        const unsigned ndir = N;
        std::vector<double> yseed(ndir*N, 0.0);
        std::vector<double*> yptr(ndir);
        std::vector<double> rel_arr(ndir, 1e-10);
        std::vector<std::array<double,N>> abs_arr(ndir);
        std::vector<double*> absptr(ndir);
        for (unsigned d=0; d<ndir; d++) { yseed[d*N+d]=1.0; yptr[d]=&yseed[d*N];
                                          for (unsigned i=0;i<N;i++) abs_arr[d][i]=abs_err[i];
                                          absptr[d]=abs_arr[d].data(); }
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-10, abs_err, false, false, yptr.data(), NULL, rel_arr.data(), absptr.data(), ndir) ||
            !it.performIntegration(T, 1000000)) { std::cerr << "FAIL sim\n"; ok=false; }
        else { std::vector<std::array<double,N>> dirs(ndir); std::vector<double*> dptr(ndir);
               for (unsigned d=0; d<ndir; d++) dptr[d]=dirs[d].data();
               if (!it.getSimultaneousDerivativesVector(dptr.data())) ok=false;
               else { for (unsigned d=0; d<ndir; d++) for (unsigned i=0;i<N;i++) if (!std::isfinite(dirs[d][i])) ok=false;
                      std::cout << "SIM d/dy0[0] dir0=" << dirs[0][0] << std::endl; } }
    }
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
