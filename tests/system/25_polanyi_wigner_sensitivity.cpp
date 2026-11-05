// T12f — Polanyi-Wigner (n=20 diffusion+decay) sensitivity (fwd/rev/sim) — gate verification
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <limits>

int main() {
    std::cout << "=== T12f Polanyi-Wigner n=20 sensitivity (fwd/rev/sim) ===" << std::endl;
    constexpr unsigned N = 20;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(N);
    for (unsigned i=0;i<N;i++) x[i] = 1.0 + 0.01*i;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(N);
    for (unsigned i=0;i<N;i++) {
        CppAD::AD<double> left  = (i>0)   ? x[i-1] : CppAD::AD<double>(0.0);
        CppAD::AD<double> right = (i<N-1) ? x[i+1] : CppAD::AD<double>(0.0);
        f[i] = 0.01*(left - 2.0*x[i] + right) - 0.1*x[i];
    }
    rhs.Dependent(x, f); rhs.optimize();

    double y0[N]; for (unsigned i=0;i<N;i++) y0[i]=1.0+0.01*i;
    double abs_err[N]; for (unsigned i=0;i<N;i++) abs_err[i]=1e-6;
    const double T = 10.0;
    bool ok = true;

    // Forward
    {
        yabi::BDFIntegrator<double> it(rhs);
        double seed[N]; for (unsigned i=0;i<N;i++) seed[i]=0.0; seed[0]=1.0;
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-6, abs_err, true, false, NULL,NULL,NULL,NULL,0) ||
            !it.performIntegration(T, 1000000) || !it.forward(1, seed, NULL, false)) { std::cerr << "FAIL fwd\n"; ok=false; }
        else { double dir[N]; it.getForwardSolutionVector(dir);
               for (unsigned i=0;i<N;i++) if (!std::isfinite(dir[i])) ok=false;
               std::cout << "FWD dir0=" << dir[0] << " dir19=" << dir[19] << std::endl; }
    }
    // Reverse
    {
        yabi::BDFIntegrator<double> it(rhs);
        double w[N]; for (unsigned i=0;i<N;i++) w[i]=0.0; w[0]=1.0;
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-6, abs_err, true, false, NULL,NULL,NULL,NULL,0) ||
            !it.performIntegration(T, 1000000) || !it.reverse(1, w)) { std::cerr << "FAIL rev\n"; ok=false; }
        else { double* adj[N]; for (unsigned i=0;i<N;i++) adj[i]=new double[1];
               if (!it.getReverseSolutionVector(adj, NULL)) ok=false;
               else { for (unsigned i=0;i<N;i++) if (!std::isfinite(adj[i][0])) ok=false;
                      std::cout << "REV adj0=" << adj[0][0] << " adj19=" << adj[19][0] << std::endl; }
               for (unsigned i=0;i<N;i++) delete[] adj[i]; }
    }
    // Simultaneous (ndir=N)
    {
        const unsigned ndir = N;
        std::vector<double> yseed(ndir*N, 0.0);
        std::vector<double*> yptr(ndir);
        std::vector<double> rel_arr(ndir, 1e-6);
        std::vector<std::array<double,N>> abs_arr(ndir);
        std::vector<double*> absptr(ndir);
        for (unsigned d=0; d<ndir; d++) { yseed[d*N+d]=1.0; yptr[d]=&yseed[d*N];
                                          for (unsigned i=0;i<N;i++) abs_arr[d][i]=abs_err[i];
                                          absptr[d]=abs_arr[d].data(); }
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-6, abs_err, false, false, yptr.data(), NULL, rel_arr.data(), absptr.data(), ndir) ||
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
