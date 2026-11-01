// T12g — Parameterized chemical network (6 species, 3 params) sensitivity (fwd/rev/sim)
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <limits>

int main() {
    std::cout << "=== T12g Parameterized network sensitivity (fwd/rev/sim) ===" << std::endl;
    constexpr unsigned N = 6, P = 3;
    // tape domain = N (states) + 1 (time) + P (params)
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(N + 1 + P);
    x[0]=1.0; x[1]=0.0; x[2]=0.0; x[3]=0.0; x[4]=0.0; x[5]=0.0;   // states
    x[6]=0.0;                                                       // time (unused)
    x[7]=0.04; x[8]=1.0e4; x[9]=3.0e7;                              // parameters
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(N);
    f[0] = -x[7]*x[0] + x[8]*x[1]*x[2];
    f[1] =  x[7]*x[0] - x[8]*x[1]*x[2] - x[9]*x[1]*x[1];
    f[2] =  x[9]*x[1]*x[1];
    f[3] =  x[9]*x[1]*x[1] - 0.1*x[3];
    f[4] =  0.1*x[3] - 5.0*x[4];
    f[5] =  5.0*x[4] - 0.01*x[5];
    rhs.Dependent(x, f); rhs.optimize();

    double y0[N] = {1.0,0.0,0.0,0.0,0.0,0.0};
    double p[P]  = {0.04, 1.0e4, 3.0e7};
    double abs_err[N] = {1e-10,1e-10,1e-10,1e-10,1e-10,1e-10};
    const double T = 100.0;
    bool ok = true;

    // Forward (state seed)
    {
        yabi::BDFIntegrator<double> it(rhs);
        double seed[N]; for (unsigned i=0;i<N;i++) seed[i]=0.0; seed[0]=1.0;
        double pseed[P] = {0.0,0.0,0.0};
        if (!it.resetIntegration(y0, 0.0, p, 0.1, 1e-10, abs_err, true, false, NULL,NULL,NULL,NULL,0) ||
            !it.performIntegration(T, 1000000) || !it.forward(1, seed, pseed, false)) { std::cerr << "FAIL fwd\n"; ok=false; }
        else { double dir[N]; it.getForwardSolutionVector(dir);
               for (unsigned i=0;i<N;i++) if (!std::isfinite(dir[i])) ok=false;
               std::cout << "FWD dir0=" << dir[0] << " dir5=" << dir[5] << std::endl; }
    }
    // Reverse
    {
        yabi::BDFIntegrator<double> it(rhs);
        double w[N]; for (unsigned i=0;i<N;i++) w[i]=0.0; w[0]=1.0;
        if (!it.resetIntegration(y0, 0.0, p, 0.1, 1e-10, abs_err, true, false, NULL,NULL,NULL,NULL,0) ||
            !it.performIntegration(T, 1000000) || !it.reverse(1, w)) { std::cerr << "FAIL rev\n"; ok=false; }
        else { double* adj_sp[N]; double* adj_p[P];
               for (unsigned i=0;i<N;i++) adj_sp[i]=new double[1];
               for (unsigned i=0;i<P;i++) adj_p[i]=new double[1];
               if (!it.getReverseSolutionVector(adj_sp, adj_p)) ok=false;
               else { for (unsigned i=0;i<N;i++) if (!std::isfinite(adj_sp[i][0])) ok=false;
                      for (unsigned i=0;i<P;i++) if (!std::isfinite(adj_p[i][0])) ok=false;
                      std::cout << "REV adj_sp0=" << adj_sp[0][0] << " adj_p0=" << adj_p[0][0] << std::endl; }
               for (unsigned i=0;i<N;i++) delete[] adj_sp[i];
               for (unsigned i=0;i<P;i++) delete[] adj_p[i]; }
    }
    // Simultaneous (ndir = N + P)
    {
        const unsigned ndir = N + P;
        std::vector<double> yseed(ndir*N, 0.0);
        std::vector<double> pseed(ndir*P, 0.0);
        std::vector<double*> yptr(ndir), pptr(ndir);
        std::vector<double> rel_arr(ndir, 1e-10);
        std::vector<std::array<double,N>> abs_arr(ndir);
        std::vector<double*> absptr(ndir);
        for (unsigned d=0; d<ndir; d++) { if (d<N) yseed[d*N+d]=1.0; else pseed[d*P+(d-N)]=1.0;
                                          yptr[d]=&yseed[d*N]; pptr[d]=&pseed[d*P];
                                          for (unsigned i=0;i<N;i++) abs_arr[d][i]=abs_err[i];
                                          absptr[d]=abs_arr[d].data(); }
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y0, 0.0, p, 0.1, 1e-10, abs_err, false, false, yptr.data(), pptr.data(), rel_arr.data(), absptr.data(), ndir) ||
            !it.performIntegration(T, 1000000)) { std::cerr << "FAIL sim\n"; ok=false; }
        else { std::vector<std::array<double,N>> dirs(ndir); std::vector<double*> dptr(ndir);
               for (unsigned d=0; d<ndir; d++) dptr[d]=dirs[d].data();
               if (!it.getSimultaniousDerivativesVector(dptr.data())) ok=false;
               else { for (unsigned d=0; d<ndir; d++) for (unsigned i=0;i<N;i++) if (!std::isfinite(dirs[d][i])) ok=false;
                      std::cout << "SIM d/dy0[0] dir0=" << dirs[0][0] << " d/dp[0] dir0=" << dirs[N][0] << std::endl; } }
    }
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
