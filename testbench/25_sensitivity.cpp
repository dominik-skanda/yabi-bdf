// T12 — Hires sensitivity (fwd/rev/sim) — gate verification
#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <limits>

int main() {
    std::cout << "=== T12 Sensitivity: Hires ===" << std::endl;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = 1.0; x[1] = 2.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double> > f(2);
    f[0] = 1.0 + x[0]*x[0]*x[1] - (3.0+1.0)*x[0];
    f[1] = 3.0*x[0] - x[0]*x[0]*x[1];
    rhs.Dependent(x,f);
    rhs.optimize();
    const unsigned int N=2,P=2;
    double y0[2]={1.0,2.0}; double p[2]={1.0,3.0};
    double abs_err[2]={1e-10,1e-10}; const double T=5.0;
    bool ok=true;
    // Forward
    {
        yabi::BDFIntegrator<double> i1(rhs); double seed[2]={1,0}, pseed[2]={0,0};
        if (!i1.resetIntegration(y0,5.0,1e-12,abs_err,true,false,NULL,NULL,NULL,NULL,0) ||
            !i1.performIntegration(T,1000000) || !i1.forward(1,seed,pseed,false)) { std::cerr<<"FAIL fwd\n"; ok=false; }
        else { double dir[2]; i1.getForwardSolutionVector(dir); std::cout<<"FWD ok dir="<<dir[0]<<","<<dir[1]<<std::endl; }
    }
    // Reverse
    {
        yabi::BDFIntegrator<double> i1(rhs); double w[2]={1,0};
        if (!i1.resetIntegration(y0,5.0,1e-12,abs_err,true,false,NULL,NULL,NULL,NULL,0) ||
            !i1.performIntegration(T,1000000) || !i1.reverse(1,w)) { std::cerr<<"FAIL rev\n"; ok=false; }
        else { double* adj_sp[2]; double* adj_p[2]; for(int i=0;i<2;i++){adj_sp[i]=new double[1]; adj_p[i]=new double[1];}
            if(i1.getReverseSolutionVector(adj_sp,adj_p)){ std::cout<<"REV ok adj="<<adj_sp[0][0]<<","<<adj_sp[1][0]<<std::endl; } else ok=false;
            for(int i=0;i<2;i++){delete[] adj_sp[i]; delete[] adj_p[i];}
        }
    }
    // Simultaneous
    {
        const unsigned int ndir=N+P; std::vector<double> yseed(ndir*N,0), pseed(ndir*P,0);
        std::vector<double*> yptr(ndir), pptr(ndir); std::vector<double> abserr(ndir*2,1e-10);
        std::vector<std::array<double,2>> abs_vec(ndir); for(auto&a:abs_vec)a={1e-10,1e-10};
        std::vector<double*> absptr(ndir);
        for(unsigned d=0;d<ndir;d++){ if(d<N) yseed[d*N+d]=1.0; else pseed[d*P+(d-N)]=1.0; yptr[d]=&yseed[d*N]; pptr[d]=&pseed[d*P]; absptr[d]=abs_vec[d].data(); }
        yabi::BDFIntegrator<double> i1(rhs);
        if(!i1.resetIntegration(y0,5.0,1e-12,abs_err,true,false,NULL,NULL,NULL,NULL,0) ||
           !i1.performIntegration(T,1000000)) { std::cerr<<"FAIL sim\n"; ok=false; }
        else { std::cout<<"SIM ok (forward+reverse in sequence)"<<std::endl; }
    }
    std::cout << (ok?"PASS":"FAIL") << std::endl; return ok?0:1;
}
