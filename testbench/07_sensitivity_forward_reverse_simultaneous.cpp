/*
 * Test 7: All sensitivity modes on known problem
 */
#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>

int main() {
    std::cout << "=== Test 7: Sensitivities (fwd/rev/simultaneous) ===" << std::endl;
    
    CppAD::vector<CppAD::AD<double> > x(4); // y0,y1,t,k1,k2? wait - N=2,P=2 so domain = 2+1+2 = 5
    // Actually follow sensitivity_demo pattern
    x.resize(5);
    x[0]=1.0; x[1]=0.0; x[2]=0.0; x[3]=0.7; x[4]=0.3;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double> > f(2);
    f[0] = -x[3]*x[0];
    f[1] = x[3]*x[0] - x[4]*x[1];
    CppAD::ADFun<double> rhs;
    rhs.Dependent(x,f);
    rhs.optimize();
    
    const unsigned int N=2, P=2;
    double y0[2]={1.0,0.0};
    double p[2]={0.7,0.3};
    double abs_err[2]={1e-12,1e-12};
    const double T=2.0;
    bool ok = true;
    
    // Forward
    {
        yabi::BDFIntegrator<double> integrator(rhs);
        double seed[2]={1,0}, pseed[2]={0,0};
        if (!integrator.resetIntegration(y0,0.0,p,0.1,1e-12,abs_err,true,false,NULL,NULL,NULL,NULL,0) ||
            !integrator.performIntegration(T,1000000) ||
            !integrator.forward(1,seed,pseed,false)) {
            std::cerr << "FAIL fwd1\n"; ok=false;
        } else {
            double dir[2]; integrator.getForwardSolutionVector(dir);
            std::cout << "FWD dy/dy0[0]=[" << dir[0]<<","<<dir[1]<<"]" << std::endl;
        }
    }
    
    // Reverse
    {
        yabi::BDFIntegrator<double> integrator(rhs);
        double weight[2]={1,0};
        if (!integrator.resetIntegration(y0,0.0,p,0.1,1e-12,abs_err,true,false,NULL,NULL,NULL,NULL,0) ||
            !integrator.performIntegration(T,1000000) ||
            !integrator.reverse(1,weight)) {
            std::cerr << "FAIL rev1\n"; ok=false;
        } else {
            double* adj_sp[2]; double* adj_p[2];
            for(int i=0;i<2;i++) { adj_sp[i]=new double[1]; adj_p[i]=new double[1]; }
            if (integrator.getReverseSolutionVector(adj_sp,adj_p)) {
                std::cout << "REV lambda_y0=[" << adj_sp[0][0]<<","<<adj_sp[1][0]<<"]" << std::endl;
            } else { std::cerr << "FAIL rev get\n"; ok=false; }
            for(int i=0;i<2;i++) { delete[] adj_sp[i]; delete[] adj_p[i]; }
        }
    }
    
    // Simultaneous
    {
        const unsigned int ndir = N+P;
        std::vector<double> yseed_arr(ndir*N,0), pseed_arr(ndir*P,0);
        std::vector<double*> yseed_ptr(ndir), pseed_ptr(ndir), absptr(ndir);
        std::vector<std::array<double,2> > abs_err_arr(ndir);
        std::vector<double> rel_err_arr(ndir,1e-12);
        for(unsigned int d=0;d<ndir;d++) {
            if(d<N) yseed_arr[d*N+d]=1.0;
            else pseed_arr[d*P + (d-N)]=1.0;
            abs_err_arr[d]={1e-12,1e-12};
            yseed_ptr[d]=&yseed_arr[d*N];
            pseed_ptr[d]=&pseed_arr[d*P];
            absptr[d]=abs_err_arr[d].data();
        }
        yabi::BDFIntegrator<double> integrator(rhs);
        if (!integrator.resetIntegration(y0,0.0,p,0.1,1e-12,abs_err,false,false,yseed_ptr.data(),pseed_ptr.data(),rel_err_arr.data(),absptr.data(),ndir) ||
            !integrator.performIntegration(T,1000000)) {
            std::cerr << "FAIL sim int\n"; ok=false;
        } else {
            std::vector<std::array<double,2> > dirs(ndir);
            std::vector<double*> dirptr(ndir);
            for(unsigned int d=0;d<ndir;d++) dirptr[d]=dirs[d].data();
            if (!integrator.getSimultaniousDerivativesVector(dirptr.data())) { std::cerr << "FAIL sim get\n"; ok=false; }
            else {
                std::cout << "SIM d/dy0[0]=[" << dirs[0][0]<<","<<dirs[0][1]<<"] d/dp[0]=[" << dirs[2][0]<<","<<dirs[2][1]<<"]" << std::endl;
            }
        }
    }
    
    if (ok) { std::cout << "PASS" << std::endl; return 0; }
    else { std::cout << "FAIL" << std::endl; return 1; }
}
