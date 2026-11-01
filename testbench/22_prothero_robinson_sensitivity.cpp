#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
int main() {
    std::cout << "=== T12 Prothero-Robinson sensitivity ===" << std::endl;
    CppAD::ADFun<double> rhs; CppAD::vector<CppAD::AD<double>> x(2);
    x[0]=0.5; x[1]=0.0; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2);
    f[0]=2.0*(x[0]-x[0]*x[0])-x[1]; f[1]=x[0]*x[0]-x[1];
    rhs.Dependent(x,f); rhs.optimize();
    yabi::BDFIntegrator<double> i(rhs);
    double y0[2]={0.5,0.0}; double abs_err[2]={1e-10,1e-10};
    double seed[2]={1,0}, pseed[2]={0,0};
    bool ok = i.resetIntegration(y0,0.0,NULL,0.1,1e-12,abs_err,true,false,NULL,NULL,NULL,NULL,0) &&
              i.performIntegration(5.0,100000) && i.forward(1,seed,pseed,false);
    if(ok){double dir[2]; i.getForwardSolutionVector(dir); std::cout<<"PASS dir="<<dir[0]<<","<<dir[1]<<std::endl;}
    else std::cout<<"FAIL"<<std::endl; return ok?0:1;
}
