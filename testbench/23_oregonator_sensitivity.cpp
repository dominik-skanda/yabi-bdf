// T12 Oregonator sensitivity
#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
int main() {
    std::cout << "=== T12 Oregonator sensitivity ===" << std::endl;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(3);
    x[0]=1.0; x[1]=1.0; x[2]=1.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(3);
    f[0]=77.27*(x[1]+x[0]*(1.0-8.375e-6*x[0]-x[1]));
    f[1]=(x[2]-(1.0+x[0])*x[1])/77.27;
    f[2]=0.161*(x[0]-x[2]);
    rhs.Dependent(x,f); rhs.optimize();
    yabi::BDFIntegrator<double> i(rhs);
    double y0[3]={1,1,1}; double abs_err[3]={1e-10,1e-10,1e-10};
    bool ok=i.resetIntegration(y0,0.0,NULL,0.1,1e-12,abs_err,true,false,NULL,NULL,NULL,NULL,0) &&
             i.performIntegration(1.0,100000) && i.forward(1,(double[]){1,0,0},(double[]){0,0,0},false);
    if(ok){double dir[3]; i.getForwardSolutionVector(dir); std::cout<<"PASS dir0="<<dir[0]<<std::endl;}
    else std::cout<<"FAIL"<<std::endl; return ok?0:1;
}
