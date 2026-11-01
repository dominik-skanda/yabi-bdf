#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <iostream>
#include <limits>
int main() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0]=1.0; x[1]=2.0; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2);
    f[0]=1.0 + x[0]*x[0]*x[1] - 4.0*x[0];
    f[1]=3.0*x[0] - x[0]*x[0]*x[1];
    rhs.Dependent(x,f); rhs.optimize();
    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[2]={1.0,2.0}, abs_err[2]={1e-10,1e-10}, y[2];
    if(!integrator.resetIntegration(y0,5.0,1e-10,abs_err,false,false,NULL,NULL,NULL,NULL,0)) return 1;
    if(!integrator.setAlgorithmSettings(5,std::numeric_limits<double>::epsilon(),true,1)) return 1;
    std::cout << "t,y1,y2" << std::endl;
    std::cout << "0," << y0[0] << "," << y0[1] << std::endl;
    for(double t : {0.01,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0}){
        if(!integrator.performIntegration(t,500000)) return 1;
        if(!integrator.getSolutionVector(y)) return 1;
        std::cout << t << "," << y[0] << "," << y[1] << std::endl;
    }
    return 0;
}
