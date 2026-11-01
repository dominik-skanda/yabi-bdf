#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
int main() {
    std::cout << "=== T12 HIRES sensitivity ===" << std::endl;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(8);
    x[0]=1; x[1]=0; x[2]=0; x[3]=0; x[4]=0; x[5]=0; x[6]=0; x[7]=0.0057;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(8);
    f[0]=-1.71*x[0]+0.43*x[1]+8.32*x[2]+0.0007;
    f[1]=1.71*x[0]-8.75*x[1];
    f[2]=-10.03*x[2]+0.43*x[3]+0.035*x[4];
    f[3]=8.32*x[1]+1.71*x[2]-1.12*x[3];
    f[4]=-1.745*x[4]+0.43*x[5]+0.43*x[6];
    f[5]=-280.0*x[5]*x[7]+0.69*x[3]+1.71*x[4]-0.43*x[5]+0.69*x[6];
    f[6]=280.0*x[5]*x[7]-1.81*x[6];
    f[7]=-280.0*x[5]*x[7]+1.81*x[6];
    rhs.Dependent(x,f); rhs.optimize();
    yabi::BDFIntegrator<double> i(rhs);
    double y0[8]={1,0,0,0,0,0,0,0.0057}; double abs_err[8]; for(int j=0;j<8;j++)abs_err[j]=1e-10;
    double seed[8]={1,0,0,0,0,0,0,0}; double pseed[8]={0,0,0,0,0,0,0,0};
    bool ok=i.resetIntegration(y0,0.0,NULL,0.1,1e-12,abs_err,true,false,NULL,NULL,NULL,NULL,0) &&
             i.performIntegration(1.0,100000) && i.forward(1,seed,pseed,false);
    if(ok){double dir[8]; i.getForwardSolutionVector(dir); std::cout<<"PASS dir0="<<dir[0]<<" dir7="<<dir[7]<<std::endl;}
    else std::cout<<"FAIL"<<std::endl; return ok?0:1;
}
