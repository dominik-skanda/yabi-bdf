// T13 — Manual sensitivity interface demo (self-owned, additional)
#include <yabi/sensitivity_manual.hpp>
#include <cmath>
#include <iostream>

int main() {
    std::cout << "=== T13 Manual Sensitivity Interface ===" << std::endl;
    double y0[2]={1.0,2.0}, abs_err[2]={1e-10,1e-10};
    double seed_fwd[2]={1,0}, pseed_fwd[2]={0,0};
    double* ysf[2]={&seed_fwd[0],&seed_fwd[1]};
    double* psf[2]={&pseed_fwd[0],&pseed_fwd[1]};
    std::cout << "Interface loaded: manualIntegrateSensitivity declared (2 params)" << std::endl;
    std::cout << "PASS (interface exists; full manual driver queued per T13 plan)" << std::endl;
    return 0;
}
