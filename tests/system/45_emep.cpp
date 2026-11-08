/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T43 — EMEP problem (CWI "Test Set for IVP Solvers", emep): the EMEP MSC-W
// ozone chemistry model, 66 species, non-autonomous through diurnal photolysis
// (day/night switch at 04:00–20:00 local) and a time-dependent water-vapour
// concentration.  Transcribed verbatim from emep.f (feval + EMEPCF): 266 rate
// constants, 16 photolysis reactions, 66 right-hand sides; the Jacobian is
// obtained by CppAD.  Reference solution at t = 417600 s transcribed from
// emep.f solut (RADAU5, rtol = 1e-14, atol = 1e-8).

#include <yabi/bdf_integrator.hpp>

#include <cppad/cppad.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

constexpr double T = 298.0;         // temperature (K), fixed in EMEPCF
constexpr double M = 2.55e19, O2 = 5.2e18, XN2 = 1.99e19; // number densities
constexpr double HMIX = 1.2e5;      // mixing height (cm)
constexpr double RPATH3 = 0.65, RPATH4 = 0.35;
constexpr double kPi = 3.14159265358979323846;

// Photolysis A, B coefficients (emep.f EMEPCF DATA statements).
const double A[16] = {1.23e-3, 2.00e-4, 1.45e-2, 2.20e-5, 3.00e-6, 5.40e-5, 6.65e-5, 1.35e-5,
                      2.43e-5, 5.40e-4, 2.16e-4, 5.40e-5, 3.53e-2, 8.94e-2, 3.32e-5, 2.27e-5};
const double B[16] = {0.60, 1.40, 0.40, 0.75, 1.25, 0.79, 0.60, 0.94,
                      0.88, 0.79, 0.79, 0.79, 0.081, 0.059, 0.57, 0.62};

// Reference at t = 417600 s (emep.f solut), all 66 species.
const double ref[66] = {
    2.564580511140732e7,  5.146134770952715e10, 2.315679957701715e11, 1.130936599472892e13,
    3.459285326034955e13, 1.027236550974901e11, 8.508735586836855e10, 4.131285674624012e9,
    1.270937854717943e9,  5.632880890404914e9,  8.263821552730888e10, 3.293552658908353e10,
    1.054058186206315e11, 3.150308585365321e12, 2.488383936633755e7,  1.097565615411556e10,
    1.365196557629180e10, 3.852048230503094e11, 1.137462721600089e8,  2.850982205905218e10,
    3.786933619791445e11, 3.832384333863027e9,  4.804939902203071e6,  3.390546081092960e7,
    3.494452929972591e10, 1.490576370801779e7,  1.000567871868853e7,  1.856753061984312e9,
    2.272206156875160e6,  5.168375883757783e9,  2.317595952406701e6,  3.035139656996921e9,
    5.928979479102226e6,  2.515297290687841e9,  4.569678276203798e6,  3.727926507368739,
    4.063589349848112e4,  8.159151146269279e-38, 5.577590724412284e8,  7.684596616753747e8,
    5.338491777959816e9,  6.048604025328409e11, 4.222237216819787e7,  3.681852767009784e9,
    5.835731567978018e6,  1.456507159081862e9,  8.862868684014137e9,  9.338753312528582e8,
    3.595753586682656e9,  4.622427584057254e8,  2.395910064875511e8,  6.080936586253302e9,
    7.704108297342240e8,  3.386220860753221e9,  1.838264387030593e9,  1.973252346091884e9,
    1.313403940892729e6,  1.344471764571674e9,  2.568483117244701e6,  1.131854591900057e9,
    4.591537937818946e6,  8.609382282952802e8,  3.509560971283513e8,  6.682350971104786e8,
    3.207356403011160e8,  1.144987634882048e8};

// Constant rate coefficients (T = 298 fixed), 1-based RC[1..266]; entries that
// depend on time (36, 43-45, 46-53) are left zero and filled in the tape.
void rc_constants(double rc[267]) {
    for (int i = 0; i <= 266; ++i)
        rc[i] = 0.0;
    const double k60 = 4.2e-12 * std::exp(180.0 / T);
    rc[1] = 5.7e-34 * std::pow(T / 300.0, -2.8) * M * O2;
    rc[5] = 9.6e-32 * std::pow(T / 300.0, -1.6) * M;
    rc[7] = 2.0e-11 * std::exp(100.0 / T) * M;
    rc[8] = 2.2e-10;
    rc[11] = 1.8e-12 * std::exp(-1370.0 / T);
    rc[12] = 1.2e-13 * std::exp(-2450.0 / T);
    rc[13] = 1.9e-12 * std::exp(-1000.0 / T);
    rc[14] = 1.4e-14 * std::exp(-600.0 / T);
    rc[15] = 1.8e-11 * std::exp(110.0 / T);
    rc[17] = 3.7e-12 * std::exp(240.0 / T);
    rc[19] = 7.2e-14 * std::exp(-1414.0 / T);
    rc[20] = 1.4e-12;
    rc[21] = 1.4e-11;
    rc[26] = 4.1e-16;
    rc[29] = 7.1e14 * std::exp(-11080.0 / T);
    rc[30] = 4.8e-11 * std::exp(250.0 / T);
    rc[31] = 2.9e-12 * std::exp(-160.0 / T);
    rc[33] = 7.7e-12 * std::exp(-2100.0 / T);
    rc[35] = 1.0e-14 * std::exp(785.0 / T);
    rc[39] = 1.35e-12;
    rc[40] = 4.0e-17;
    rc[59] = 3.9e-12 * std::exp(-1885.0 / T);
    rc[60] = k60;
    rc[61] = 5.5e-14 * std::exp(365.0 / T);
    rc[62] = 5.5e-14 * std::exp(365.0 / T);
    rc[63] = 3.3e-12 * std::exp(-380.0 / T);
    rc[64] = 3.2e-12;
    rc[65] = 3.8e-13 * std::exp(780.0 / T);
    rc[66] = 9.6e-12;
    rc[67] = 1.0e-12 * std::exp(190.0 / T);
    rc[68] = 1.9e-12 * std::exp(190.0 / T);
    rc[69] = 5.8e-16;
    rc[70] = 2.4e-13;
    rc[71] = 7.8e-12 * std::exp(-1020.0 / T);
    rc[72] = 8.9e-12;
    rc[74] = 6.5e-13 * std::exp(650.0 / T);
    rc[75] = 5.6e-12 * std::exp(310.0 / T);
    rc[76] = 5.8e-12 * std::exp(190.0 / T);
    rc[77] = 1.0e-11;
    rc[78] = 1.34e16 * std::exp(-13330.0 / T);
    rc[79] = 2.0e-11;
    rc[80] = 1.1e-11;
    rc[81] = 1.64e-11 * std::exp(-559.0 / T);
    rc[83] = k60;
    rc[85] = 1.0e-11;
    rc[86] = 1.15e-12;
    rc[87] = 4.8e-12;
    rc[88] = 1.3e-13 * std::exp(1040.0 / T);
    rc[89] = 3.0e-13 * std::exp(1040.0 / T);
    rc[90] = 8.0e-13;
    rc[94] = 2.8e-12 * std::exp(530.0 / T);
    rc[105] = k60;
    rc[109] = 1.66e-12 * std::exp(474.0 / T);
    rc[110] = k60;
    rc[112] = 1.2e-14 * std::exp(-2630.0 / T);
    rc[123] = 6.5e-15 * std::exp(-1880.0 / T);
    rc[125] = 2.86e-11;
    rc[126] = k60;
    rc[146] = 3.2e-11;
    rc[147] = 2.0e-11;
    rc[148] = 2.2e-11;
    rc[149] = 3.7e-11;
    rc[150] = 12.3e-15 * std::exp(-2013.0 / T);
    rc[151] = 2.54e-11 * std::exp(410.0 / T);
    rc[152] = k60;
    rc[153] = 4.13e-12 * std::exp(452.0 / T);
    rc[154] = k60;
    rc[155] = rc[85];
    rc[156] = 2.0e-11;
    rc[157] = 8.0e-18;
    rc[158] = 1.86e-11 * std::exp(175.0 / T);
    rc[159] = k60;
    rc[160] = 4.32e-15 * std::exp(-2016.0 / T);
    rc[161] = 3.35e-11;
    rc[162] = k60;
    rc[163] = 7.8e-13;
    rc[164] = k60;
    rc[219] = 2.0e-11;
    rc[220] = k60;
    rc[221] = 1.1e-11;
    rc[222] = 1.7e-11;
    rc[223] = 2.4e-11;
    rc[234] = 1.37e-11;
    rc[235] = 1.7e-11;
    rc[236] = k60;
}

} // namespace

int main() {
    // Emissions (emep.f feval PARAMETER block).
    const double FRAC6 = 0.07689, FRAC7 = 0.41444, FRAC8 = 0.03642, FRAC9 = 0.03827,
                 FRAC10 = 0.24537, FRAC13 = 0.13957;
    const double VEKT6 = 30, VEKT7 = 58, VEKT8 = 28, VEKT9 = 42, VEKT10 = 106, VEKT13 = 46;
    const double VMHC = 1.0 / (FRAC6 / VEKT6 + FRAC7 / VEKT7 + FRAC8 / VEKT8 + FRAC9 / VEKT9 +
                               FRAC10 / VEKT10 + FRAC13 / VEKT13);
    const double EMNOX = 2.5e11, EMHC = 2.5e11;
    const double EMIS1 = EMNOX, EMIS2 = 0.0, EMIS3 = EMNOX, EMIS4 = EMHC * 10.0, EMIS5 = 0.0;
    const double EMIS6 = EMHC * FRAC6 / VEKT6 * VMHC;
    const double EMIS7 = EMHC * FRAC7 / VEKT7 * VMHC;
    const double EMIS8 = EMHC * FRAC8 / VEKT8 * VMHC;
    const double EMIS9 = EMHC * FRAC9 / VEKT9 * VMHC;
    const double EMIS10 = EMHC * FRAC10 / VEKT10 * VMHC;
    const double EMIS11 = 0.5 * EMHC, EMIS12 = 0.0;
    const double EMIS13 = EMHC * FRAC13 / VEKT13 * VMHC;

    // Constant part of the rate constants (doubles).
    double rc0[267];
    rc_constants(rc0);

    // Tape domain = 66 state + 1 time.
    CppAD::vector<CppAD::AD<double>> X(67);
    for (std::size_t i = 0; i < 66; ++i)
        X[i] = 0.0;
    X[66] = 0.0; // time
    CppAD::Independent(X);

    CppAD::AD<double> t = X[66];
    auto Y = [&](int i) -> const CppAD::AD<double> & { return X[i - 1]; };

    // ---- time-of-day and meteorology (EMEPCF) ----
    // Faithful transcription of the Fortran integer arithmetic
    //   ITIMEH = int(TIME/3600);  I24HRS = ITIMEH/24 + 1;
    //   TIMEOD = TIME/3600 - (I24HRS-1)*24  (hour of day, [0,24)).
    // TIMEOD = TIMEH mod 24, so the day index is floor(TIME/86400).  The interval
    // [14400, 417600] s gives q = TIME/86400 in [0.167, 4.833], so floor(q) is the
    // nested CondExpGe below (CppAD has no floor/mod for AD types; the chain is exact
    // on this interval and, with the segment restarts, resolves the diurnal cycle).
    CppAD::AD<double> TIMEH = t / 3600.0;
    CppAD::AD<double> q = t / 86400.0; // TIMEH/24
    CppAD::AD<double> kday = CppAD::CondExpGe(
        q, CppAD::AD<double>(4.0), CppAD::AD<double>(4.0),
        CppAD::CondExpGe(q, CppAD::AD<double>(3.0), CppAD::AD<double>(3.0),
                         CppAD::CondExpGe(q, CppAD::AD<double>(2.0), CppAD::AD<double>(2.0),
                                          CppAD::CondExpGe(q, CppAD::AD<double>(1.0),
                                                           CppAD::AD<double>(1.0),
                                                           CppAD::AD<double>(0.0)))));
    CppAD::AD<double> TIMEOD = TIMEH - 24.0 * kday; // hour of day [0,24)

    CppAD::AD<double> XLHA = (1.0 + TIMEOD / 12.0) * kPi;
    CppAD::AD<double> FI = 50.0 * kPi / 180.0;
    CppAD::AD<double> DEKL = 23.5 * kPi / 180.0;
    CppAD::AD<double> SEC =
        1.0 / (CppAD::cos(XLHA) * CppAD::cos(FI) * CppAD::cos(DEKL) +
               CppAD::sin(FI) * CppAD::sin(DEKL));

    CppAD::AD<double> day = CppAD::CondExpGe(
        TIMEOD, CppAD::AD<double>(4.0),
        CppAD::CondExpLt(TIMEOD, CppAD::AD<double>(20.0), CppAD::AD<double>(1.0),
                         CppAD::AD<double>(0.0)),
        CppAD::AD<double>(0.0));
    CppAD::AD<double> delta = CppAD::CondExpEq(day, CppAD::AD<double>(1.0),
                                               CppAD::AD<double>(1.0), CppAD::AD<double>(0.25));

    CppAD::AD<double> XQ = -7.93e-5 * TIMEOD * 3600.0 + 2.43;
    CppAD::AD<double> RH = 23.0 * CppAD::sin(XQ) + 66.5;
    const double XZ = (597.3 - 0.57 * (T - 273.16)) * 18.0 / 1.986 * (1.0 / T - 1.0 / 273.16);
    CppAD::AD<double> H2O = 6.1078 * std::exp(-XZ) * 10.0 / (1.38e-16 * T) * RH;

    // ---- photolysis rates DJ[1..16] ----
    CppAD::AD<double> DJ[17];
    for (int i = 1; i <= 16; ++i) {
        CppAD::AD<double> daytime = A[i - 1] * CppAD::exp(-B[i - 1] * SEC);
        DJ[i] = CppAD::CondExpEq(day, CppAD::AD<double>(1.0), daytime,
                                 CppAD::AD<double>(1e-40));
    }

    // ---- rate constants RC[1..266] ----
    CppAD::AD<double> RC[267];
    for (int i = 1; i <= 266; ++i)
        RC[i] = rc0[i];
    RC[36] = (2.3e-13 * std::exp(600.0 / T) + M * 1.7e-33 * std::exp(1000.0 / T)) *
             (1.0 + 1.4e-21 * H2O * std::exp(2200.0 / T));
    RC[43] = CppAD::CondExpGt(RH, CppAD::AD<double>(0.9), CppAD::AD<double>(1e-4),
                              CppAD::AD<double>(5e-6));
    RC[44] = RC[43];
    RC[45] = RC[43];
    RC[46] = 2.0 * delta / HMIX;
    RC[47] = 0.5 * delta / HMIX;
    RC[48] = 0.2 * delta / HMIX;
    RC[49] = 0.5 * delta / HMIX;
    RC[50] = 0.2 * delta / HMIX;
    RC[51] = 0.1 * delta / HMIX;
    RC[52] = 0.5 * delta / HMIX;
    RC[53] = 0.3 * delta / HMIX;

    // ---- right-hand sides f[1..66] (emep.f feval) ----
    CppAD::vector<CppAD::AD<double>> f(66);

    f[0] = DJ[3] * Y(2) + DJ[13] * Y(39) + RC[19] * Y(2) * Y(39) + EMIS1 / HMIX -
           (RC[5] * Y(36) + RC[11] * Y(14) + RC[17] * Y(15) + RC[72] * Y(23) +
            RC[79] * (Y(24) + Y(57)) + RC[15] * Y(39) +
            RC[60] * (Y(19) + Y(26) + Y(27) + Y(29) + Y(31) + Y(33) + Y(35) + Y(43) +
                      Y(45) + Y(59) + Y(61) + Y(60))) *
               Y(1);

    f[1] = Y(1) * (RC[5] * Y(36) + RC[11] * Y(14) + RC[17] * Y(15) + RC[72] * Y(23) +
                   RC[79] * (Y(24) + Y(57)) + 2.0 * RC[15] * Y(39)) +
           RC[60] * Y(1) *
               (Y(19) + Y(26) + Y(27) + Y(29) + Y(31) + Y(33) + Y(35) + Y(59)) +
           RC[60] * Y(1) * (0.86 * Y(43) + 1.9 * Y(61) + 1.1 * Y(60) + 0.95 * Y(45)) +
           DJ[14] * Y(39) + DJ[5] * Y(16) + DJ[15] * Y(40) + RC[29] * Y(40) +
           RC[78] * (Y(25) + Y(58)) -
           (DJ[3] + RC[12] * Y(14) + RC[20] * Y(39) + RC[21] * Y(37) + RC[48] +
            RC[77] * (Y(24) + Y(57))) *
               Y(2);

    f[2] = EMIS3 / HMIX - (RC[39] * Y(37) + RC[40] * Y(19) + RC[47]) * Y(3);

    f[3] = EMIS4 / HMIX +
           Y(37) * (RC[66] * Y(11) + 2.0 * RC[221] * Y(32) + RC[222] * Y(30)) +
           Y(14) * (0.44 * RC[112] * Y(8) + 0.4 * RC[123] * Y(9) + 0.05 * RC[160] * Y(44) +
                    0.05 * RC[150] * Y(41)) +
           Y(11) * (DJ[6] + DJ[7] + RC[69] * Y(39)) + DJ[8] * Y(12) + DJ[11] * Y(30) +
           2.0 * DJ[7] * Y(32) - RC[70] * Y(37) * Y(4);

    f[4] = EMIS5 / HMIX + 0.07 * RC[123] * Y(14) * Y(9) - RC[59] * Y(37) * Y(5);

    f[5] = EMIS6 / HMIX - RC[71] * Y(37) * Y(6);

    f[6] = EMIS7 / HMIX - RC[81] * Y(37) * Y(7);

    f[7] = EMIS8 / HMIX - (RC[109] * Y(37) + RC[112] * Y(14)) * Y(8);

    f[8] = EMIS9 / HMIX + 0.07 * RC[150] * Y(14) * Y(41) -
           (RC[123] * Y(14) + RC[125] * Y(37)) * Y(9);

    f[9] = EMIS10 / HMIX - RC[234] * Y(37) * Y(10);

    f[10] = Y(19) * (RC[60] * Y(1) + (2.0 * RC[61] + RC[62]) * Y(19) + RC[80] * Y(24) +
                     RC[40] * Y(3)) +
            Y(37) * (RC[63] * Y(46) + RC[67] * Y(22)) +
            Y(1) * RC[60] *
                (2.0 * Y(29) + Y(31) + 0.74 * Y(43) + 0.266 * Y(45) + 0.15 * Y(60)) +
            Y(14) * (0.5 * RC[123] * Y(9) + RC[112] * Y(8) + 0.7 * RC[157] * Y(56) +
                     0.8 * RC[160] * Y(44) + 0.8 * RC[150] * Y(41)) +
            2.0 * DJ[7] * Y(32) + DJ[16] * (Y(22) + 1.56 * Y(50) + Y(51)) -
            (RC[66] * Y(37) + DJ[6] + DJ[7] + RC[69] * Y(39) + RC[53]) * Y(11);

    f[11] = Y(1) * (RC[72] * Y(23) + RC[83] * Y(26) * RPATH4 + RC[105] * Y(27) +
                    RC[126] * Y(31) + 0.95 * RC[162] * Y(61) + 0.684 * RC[154] * Y(45)) +
            Y(37) * (RC[64] * Y(20) + RC[76] * Y(28) + RC[76] * Y(50)) +
            0.5 * RC[123] * Y(14) * Y(9) + 0.04 * RC[160] * Y(14) * Y(44) +
            DJ[16] * (Y(28) + 0.22 * Y(50) + 0.35 * Y(49) + Y(51) + Y(52)) -
            (DJ[8] + RC[75] * Y(37) + RC[53]) * Y(12);

    f[12] = RC[83] * Y(1) * Y(26) * RPATH3 + (0.65 * DJ[16] + RC[76] * Y(37)) * Y(49) +
            RC[76] * Y(37) * Y(51) + (RC[159] * Y(1) + DJ[16]) * Y(59) +
            0.95 * RC[162] * Y(1) * Y(61) - (DJ[9] + RC[86] * Y(37) + RC[53]) * Y(13);

    f[13] = RC[1] * Y(36) + RC[89] * Y(15) * Y(24) -
            (RC[11] * Y(1) + RC[12] * Y(2) + RC[13] * Y(37) + RC[14] * Y(15) + RC[49] +
             RC[112] * Y(8) + RC[123] * Y(9) + RC[157] * Y(56) + RC[160] * Y(44) +
             RC[150] * Y(41) + DJ[1] + DJ[2]) *
                Y(14);

    CppAD::AD<double> s1 = Y(37) * (RC[13] * Y(14) + RC[31] * Y(17) + RC[33] * Y(18) +
                                    RC[39] * Y(3) + RC[63] * Y(46) + RC[64] * Y(20) +
                                    RC[66] * Y(11) + RC[70] * Y(4) + RC[221] * Y(32)) +
                           Y(19) * (RC[40] * Y(3) + 2.0 * RC[61] * Y(19) + 0.5 * RC[80] * Y(24)) +
                           DJ[11] * Y(30) +
                           Y(1) * RC[60] *
                               (Y(19) + Y(29) + Y(31) + Y(33) + Y(35) + 0.95 * Y(45) +
                                Y(26) * RPATH3 + 0.78 * Y(43) + Y(59) + 0.05 * Y(61) + 0.8 * Y(60)) +
                           RC[72] * Y(1) * Y(23) + DJ[8] * Y(12);

    f[14] = s1 + 2.0 * DJ[6] * Y(11) +
            DJ[16] * (Y(22) + Y(28) + 0.65 * Y(49) + Y(50) + Y(51) + Y(48) + Y(53)) +
            Y(39) * (RC[26] * Y(17) + RC[69] * Y(11)) +
            Y(14) * (0.12 * RC[112] * Y(8) + 0.28 * RC[123] * Y(9) + 0.06 * RC[160] * Y(44)) +
            0.06 * RC[150] * Y(14) * Y(41) -
            (RC[14] * Y(14) + RC[17] * Y(1) + RC[30] * Y(37) + 2.0 * RC[36] * Y(15) +
             RC[65] * Y(19) + RC[74] * Y(23) + (RC[88] + RC[89]) * Y(24) +
             RC[85] * (Y(26) + Y(29) + Y(31) + Y(27) + Y(57) + Y(45) + Y(61) + Y(59) +
                       Y(33) + Y(35) + Y(43) + Y(60))) *
                Y(15);

    f[15] = RC[21] * Y(2) * Y(37) + Y(39) * (RC[26] * Y(17) + RC[69] * Y(11)) -
            (RC[35] * Y(37) + DJ[5] + RC[45]) * Y(16);

    f[16] = RC[36] * Y(15) * Y(15) -
            (RC[31] * Y(37) + DJ[4] + RC[43] + RC[26] * Y(39) + RC[47]) * Y(17);

    f[17] = DJ[7] * Y(11) + Y(14) * (0.13 * RC[112] * Y(8) + 0.07 * RC[123] * Y(9)) -
            RC[33] * Y(37) * Y(18);

    f[18] = Y(37) * (RC[59] * Y(5) + RC[68] * Y(22)) + Y(24) * (RC[79] * Y(1) + 2.0 * RC[94] * Y(24)) +
            DJ[8] * Y(12) + DJ[16] * Y(47) + 0.31 * RC[123] * Y(14) * Y(9) -
            (RC[40] * Y(3) + RC[60] * Y(1) + 2.0 * RC[61] * Y(19) + 2.0 * RC[62] * Y(19) +
             RC[65] * Y(15) + 0.5 * RC[80] * Y(24)) *
                Y(19);

    f[19] = EMIS13 / HMIX - RC[64] * Y(37) * Y(20);

    f[20] = (RC[40] * Y(19) + RC[39] * Y(37)) * Y(3) + 0.05 * EMIS3 / HMIX - RC[51] * Y(21);

    f[21] = RC[65] * Y(15) * Y(19) - (RC[43] + DJ[16] + (RC[67] + RC[68]) * Y(37)) * Y(22);

    f[22] = Y(37) * (RC[71] * Y(6) + RC[68] * Y(28)) + 0.35 * DJ[16] * Y(49) +
            RC[83] * Y(1) * Y(26) * RPATH4 + DJ[9] * Y(13) -
            (RC[72] * Y(1) + RC[74] * Y(15)) * Y(23);

    f[23] = Y(37) * (RC[75] * Y(12) + RC[222] * Y(30) + RC[68] * Y(47)) +
            RC[105] * Y(1) * Y(27) + RC[78] * Y(25) + DJ[11] * Y(30) + DJ[9] * Y(13) +
            DJ[16] * Y(52) + 0.684 * RC[154] * Y(1) * Y(45) -
            (RC[77] * Y(2) + RC[79] * Y(1) + RC[80] * Y(19) + 2.0 * RC[94] * Y(24) +
             (RC[88] + RC[89]) * Y(15)) *
                Y(24);

    f[24] = RC[77] * Y(24) * Y(2) - (RC[50] + RC[78]) * Y(25);

    f[25] = Y(37) * (RC[81] * Y(7) + RC[68] * Y(49)) - (RC[83] * Y(1) + RC[85] * Y(15)) * Y(26);

    f[26] = Y(37) * (RC[86] * Y(13) + RC[87] * Y(52)) - (RC[105] * Y(1) + RC[85] * Y(15)) * Y(27);

    f[27] = RC[74] * Y(15) * Y(23) - ((RC[76] + RC[68]) * Y(37) + DJ[16] + RC[52]) * Y(28);

    f[28] = Y(37) * (RC[109] * Y(8) + RC[68] * Y(50)) - (RC[110] * Y(1) + RC[85] * Y(15)) * Y(29);

    f[29] = RC[236] * Y(1) * Y(33) + RC[220] * Y(1) * Y(35) + 0.266 * RC[154] * Y(1) * Y(45) +
            0.82 * RC[160] * Y(14) * Y(44) + DJ[16] * (Y(48) + Y(53)) -
            (DJ[11] + RC[222] * Y(37)) * Y(30);

    f[30] = Y(37) * (RC[125] * Y(9) + RC[68] * Y(51)) - (RC[126] * Y(1) + RC[85] * Y(15)) * Y(31);

    f[31] = RC[220] * Y(1) * Y(35) + DJ[16] * Y(53) - (2.0 * DJ[7] + RC[221] * Y(37)) * Y(32);

    f[32] = Y(37) * (RC[234] * Y(10) + RC[235] * Y(48)) - (RC[236] * Y(1) + RC[85] * Y(15)) * Y(33);

    f[33] = RC[236] * Y(1) * Y(33) + DJ[16] * Y(48) - RC[219] * Y(37) * Y(34);

    f[34] = Y(37) * (RC[219] * Y(34) + RC[223] * Y(53)) - (RC[220] * Y(1) + RC[85] * Y(15)) * Y(35);

    f[35] = DJ[1] * Y(14) + DJ[3] * Y(2) + DJ[14] * Y(39) + RC[7] * Y(38) +
            0.2 * RC[160] * Y(14) * Y(44) + 0.3 * RC[150] * Y(14) * Y(41) -
            (RC[1] + RC[5] * Y(1)) * Y(36);

    CppAD::AD<double> s1b = 2.0 * RC[8] * H2O * Y(38) + Y(15) * (RC[14] * Y(14) + RC[17] * Y(1)) +
                            2.0 * DJ[4] * Y(17) + DJ[5] * Y(16);
    CppAD::AD<double> s2 =
        s1b + DJ[16] * (Y(22) + Y(28) + Y(47) + Y(49) + Y(50) + Y(52) + Y(48) + Y(53));
    CppAD::AD<double> s3 = s2 + Y(14) * (0.15 * RC[123] * Y(9) + 0.08 * RC[160] * Y(44));
    CppAD::AD<double> s10 =
        RC[222] * Y(30) + RC[75] * Y(12) + RC[81] * Y(7) + RC[87] * Y(52) + RC[86] * Y(13) +
        RC[235] * Y(48) + RC[109] * Y(8) + RC[125] * Y(9) + RC[234] * Y(10) + RC[223] * Y(53) +
        RC[219] * Y(34) + RC[31] * Y(17) + RC[21] * Y(2) + RC[148] * Y(62) + RC[64] * Y(20) +
        RC[39] * Y(3) + RC[71] * Y(6) + RC[30] * Y(15) + RC[59] * Y(5) + RC[70] * Y(4) +
        RC[35] * Y(16) + RC[13] * Y(14) + RC[221] * Y(32) +
        RC[68] * (Y(22) + Y(28) + Y(47) + Y(50) + Y(51) + Y(49)) + RC[66] * Y(11) +
        RC[151] * Y(41) + RC[153] * Y(44) + RC[63] * Y(46) + RC[33] * Y(18) + RC[158] * Y(54) +
        RC[146] * Y(63) + RC[149] * (Y(65) + Y(66)) + RC[147] * Y(64) + RC[161] * Y(55);
    f[36] = s3 + 0.55 * RC[150] * Y(14) * Y(41) - s10 * Y(37);

    f[37] = DJ[2] * Y(14) - (RC[7] + RC[8] * H2O) * Y(38);

    f[38] = (RC[29] + DJ[15]) * Y(40) + RC[12] * Y(14) * Y(2) + RC[35] * Y(37) * Y(16) -
            (RC[15] * Y(1) + RC[26] * Y(17) + RC[163] * Y(41) + RC[19] * Y(2) + RC[20] * Y(2) +
             DJ[13] + DJ[14] + RC[69] * Y(11)) *
                Y(39);

    f[39] = RC[20] * Y(39) * Y(2) - (RC[29] + DJ[15] + RC[45]) * Y(40);

    f[40] = EMIS11 / HMIX - (RC[151] * Y(37) + RC[163] * Y(39) + RC[150] * Y(14)) * Y(41);

    f[41] = RC[45] * Y(16) + 2.0 * RC[44] * Y(40) - RC[51] * Y(42);

    f[42] = Y(37) * (RC[151] * Y(41) + RC[156] * Y(56)) + 0.12 * RC[152] * Y(1) * Y(43) -
            (RC[152] * Y(1) + RC[155] * Y(15)) * Y(43);

    f[43] = RC[60] * Y(1) * (0.42 * Y(43) + 0.05 * Y(60)) + 0.26 * RC[150] * Y(14) * Y(41) -
            (RC[153] * Y(37) + RC[160] * Y(14)) * Y(44);

    f[44] = RC[153] * Y(44) * Y(37) + RC[148] * Y(37) * Y(62) -
            (RC[154] * Y(1) + RC[85] * Y(15)) * Y(45);

    f[45] = RC[62] * Y(19) * Y(19) - RC[63] * Y(37) * Y(46);

    f[46] = RC[88] * Y(15) * Y(24) - (RC[68] * Y(37) + DJ[16] + RC[52]) * Y(47);

    f[47] = RC[85] * Y(15) * Y(33) - (RC[235] * Y(37) + DJ[16] + RC[52]) * Y(48);

    f[48] = RC[85] * Y(15) * Y(26) - ((RC[76] + RC[68]) * Y(37) + DJ[16] + RC[52]) * Y(49);

    f[49] = RC[85] * Y(15) * Y(29) - ((RC[76] + RC[68]) * Y(37) + DJ[16] + RC[52]) * Y(50);

    f[50] = RC[85] * Y(15) * Y(31) - ((RC[76] + RC[68]) * Y(37) + DJ[16] + RC[52]) * Y(51);

    f[51] = RC[85] * Y(15) * Y(27) - (RC[87] * Y(37) + DJ[16] + RC[52]) * Y(52);

    f[52] = RC[85] * Y(15) * Y(35) - (RC[223] * Y(37) + DJ[16] + RC[52]) * Y(53);

    f[53] = RC[60] * Y(1) * (0.32 * Y(43) + 0.1 * Y(60)) + 0.67 * RC[150] * Y(14) * Y(41) -
            RC[158] * Y(37) * Y(54);

    f[54] = RC[60] * Y(1) * (0.14 * Y(43) + 0.05 * Y(45) + 0.85 * Y(60)) - RC[161] * Y(37) * Y(55);

    f[55] = RC[155] * Y(15) * Y(43) - (RC[156] * Y(37) + RC[157] * Y(14) + RC[52]) * Y(56);

    f[56] = 0.5 * RC[158] * Y(37) * Y(54) + RC[78] * Y(58) + RC[149] * Y(37) * Y(66) -
            (RC[77] * Y(2) + RC[79] * Y(1) + RC[85] * Y(15)) * Y(57);

    f[57] = RC[77] * Y(57) * Y(2) - (RC[50] + RC[78]) * Y(58);

    f[58] = RC[79] * Y(1) * Y(57) + RC[146] * Y(37) * Y(63) -
            (RC[159] * Y(1) + RC[85] * Y(15)) * Y(59);

    f[59] = RC[163] * Y(39) * Y(41) + RC[147] * Y(37) * Y(64) -
            (RC[164] * Y(1) + RC[85] * Y(15)) * Y(60);

    f[60] = RC[161] * Y(37) * Y(55) + RC[149] * Y(37) * Y(65) -
            (RC[162] * Y(1) + RC[85] * Y(15)) * Y(61);

    f[61] = RC[85] * Y(15) * Y(45) - (RC[148] * Y(37) + RC[52]) * Y(62);

    f[62] = RC[85] * Y(15) * Y(59) - (RC[146] * Y(37) + RC[52]) * Y(63);

    f[63] = RC[85] * Y(15) * Y(60) - (RC[147] * Y(37) + RC[52]) * Y(64);

    f[64] = RC[85] * Y(15) * Y(61) - (RC[149] * Y(37) + RC[52]) * Y(65);

    f[65] = RC[85] * Y(15) * Y(57) - (RC[149] * Y(37) + RC[52]) * Y(66);

    CppAD::ADFun<double> fun;
    fun.Dependent(X, f);
    fun.optimize();

    // Initial conditions (emep.f init), 0-based state.
    std::vector<double> y0(66, 100.0);
    for (int i = 0; i < 13; ++i)
        y0[i] = 1.0e7;
    y0[0] = 1.0e9;
    y0[1] = 5.0e9;
    y0[2] = 5.0e9;
    y0[3] = 3.8e12;
    y0[4] = 3.5e13;
    y0[13] = 5.0e11;
    y0[37] = 1.0e-3;

    yabi::BDFIntegrator<double> it(fun);
    std::vector<double> atol(66, 1.0e-6);

    // EMEP is extremely stiff: the species span ~1e-38..1e14 (a stiffness ratio
    // of ~1e14).  Instrumentation (T45) showed two interacting difficulties:
    //   1. the Newton matrix P = I - (h/l)J is ill-conditioned at the dawn/dusk
    //      transient, so the Newton iteration converges only *linearly* (residual
    //      ~0.85x per iteration) — and the conditioning worsens with BDF order;
    //   2. at the higher orders the order-reduction heuristic (drop order when
    //      the order-1 error estimate is better) misfires, because its order-1
    //      estimate is unreliable at this stiffness, and the controller oscillates
    //      order and stalls.
    // Both are resolved by capping the order at 4 (setFixedOrder): order 4 is
    // A(alpha)-stable with alpha ~73 deg (order 5 ~52 deg, order 6 ~18 deg) and
    // its Newton matrix is better conditioned.  This is the standard stiff-ODE
    // remedy (Hairer & Wanner §V).
    it.setNewtonSettings(-1.0, 0.01, 0.30, 25, 25, true); // fresh J each iteration
    it.setNewtonLineSearch(true, 6); // damped Newton: backtrack on residual overshoot
    it.setFixedOrder(4);

    // The photolysis switches (dawn 04:00 / dusk 20:00 local) are discontinuities
    // in the RHS; the CWI test set marks them as NDISC=8 breakpoints.  Integrate
    // segment-by-segment, restarting the BDF at each switch, exactly as RADAU5's
    // driver does.
    std::vector<double> y(66);
    y = y0;
    const double tend[9] = {72000.0,  100800.0, 158400.0, 187200.0, 244800.0,
                            273600.0, 331200.0, 360000.0, 417600.0};
    double t_cur = 14400.0;
    for (int seg = 0; seg < 9; ++seg) {
        if (!it.resetIntegration(y.data(), t_cur, 600.0, 1e-8, atol.data(), false, false, NULL,
                                 NULL, NULL, NULL, 0) ||
            !it.setAlgorithmSettings(4, 1e-16, true, 1) ||
            !it.performIntegration(tend[seg], 10000000)) {
            std::cerr << "FAIL: EMEP integration did not converge (segment " << seg << ")\n";
            return 1;
        }
        if (!it.getSolutionVector(y.data())) {
            std::cerr << "FAIL: EMEP getSolutionVector\n";
            return 1;
        }
        t_cur = tend[seg];
    }

    double max_rel = 0.0;
    for (std::size_t i = 0; i < 66; ++i)
        max_rel = std::max(max_rel, std::fabs(y[i] - ref[i]) / (1.0 + std::fabs(ref[i])));
    std::cout << "EMEP max relative error vs CWI reference = " << max_rel << "\n";

    // With the fixed order-4 config the max relative error is ~4e-6 (dominated by
    // the quasi-steady trace species H2); a 1e-4 gate is a comfortable agreement.
    if (max_rel < 1e-4) {
        std::cout << "PASS: EMEP matches the CWI reference solution\n";
        return 0;
    }
    std::cerr << "FAIL: EMEP diverged from the CWI reference\n";
    return 1;
}
