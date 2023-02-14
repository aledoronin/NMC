/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of Neu(t)ralMC: a unified, power-efficient platform platform for photon transport simulations
//  accelerated by Metal/GPU computing and Machine Learning
//  created by Alexander Doronin
//  Source code:    https://github.com/aledoronin
//  Web:            http://www.lighttransport.net/
//  Licence:        BSD-3-Clause, see LICENCE file
//  Contributors:   the respective contributors, as shown by the AUTHORS file
//  Year conceived: 2023
//  This file:      definitions.h provides basic definitions used in Photon transport simulations
/*---------------------------------------------------------------------------------------------------------------------*/

#ifndef definitions_h
#define definitions_h

#define    PHOTON_TOTAL 10000000
#define    PHOTON_BATCH 25000
#define    MAX_SCATT    1000
#define    WEIGHT_MIN   10e-4
#define    RAY_DEPTH    10

#define mc_kernel_type 1 /* 0 - mcsub, 1 - mcxyz, 2 - neural mc, etc.*/

/* mcsub benchmark initialization */
#define BINS        101      /* number of bins, NZ and NR, for z and r */
/* number of file for saving */
#define Nfile 0       /* saves as mcOUTi.dat, where i = Nfile */
/* tissue parameters */
#define mua_mcsub  1.0      /* excitation absorption coeff. [cm^-1] */
#define mus_mcsub  100.0      /* excitation scattering coeff. [cm^-1] */
#define g_mcsub    0.90     /* excitation anisotropy [dimensionless] */
#define n1_mcsub   1.40     /* refractive index of medium */
#define n2_mcsub   1.40     /* refractive index outside medium */
/* beam parameters */
#define mcflag_mcsub 0      /* 0 = collimated, 1 = focused Gaussian,
                           2 >= isotropic pt. */
#define radius_mcsub 0.0    /* used if mcflag = 0 or 1 */
#define waist_mcsub  0.10 /* used if mcflag = 1 */
#define zfocus_mcsub 1.0    /* used if mcflag = 1 */
#define xs_mcsub     0.0    /* used if mcflag = 2, isotropic pt source */
#define ys_mcsub     0.0    /* used if mcflag = 2, isotropic pt source */
#define zs_mcsub     0.0    /* used if mcflag = 2, isotropic pt source, or mcflag = 0 collimated*/
#define boundaryflag_mcsub 1 /* 0 = infinite medium, 1 = air/tissue surface boundary*/

#define dr_mcsub     0.0020 /* radial bin size [cm] */
#define dz_mcsub     0.0020 /* depth bin size [cm] */
#define  PRINTOUT_mcsub = 1

/* mcxyz parameters */
#define Ntiss       19          /* Number of tissue types. */
#define STRLEN      32          /* String length. */
#define ls          1.0E-7      /* Moving photon a little bit off the voxel face */
#define PI          3.1415926
#define LIGHTSPEED  2.997925E10 /* in vacuo speed of light [cm/s] */
#define ALIVE       1           /* if photon not yet terminated */
#define DEAD        0            /* if photon is to be terminated */
#define THRESHOLD   0.01        /* used in roulette */
#define CHANCE      0.1          /* used in roulette */
#define MIN_VALUE   1E-4
#define MIN_WEIGHT  1E-12
#define SQR(x)      (x*x)
#define SIGN(x)     ((x)>=0.0 ? 1.0:-1.0)
#define COS90D      1.0E-6          /* If cos(theta) <= COS90D, theta >= PI/2 - 1e-6 rad. */
#define ONE_MINUS_COSZERO 1.0E-12   /* If 1-cos(theta) <= ONE_MINUS_COSZERO, fabs(theta) <= 1e-6 rad. */
#define MC_ZERO 0.0
#define MC_ONE 1.0
#define MC_TWO 2.0
#define RandomNum rng_gen.rand()

/* Run parameters */
struct RunParams
{
    char     myname[STRLEN];        // Holds the user's choice of myname, used in input and output files.
    /* launch parameters */
    int      mcflag, launchflag, boundaryflag;
    float    xfocus, yfocus, zfocus;
    float    ux0, uy0, uz0;
    float    radius;
    float    waist;

    /* mcxyz bin variables */
    float    dx, dy, dz;     /* bin size [cm] */
    int      Nx, Ny, Nz, Nt; /* # of bins */
    float    xs, ys, zs;        /* launch position */

    /* time */
    float    time_min;               // Requested time duration of computation.
};

struct TissueParams
{
    /* tissue parameters */
    float     muav[Ntiss];            // muav[0:Ntiss-1], absorption coefficient of ith tissue type
    float     musv[Ntiss];            // scattering coeff.
    float     gv[Ntiss];              // anisotropy of scattering
};

#endif /* definitions_h */
