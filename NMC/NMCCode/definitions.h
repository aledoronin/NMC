/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of NMC: a unified, power-efficient platform platform for photon transport simulations
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

#define    PHOTON_TOTAL 100000
#define    PHOTON_BATCH 10000
#define    MAX_SPECKLES_PER_PIXEL 1000
#define    MAX_SCATT    10000
#define    WEIGHT_MIN   10e-4
#define    RAY_DEPTH    100
#define    WEIGHT_0     1.0
#define    NANGLES    1000
#define    MAXLAYERS 100
#define    MAX_SCATT_POL 150
#define    POL_CHANNELS 4

#define    FIRST_MEDIUM_LAYER  1

/* mcsub benchmark initialization */
#define BINS        101      /* number of bins, NZ and NR, for z and r */
/* number of file for saving */
#define Nfile 0       /* saves as mcOUTi.dat, where i = Nfile */
/* tissue parameters */
//#define mua_mcsub  10.0      /* excitation absorption coeff. [cm^-1] */
//#define mus_mcsub  90.0      /* excitation scattering coeff. [cm^-1] */
//#define g_mcsub    0.0     /* excitation anisotropy [dimensionless] */
//#define n1_mcsub   1.5     /* refractive index of medium */
//#define n2_mcsub   1.0     /* refractive index outside medium */
/* beam parameters */
#define mcflag_mcsub 0      /* 0 = collimated, 1 = focused Gaussian,
  //                         2 >= isotropic pt. */
#define radius_mcsub 0.0    /* used if mcflag = 0 or 1 */
#define waist_mcsub  0.10 /* used if mcflag = 1 */
#define zfocus_mcsub 1.0    /* used if mcflag = 1 */
#define xs_mcsub     0.0    /* used if mcflag = 2, isotropic pt source */
#define ys_mcsub     0.0    /* used if mcflag = 2, isotropic pt source */
#define zs_mcsub     0.0    /* used if mcflag = 2, isotropic pt source, or mcflag = 0 collimated*/
#define boundaryflag_mcsub 1 /* 0 = infinite medium, 1 = air/tissue surface boundary*/
#define FIRST_TISSUE_LAYER_MCSUB 1 /* the ID of the first tissue layer for mssub*/

#define dr_mcsub     0.0020 /* radial bin size [cm] */
#define dz_mcsub     0.0020 /* depth bin size [cm] */
//#define  PRINTOUT_mcsub = 1

/* mcxyz parameters */
#define Ntiss       19          /* Number of tissue types. */
#define STRLEN      32          /* String length. */
#define ls          10E-8       /* Moving photon a little bit off the voxel face */
#define PI          3.1415926
#define LIGHTSPEED  2.997925E11 /* in vacuo speed of light [mm/s] */
#define PHOTON_MAX_TIME 5E-5    /* 50 picoseconds */
//#define PHOTON_MAX_TIME 1E-9 /* 1 nanosecond */
#define ALIVE       1           /* if photon not yet terminated */
#define DEAD        0            /* if photon is to be terminated */
#define THRESHOLD   0.0001       /* used in roulette */
#define CHANCE      0.1          /* used in roulette */
#define MIN_VALUE   1E-6
#define MIN_WEIGHT  1E-12
#define SQR(x)      (x*x)
#define SIGN(x)     ((x)>=0.0 ? 1.0:-1.0)
#define COS90D      1.0E-6          /* If cos(theta) <= COS90D, theta >= PI/2 - 1e-6 rad. */
#define COSZERO (1.0F - 1.0E-6F)
#define ONE_MINUS_COSZERO 1.0E-12   /* If 1-cos(theta) <= ONE_MINUS_COSZERO, fabs(theta) <= 1e-6 rad. */
#define MC_ZERO 0.0
#define MC_ONE 1.0
#define MC_TWO 2.0
#define DEG_RAD    (PI / 180.0F)
#define RAD_DEG    (180.0F / PI)
#define PRE_MEDIUM_LAYER    0
#define FIRST_MEDIUM_LAYER  1
#define RandomNum rng_gen.rand()


/*        Units are length: mm, time: ps            */
// Raman
#define YES_RAMAN                1
#define NO_RAMAN                 0
#define PI_RAMAN               3.1415926535897932384626433832795
#define C_RAMAN                .299792458


#ifndef SEED_FROM_CLOCK
#define SEED_FROM_CLOCK            YES
#endif

#ifndef PLOT
#define PLOT                NO
#endif

#ifndef CPU_THREADS
#define CPU_THREADS            1
#endif

#ifndef ELASTIC
#define ELASTIC                YES
#endif

#ifndef ABSORPTION
#define ABSORPTION            YES
#endif

#ifndef RAMAN
#define RAMAN                YES
#endif

#ifndef SRS
#define SRS                YES
#endif

#ifndef SIDEBOUND
#define SIDEBOUND            NO
#endif

#ifndef DETECTPUMP
#define DETECTPUMP            YES
#endif

#ifndef DETECTRAMAN
#define DETECTRAMAN            YES
#endif

#ifndef DETECTFORWARD
#define DETECTFORWARD            YES
#endif

#ifndef DETECTBACKWARD
#define DETECTBACKWARD            YES
#endif

#ifndef TIMING
#define TIMING                YES
#endif

#ifndef CHUNK
#define CHUNK                10
#endif


// Monte Carlo simulation type
enum MC_SIM_TYPE
{
    SIM_TYPE_MCSUB                              = 0,      // mcsub benchmark
    SIM_TYPE_MCXYZ                              = 1,      // mcxyz benchmark
    SIM_TYPE_MCXYZ_TT                           = 2,      // mcxyztt (two-term)
    SIM_TYPE_POLARIZATION_CBS_ELECTRIC_FIELDS   = 3,      // Polarization & Coherent Backscattering (electric fields tracking)
    SIM_TYPE_POLARIZATION_MUELLER_STOKES        = 4,      // Polarization using Mueller-Stokes formalizm
    SIM_TYPE_RAMAN                              = 5,      // Raman
    SIM_TYPE_DLS                                = 6,      // DLS simulation
    SIM_TYPE_COMPLEX_LIGHT                      = 7,      // Complex light
    SIM_TYPE_FLUORESCENCE                       = 8,      // Fluorescence
    SIM_TYPE_SAMPLING_VOLUME                    = 9,      // Sampling volume
    SIM_TYPE_SPECKLES                           = 10,     // Speckle fromation
};

// Monte Carlo photon packets propogation
enum MC_PH_PACKETS_DETECTION_STATE
{
    SIM_PH_PACKETS_DETECTION_REF    = 0,    // Reflectance
    SIM_PH_PACKETS_DETECTION_TRANS  = 1,    // Transmittance
};


// The quantity of photon packets where the simulation stops (per GPU)
enum MC_PH_PACKETS_QUANTITY
{
    SIM_PH_PACKETS_QUANTITY_LOW            = 100000,
    SIM_PH_PACKETS_QUANTITY_MEDIUM         = 1000000,
    SIM_PH_PACKETS_QUANTITY_HIGH           = 10000000,
    SIM_PH_PACKETS_QUANTITY_SUPREME        = 100000000,
    SIM_PH_PACKETS_QUANTITY_ULTRA          = 1000000000,
};


/* Run parameters */
struct RunParams
{
    char     myname[STRLEN];        // Holds the user's choice of myname, used in input and output files.
    /* launch parameters */
    int      mcflag, launchflag, boundaryflag, speckleflag, semiflag;
    float    xfocus, yfocus, zfocus;
    float    ux0, uy0, uz0;
    float    radius, det_radius, xd;
    float    waist;
    float    zsurf;
    float    n1, n2;
    float    na, lambda, lc;
    enum MC_PH_PACKETS_DETECTION_STATE det_state;
    enum MC_PH_PACKETS_QUANTITY        ph_quant;

    /* mcxyz bin variables */
    float    dx, dy, dz;     /* bin size [cm] */
    int      Nx, Ny, Nz, Nt; /* # of bins */
    float    xs, ys, zs;        /* launch position */

    /* time */
    float    time_min;               // Requested time duration of computation.
    enum MC_SIM_TYPE mckernelflag;        // Type of MC simulation
};

struct TissueParams
{
    /* tissue parameters */
    float     muav[Ntiss];            // muav[0:Ntiss-1], absorption coefficient of ith tissue type
    float     musv[Ntiss];            // scattering coeff.
    float     gv[Ntiss];              // anisotropy of scattering
    float     gf[Ntiss],af[Ntiss],gb[Ntiss],ab[Ntiss],CC[Ntiss]; // two term phase scattering function paprameters
};

struct SPECKLE
{
    float Phase;
    float Path;
    float Scatt;
    float Total;
    float XX;
    float XY;
    float Ryx;
};

typedef struct
{
    float x, y, z;
} REAL3;

typedef struct
{
    unsigned int z1, z2, z3, z4;
} TAUS_SEED;

typedef struct
{
    REAL3 r;
    REAL3 v;
    int type;
    TAUS_SEED seed;
    REAL3 creation_point;
    float start_t;
} PHOTON;

typedef struct
{
    float x;
    float y;
} REAL2;

typedef struct
{
    float r;
    float theta;
} REAL2_POLAR;


#endif /* definitions_h */
