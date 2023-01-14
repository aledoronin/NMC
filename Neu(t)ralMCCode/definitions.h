//
//  definitions.h
//  MetalComputeBasic
//
//  Created by Alexander Doronin on 11/11/19.
//  Copyright © 2019 Apple. All rights reserved.
//

#ifndef definitions_h
#define definitions_h

#define    PHOTON_TOTAL 100000
#define    PHOTON_BATCH 10000
#define    MAX_SCATT   10000
#define    WEIGHT_MIN   10e-4
/* Constants */
#define    PI          3.1415926
#define    ALIVE       1          /* if photon not yet terminated */
#define    DEAD        0           /* if photon is to be terminated */
#define    THRESHOLD   0.0001        /* used in roulette */
#define    CHANCE      0.1           /* used in roulette */

#define COS90D      1.0E-6
     /* If cos(theta) <= COS90D, theta >= PI/2 - 1e-6 rad. */
#define ONE_MINUS_COSZERO 1.0E-12
     /* If 1-cos(theta) <= ONE_MINUS_COSZERO, fabs(theta) <= 1e-6 rad. */
     /* If 1+cos(theta) <= ONE_MINUS_COSZERO, fabs(PI-theta) <= 1e-6 rad. */
#define SIGN(x)           ((x)>=0 ? 1:-1)
#define BINS        101      /* number of bins, NZ and NR, for z and r */
/* number of file for saving */
#define Nfile 0       /* saves as mcOUTi.dat, where i = Nfile */
/* tissue parameters */
#define mua  1.0      /* excitation absorption coeff. [cm^-1] */
#define mus  100.0      /* excitation scattering coeff. [cm^-1] */
#define g    0.90     /* excitation anisotropy [dimensionless] */
#define n1   1.40     /* refractive index of medium */
#define n2   1.40     /* refractive index outside medium */
/* beam parameters */
#define mcflag 0      /* 0 = collimated, 1 = focused Gaussian,
                           2 >= isotropic pt. */
#define radius 0.0    /* used if mcflag = 0 or 1 */
#define waist  0.10 /* used if mcflag = 1 */
#define zfocus 1.0    /* used if mcflag = 1 */
#define xs     0.0    /* used if mcflag = 2, isotropic pt source */
#define ys     0.0    /* used if mcflag = 2, isotropic pt source */
#define zs     0.0    /* used if mcflag = 2, isotropic pt source, or mcflag = 0 collimated*/
#define boundaryflag 1 /* 0 = infinite medium, 1 = air/tissue surface boundary*/

#define dr     0.0020 /* radial bin size [cm] */
#define dz     0.0020 /* depth bin size [cm] */
#define  PRINTOUT = 1

   /* IF NR IS ALTERED, THEN USER MUST ALSO ALTER THE ARRAY DECLARATION TO A SIZE = NR + 1. */
#define RandomNum (float)rand()/(float)(RAND_MAX)



#endif /* definitions_h */
