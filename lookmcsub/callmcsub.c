/*************
 * callmcsub.c
 * A calling program that
 *	1. defines parameters for Monte Carlo run
 *	2. calls the mcsub() routine
 *	3. saves the results into an output file using SaveFile()
 *************/

#define NSEC_PER_SEC 1000000000ull

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mcsubLIB.h"

/*************************/
/**** USER CHOICES *******/
/*************************/
#define BINS        101      /* number of bins, NZ and NR, for z and r */
/*************************/
/*************************/


/**********************
 * MAIN PROGRAM
 *********************/
int main() {

/*************************/
/**** USER CHOICES *******/
/*************************/
/* number of file for saving */
int      Nfile = 1;       /* saves as mcOUTi.dat, where i = Nfile */
/* tissue parameters */
double   mua  = 1.0;      /* excitation absorption coeff. [cm^-1] */
double   mus  = 100;      /* excitation scattering coeff. [cm^-1] */
double   g    = 0.90;     /* excitation anisotropy [dimensionless] */
double   n1   = 1.40;     /* refractive index of medium */
double   n2   = 1.00;     /* refractive index outside medium */
/* beam parameters */
short    mcflag = 0;      /* 0 = collimated, 1 = focused Gaussian, 
                           2 >= isotropic pt. */
double   radius = 0;    /* used if mcflag = 0 or 1 */
double   waist  = 0.10; /* used if mcflag = 1 */
double   zfocus = 1.0;    /* used if mcflag = 1 */
double   xs     = 0.0;    /* used if mcflag = 2, isotropic pt source */
double   ys     = 0.0;    /* used if mcflag = 2, isotropic pt source */
double   zs     = 0.0;    /* used if mcflag = 2, isotropic pt source, or mcflag = 0 collimated*/
int      boundaryflag = 1; /* 0 = infinite medium, 1 = air/tissue surface boundary
/* Run parameters */
double   Nphotons = 10000; /* number of photons to be launched */
double   dr     = 0.0100; /* radial bin size [cm] */
double   dz     = 0.0100; /* depth bin size [cm] */
short    PRINTOUT = 1;
/*************************/
/****** Setup output parameters, vectors, arrays **********/
double    S;    /* specular reflectance at air/tissue boundary */
double    A;    /* total fraction of light absorbed by tissue */
double    E;    /* total fraction of light escaping tissue */
double*   J;    /* escaping flux, J[ir], [W/cm2 per W incident] */
double**  F;   /* fluence rate, F[iz][ir], [W/cm2 per W incident] */
short     NR = BINS;    /* number of radial bins */
short     NZ = BINS;    /* number of depth bins */
J       = AllocVector(1, NR);        /* for escaping flux */
F       = AllocMatrix(1, NZ, 1, NR); /* for absorbed fluence rate */
/*************************/
/*************************/

double albedo, Nsteps, THRESH, t, tperstep;
	
	n1 = 1.4;
	n2 = 1.4;
	dr = 0.0020;
	dz = 0.0020;
	mua = 1.0;
	mus = 100.0;
	g = 0.9;
	Nfile = 0;

	// choose Nphotons
	THRESH = 1e-4;
	albedo = mus/(mua + mus);
	Nsteps = log(THRESH)/log(albedo);
	tperstep = 249e-9;
	t = 30; // s
	//Nphotons = (double)( (long)( t/(tperstep*Nsteps) ) );
	Nphotons = (double)(10000000);
	printf("Nphotons = %5.4e\n", Nphotons);

    clock_t start, end;
    double cpu_time_used;
	
	start = clock();

	mcsub(	mua, mus, g, n1, n2,   /* CALL THE MONTE CARLO SUBROUTINE */
		NR, NZ, dr, dz, Nphotons,
		mcflag, xs, ys, zs, boundaryflag,
		radius, waist, zfocus,
		J, F, &S, &A, &E,
		PRINTOUT);               /* returns J, F, S, A, E */

	if (1==1) {
		SaveFile(	Nfile, J, F, S, A, E,       // save to "mcOUTi.dat", i = Nfile
				mua, mus, g, n1, n2, 
				mcflag, radius, waist, xs, ys, zs,
				NR, NZ, dr, dz, Nphotons);  
		}
	
	printf("Nphotons = %5.1e\n", Nphotons);
	printf("Specular = %5.6f\n", S);
	printf("Absorbed = %5.6f\n", A);
	printf("Escaped  = %5.6f\n", E);
	printf("total    = %5.6f\n", S+A+E);

/*************************/
/*************************/
FreeVector(J, 1, NR);
FreeMatrix(F, 1, NZ, 1, NR);

end = clock();
cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    
 printf("Total execution time (CPU):%f\nDoing %d [photons/sec]\n", cpu_time_used,(int)(Nphotons/cpu_time_used));


return(1);
}
