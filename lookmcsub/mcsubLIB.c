/**********************************************************
 * mcsubLIB.c
 * Monte Carlo simulation of fluence rate F and escaping flux J
 * in a semi-infinite medium such as biological tissue,
 * with an external_medium/tissue surface boundary.
 *
 * Contains subroutines,
 *		mcsub()
 *		RFresnel()
 *		SaveFile()
 *		RandomGen()
 *	and memory allocation routines,
 *		nerror()
 *		*AllocVector()
 *		**AllocMatrix()
 *		FreeVector()
 *		FreeMatrix()
 **********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/**********************
 * DECLARE SUBROUTINES:
 *********************/
 
/*****
 * The Monte Carlo subroutine mcsub()
 *	Tissue properties:
 *			mua       = absorption coefficient [cm^-1]
 *			mus       = scattering coefficient [cm^-1]
 *			g         = anisotropy of scattering [dimensionless]
 *			n1        = refractive index of tissue
 *			n2        = refractive index of external medium
 *	Incident beam characteristics:
 *			mcflag:   0 = collimated uniform, 1 = Gaussian, 2 = isotropic point
 *			xs,ys,zs  = position of istropic point source (if mcflag = 2)
 *			boundaryflag = 1 if air/tissue surface, = 0 if infinite medium
 *			radius    = radius of beam (1/e width if Gaussian) (if mcflag < 2)
 *			waist     = 1/e radius of focus (if Gaussian (if mcflag = 1)
 *			zfocus    = depth of focus (if Gaussian (if mcflag = 1)
 *	OUTPUT:
 *			J[ir]     = vector of escaping flux [cm^-2] versus radial position
 *			F[iz][ir] = matrix of fluence rate [cm^-2] = [W/cm2 per W]
 *			Sptr      = pointer to S, specular refelctance
 *			Aptr      = pointer to A, total absorbed fraction
 *			Eptr      = pointer to E, total escaping fraction 
 */
void mcsub(double mua, double mus, double g, double n1, double n2, 
           long NR, long NZ, double dr, double dz, double Nphotons,
           int mcflag, double xs, double ys, double zs, int boundaryflag,
           double radius, double waist, double zfocus,
           double *J, double **F, double *Sptr, double *Aptr, double *Eptr,
           short PRINTOUT);

/* Computes internal reflectance at tissue/air interface */
double RFresnel(double n1, double n2, double ca1, double *ca2_Ptr);

/* Saves OUTPUT file, "mcOUTi.dat" where i = Nfile.
 * The INPUT parameters are saved
 * surface escape R(r) and fluence rate distribution F(z,r) 
 * SAVE RESULTS TO FILES 
 *   to files named ñJi.datî and ñFi.datî where i = Nfile.
 * Saves ñJi.datî in following format:
 *   Saves r[ir]  values in first  column, (1:NR,1) = (rows,cols).
 *   Saves Ji[ir] values in second column, (1:NR,2) = (rows,cols).
 *   Last row is overflow bin.
 * Saves ñFi.datî in following format:
 *   The upper element (1,1) is filled with zero, and ignored.
 *   Saves z[iz] values in first column, (2:NZ,1) = (rows,cols).
 *   Saves r[ir] values in first row,    (1,2:NZ) = (rows,cols).
 *   Saves Fi[iz][ir] in (2:NZ+1, 2:NR+1).
 *   Last row and column are overflow bins.
 * Saves "mcSAEi.dat" as three tab-delimited values in one row = [S A E].
 */
void SaveFile(int Nfile, double *J, double **F, double S, double A, double E, 
	double mua, double mus, double g, double n1, double n2, 
	short mcflag, double radius, double waist, double xs, double ys, double zs, 
	short NR, short NZ, double dr, double dz, double Nphotons);

/* Random number generator 
   Initiate by RandomGen(0,1,NULL)
   Use as rnd = RandomGen(1,0,NULL) */
double RandomGen(char Type, long Seed, long *Status); 

/* Memory allocation routines 
 * from MCML ver. 1.0, 1992 L. V. Wang, S. L. Jacques,
 * which are modified versions from Numerical Recipes in C. */
void   nrerror(char error_text[]);
double *AllocVector(short nl, short nh);
double **AllocMatrix(short nrl,short nrh,short ncl,short nch);
void   FreeVector(double *v,short nl,short nh);
void   FreeMatrix(double **m,short nrl,short nrh,short ncl,short nch);



/**********************************************************
 *             list SUBROUTINES:
 **********************************************************/

/**********************************************************
 * The Monte Carlo SUBROUTINE
 **********************************************************/
void mcsub(double mua, double mus, double g, double n1, double n2, 
           long NR, long NZ, double dr, double dz, double Nphotons,
           int mcflag, double xs, double ys, double zs, int boundaryflag,
           double radius, double waist, double zfocus,
           double * J, double ** F, double * Sptr, double * Aptr, double * Eptr,
           short PRINTOUT) 
{
/* Constants */
double	PI          = 3.1415926;
short	ALIVE       = 1;           /* if photon not yet terminated */
short	DEAD        = 0;           /* if photon is to be terminated */
double	THRESHOLD   = 0.0001;        /* used in roulette */
double	CHANCE      = 0.1;           /* used in roulette */

/* Variable parameters */
double	mut, albedo, absorb, rsp, Rsptot, Atot;
double	rnd, xfocus, S, A, E;
double	x,y,z, ux,uy,uz,uz1, uxx,uyy,uzz, s,r,W,temp;
double	psi,costheta,sintheta,cospsi,sinpsi;
long   	iphoton, ir, iz, CNT;
short	photon_status;

/**** INITIALIZATIONS *****/
RandomGen(0, -(int)time(NULL)%(1<<15), NULL); /* initiate with seed = 1, or any long integer. */
CNT = 0;
mut    = mua + mus;
albedo = mus/mut;
Rsptot = 0.0; /* accumulate specular reflectance per photon */
Atot   = 0.0; /* accumulate absorbed photon weight */

/* initialize arrays to zero */
for (ir=1; ir<=NR; ir++) {
	J[ir] = 0.0;
	for (iz=1; iz<=NZ; iz++)
  		F[iz][ir] = 0.0;
  	}

/*============================================================
======================= RUN N photons =====================
 * Launch N photons, initializing each one before progation.
============================================================*/
for (iphoton=1; iphoton<=Nphotons; iphoton++) {

	/* Print out progress for user if mcflag < 3 */
	temp = (double)iphoton;
	if ((PRINTOUT == 1) & (mcflag < 3) & (temp >= 100)) {
		if (temp<1000) {
			if (fmod(temp,100)==0) printf("%1.0f     photons\n",temp);
			}
		if (temp<10000) {
			if (fmod(temp,1000)==0) printf("%1.0f     photons\n",temp);
			}
		else if (temp<100000) {
			if (fmod(temp,10000)==0) printf("%1.0f    photons\n",temp);
			}
		else if (temp<1000000) {
			if (fmod(temp,100000)==0) printf("%1.0f   photons\n",temp);
			}
		else if (temp<10000000) {
			if (fmod(temp,1000000)==0) printf("%1.0f  photons\n",temp);
			}
		else if (temp<100000000) {
			if (fmod(temp,10000000)==0) printf("%1.0f photons\n",temp);
			}
		}
	
/**** LAUNCH 
   Initialize photon position and trajectory.
   Implements an isotropic point source.
*****/

if (mcflag == 0) {
	/* UNIFORM COLLIMATED BEAM INCIDENT AT z = zs */
	/* Launch at (r,zz) = (radius*sqrt(rnd), 0).
	 * Due to cylindrical symmetry, radial launch position is
	 * assigned to x while y = 0. 
	 * radius = radius of uniform beam. */
	/* Initial position */
	rnd = RandomGen(1,0,NULL); 
	x = radius*sqrt(rnd); 
	y = 0;
	z = zs;
	/* Initial trajectory as cosines */
	ux = 0;
	uy = 0;
	uz = 1;  
	/* specular reflectance */
	temp   = n1/n2; /* refractive index mismatch, internal/external */
	temp   = (1.0 - temp)/(1.0 + temp);
	rsp    = temp*temp; /* specular reflectance at boundary */
	}
else if (mcflag == 1) {
	/* GAUSSIAN BEAM AT SURFACE */
	/* Launch at (r,z) = (radius*sqrt(-log(rnd)), 0).
	 * Due to cylindrical symmetry, radial launch position is
	 * assigned to x while y = 0. 
	 * radius = 1/e radius of Gaussian beam at surface. 
	 * waist  = 1/e radius of Gaussian focus.
	 * zfocus = depth of focal point. */
	/* Initial position */
	while ((rnd = RandomGen(1,0,NULL)) <= 0.0); /* avoids rnd = 0 */
	x = radius*sqrt(-log(rnd));
	y = 0.0;
	z = 0.0;
	/* Initial trajectory as cosines */
	/* Due to cylindrical symmetry, radial launch trajectory is
	 * assigned to ux and uz while uy = 0. */
	while ((rnd = RandomGen(1,0,NULL)) <= 0.0); /* avoids rnd = 0 */ 
	xfocus = waist*sqrt(-log(rnd));
	temp = sqrt((x - xfocus)*(x - xfocus) + zfocus*zfocus);
	sintheta = -(x - xfocus)/temp;
	costheta = zfocus/temp;
	ux = sintheta;
	uy = 0.0;
	uz = costheta;
	/* specular reflectance and refraction */
	rsp = RFresnel(n2, n1, costheta, &uz); /* new uz */
	ux  = sqrt(1.0 - uz*uz); /* new ux */
	}
else if  (mcflag == 2) {
	/* ISOTROPIC POINT SOURCE AT POSITION xs,ys,zs */
	/* Initial position */
	x = xs;
	y = ys;
	z = zs;
	/* Initial trajectory as cosines */
	costheta = 1.0 - 2.0*RandomGen(1,0,NULL);
	sintheta = sqrt(1.0 - costheta*costheta);
	psi = 2.0*PI*RandomGen(1,0,NULL);
	cospsi = cos(psi);
	if (psi < PI)
		sinpsi = sqrt(1.0 - cospsi*cospsi); 
	else
		sinpsi = -sqrt(1.0 - cospsi*cospsi);
	ux = sintheta*cospsi;
	uy = sintheta*sinpsi;
	uz = costheta;
	/* specular reflectance */
	rsp = 0.0;
	}
else {
	printf("choose mcflag between 0 to 3\n");
	}

W             = 1.0 - rsp;  /* set photon initial weight */
Rsptot       += rsp; /* accumulate specular reflectance per photon */
photon_status = ALIVE;

/******************************************
****** HOP_ESCAPE_SPINCYCLE **************
* Propagate one photon until it dies by ESCAPE or ROULETTE. 
*******************************************/
do {

/**** HOP
 * Take step to new position
 * s = stepsize
 * ux, uy, uz are cosines of current photon trajectory
 *****/
	while ((rnd = RandomGen(1,0,NULL)) <= 0.0);   /* avoids rnd = 0 */
	s = -log(rnd)/mut;   /* Step size.  Note: log() is base e */
	x += s*ux;           /* Update positions. */
	y += s*uy;
	z += s*uz;

	/* Does photon ESCAPE at surface? ... z <= 0? */
 	if ( (boundaryflag == 1) & (z <= 0)) {
		rnd = RandomGen(1,0,NULL); 
		/* Check Fresnel reflectance at surface boundary */
		if (rnd > RFresnel(n1, n2, -uz, &uz1)) {  
			/* Photon escapes at external angle, uz1 = cos(angle) */
			x -= s*ux;       /* return to original position */
			y -= s*uy;
			z -= s*uz;
			s  = fabs(z/uz); /* calculate stepsize to reach surface */
			x += s*ux;       /* partial step to reach surface */
			y += s*uy;
			r = sqrt(x*x + y*y);   /* find radial position r */
			ir = (long)(r/dr) + 1; /* round to 1 <= ir */
			if (ir > NR) ir = NR;  /* ir = NR is overflow bin */
			J[ir] += W;      /* increment escaping flux */
			photon_status = DEAD;
			}
		else {
			z = -z;   /* Total internal reflection. */
			uz = -uz;
			}
		}

if (photon_status  == ALIVE) {
	/*********************************************
	 ****** SPINCYCLE = DROP_SPIN_ROULETTE ******
	 *********************************************/

	/**** DROP
	 * Drop photon weight (W) into local bin.
	 *****/
	absorb = W*(1 - albedo);      /* photon weight absorbed at this step */
	W -= absorb;                  /* decrement WEIGHT by amount absorbed */
	Atot += absorb;               /* accumulate absorbed photon weight */
	/* deposit power in cylindrical coordinates z,r */
	r  = sqrt(x*x + y*y);         /* current cylindrical radial position */
	ir = (long)(r/dr) + 1;        /* round to 1 <= ir */
	iz = (long)(fabs(z)/dz) + 1;  /* round to 1 <= iz */
	if (ir >= NR) ir = NR;        /* last bin is for overflow */
	if (iz >= NZ) iz = NZ;        /* last bin is for overflow */
	F[iz][ir] += absorb;          /* DROP absorbed weight into bin */

	/**** SPIN 
	 * Scatter photon into new trajectory defined by theta and psi.
	 * Theta is specified by cos(theta), which is determined 
	 * based on the Henyey-Greenstein scattering function.
	 * Convert theta and psi into cosines ux, uy, uz. 
	 *****/
	/* Sample for costheta */
	rnd = RandomGen(1,0,NULL);
	if (g == 0.0)
		costheta = 2.0*rnd - 1.0;
	else if (g == 1.0)
		costheta = 1.0;
	else {
		temp = (1.0 - g*g)/(1.0 - g + 2*g*rnd);
		costheta = (1.0 + g*g - temp*temp)/(2.0*g);
		}
	sintheta = sqrt(1.0 - costheta*costheta);/*sqrt faster than sin()*/

	/* Sample psi. */
	psi = 2.0*PI*RandomGen(1,0,NULL);
	cospsi = cos(psi);
	if (psi < PI)
		sinpsi = sqrt(1.0 - cospsi*cospsi); /*sqrt faster */
	else
		sinpsi = -sqrt(1.0 - cospsi*cospsi);

	/* New trajectory. */
	if (1 - fabs(uz) <= 1.0e-12) {  /* close to perpendicular. */
		uxx = sintheta*cospsi;
		uyy = sintheta*sinpsi;
		uzz = costheta*((uz)>=0 ? 1:-1);
		} 
	else {   /* usually use this option */
		temp = sqrt(1.0 - uz*uz);
		uxx = sintheta*(ux*uz*cospsi - uy*sinpsi)/temp + ux*costheta;
		uyy = sintheta*(uy*uz*cospsi + ux*sinpsi)/temp + uy*costheta;
		uzz = -sintheta*cospsi*temp + uz*costheta;
		}

	/* Update trajectory */
	ux = uxx;
	uy = uyy;
	uz = uzz;

	/**** CHECK ROULETTE 
	 * If photon weight below THRESHOLD, then terminate photon using
	 * Roulette technique. Photon has CHANCE probability of having 
	 * its weight increased by factor of 1/CHANCE,
	 * and 1-CHANCE probability of terminating.
	 *****/
	if (W < THRESHOLD) {
		rnd = RandomGen(1,0,NULL);
		if (rnd <= CHANCE)
			W /= CHANCE;
		else photon_status = DEAD;
		}

	}/**********************************************
	  **** END of SPINCYCLE = DROP_SPIN_ROULETTE *
	  **********************************************/

} 
while (photon_status == ALIVE);
/******************************************
 ****** END of HOP_ESCAPE_SPINCYCLE ******
 ****** when photon_status == DEAD) ******
 ******************************************/

/* If photon dead, then launch new photon. */
} /*======================= End RUN N photons =====================
=====================================================================*/

/************************
 * NORMALIZE 
 *   J[ir]      escaping flux density [W/cm^2 per W incident] 
 *              where bin = 2.0*PI*r[ir]*dr [cm^2].
 *	  F[iz][ir]  fluence rate [W/cm^2 per W incident] 
 *              where bin = 2.0*PI*r[ir]*dr*dz [cm^3].
 ************************/
temp = 0.0;
for (ir=1; ir<=NR; ir++) {
	r = (ir - 0.5)*dr;
	temp += J[ir];    /* accumulate total escaped photon weight */
	J[ir] /= 2.0*PI*r*dr*Nphotons;                /* flux density */
	for (iz=1; iz<=NZ; iz++)
		F[iz][ir] /= 2.0*PI*r*dr*dz*Nphotons*mua; /* fluence rate */
	}

*Sptr = S = Rsptot/Nphotons;
*Aptr = A = Atot/Nphotons;
*Eptr = E = temp/Nphotons;

} /******** END SUBROUTINE **********/


/***********************************************************
 *	FRESNEL REFLECTANCE
 * Computes reflectance as photon passes from medium 1 to 
 * medium 2 with refractive indices n1,n2. Incident
 * angle a1 is specified by cosine value ca1 = cos(a1).
 * Program returns value of transmitted angle a1 as
 * value in *ca2_Ptr = cos(a2).
 ****/
double RFresnel(	double n1,		/* incident refractive index.*/
					double n2,		/* transmit refractive index.*/
					double ca1,		/* cosine of the incident */
									/* angle a1, 0<a1<90 degrees. */
					double *ca2_Ptr) 	/* pointer to the cosine */
									/* of the transmission */
									/* angle a2, a2>0. */
{
double r;

if(n1==n2) { /** matched boundary. **/
	*ca2_Ptr = ca1;
	r = 0.0;
	}
else if(ca1>(1.0 - 1.0e-12)) { /** normal incidence. **/
	*ca2_Ptr = ca1;
	r = (n2-n1)/(n2+n1);
	r *= r;
	}
else if(ca1< 1.0e-6)  {	/** very slanted. **/
	*ca2_Ptr = 0.0;
	r = 1.0;
	}
else  {			  		/** general. **/
	double sa1, sa2; /* sine of incident and transmission angles. */
	double ca2;      /* cosine of transmission angle. */
	sa1 = sqrt(1-ca1*ca1);
	sa2 = n1*sa1/n2;
	if(sa2>=1.0) {	
		/* double check for total internal reflection. */
		*ca2_Ptr = 0.0;
		r = 1.0;
		}
	else {
		double cap, cam;	/* cosines of sum ap or diff am of the two */
							/* angles: ap = a1 + a2, am = a1 - a2. */
		double sap, sam;	/* sines. */
		*ca2_Ptr = ca2 = sqrt(1-sa2*sa2);
		cap = ca1*ca2 - sa1*sa2; /* c+ = cc - ss. */
		cam = ca1*ca2 + sa1*sa2; /* c- = cc + ss. */
		sap = sa1*ca2 + ca1*sa2; /* s+ = sc + cs. */
		sam = sa1*ca2 - ca1*sa2; /* s- = sc - cs. */
		r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam); 
		/* rearranged for speed. */
		}
	}
return(r);
} /******** END SUBROUTINE **********/


/***********************************************************
 * SAVE RESULTS TO FILES 
***********************************************************/
void SaveFile(int Nfile, double *J, double **F, double S, double A, double E, 
	double mua, double mus, double g, double n1, double n2, 
	short mcflag, double radius, double waist, double xs, double ys, double zs, 
	short NR, short NZ, double dr, double dz, double Nphotons)
{
double	r, z, r1, r2;
long  	ir, iz;
char	name[20];
FILE*	target;

sprintf(name, "mcOUT%d.dat", Nfile);
target = fopen(name, "w");

/* print run parameters */
fprintf(target, "%0.3e\tmua, absorption coefficient [1/cm]\n", mua);
fprintf(target, "%0.4f\tmus, scattering coefficient [1/cm]\n", mus);
fprintf(target, "%0.4f\tg, anisotropy [-]\n", g);
fprintf(target, "%0.4f\tn1, refractive index of tissue\n", n1);
fprintf(target, "%0.4f\tn2, refractive index of outside medium\n", n2);
fprintf(target, "%d\tmcflag\n", mcflag);
fprintf(target, "%0.4f\tradius, radius of flat beam or 1/e radius of Gaussian beam [cm]\n", radius);
fprintf(target, "%0.4f\twaist, 1/e waist of focus [cm]\n", waist);
fprintf(target, "%0.4f\txs, x position of isotropic source [cm]\n", xs);
fprintf(target, "%0.4f\tys, y\n", ys);
fprintf(target, "%0.4f\tzs, z\n", zs);
fprintf(target, "%d\tNR\n", NR);
fprintf(target, "%d\tNZ\n", NZ);
fprintf(target, "%0.5f\tdr\n", dr);
fprintf(target, "%0.5f\tdz\n", dz);
fprintf(target, "%0.1e\tNphotons\n", Nphotons);

/* print SAE values */
fprintf(target, "%1.6e\tSpecular reflectance\n", S);
fprintf(target, "%1.6e\tAbsorbed fraction\n", A);
fprintf(target, "%1.6e\tEscaping fraction\n", E);

/* print r[ir] to row */
fprintf(target, "%0.1f", 0.0); /* ignore upperleft element of matrix */
for (ir=1; ir<=NR; ir++) { 
	r2 = dr*ir;
	r1 = dr*(ir-1);
	r = 2.0/3*(r2*r2 + r2*r1 + r1*r1)/(r1 + r2);
	fprintf(target, "\t%1.5f", r);
	}
fprintf(target, "\n");

/* print J[ir] to next row */
fprintf(target, "%0.1f", 0.0); /* ignore this 1st element of 2nd row */
for (ir=1; ir<=NR; ir++) {
	fprintf(target, "\t%1.12e", J[ir]);
	}
fprintf(target, "\n");

/* printf z[iz], F[iz][ir] to remaining rows */
for (iz=1; iz<=NZ; iz++) {
	z = (iz - 0.5)*dz; /* z values for depth position in 1st column */
	fprintf(target, "%1.5f", z);
	for (ir=1; ir<=NR; ir++)
		fprintf(target, "\t %1.6e", F[iz][ir]);
	fprintf(target, "\n");
	}
fclose(target);
} /******** END SUBROUTINE **********/


/*********************************************************************
 *      RANDOM NUMBER GENERATOR
 *      A random number generator that generates uniformly
 *      distributed random numbers between 0 and 1 inclusive.
 *      The algorithm is based on:
 *      W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P.
 *      Flannery, "Numerical Recipes in C," Cambridge University
 *      Press, 2nd edition, (1992).
 *      and
 *      D.E. Knuth, "Seminumerical Algorithms," 2nd edition, vol. 2
 *      of "The Art of Computer Programming", Addison-Wesley, (1981).
 *
 *      When Type is 0, sets Seed as the seed. Make sure 0<Seed<32000.
 *      When Type is 1, returns a random number.
 *      When Type is 2, gets the status of the generator.
 *      When Type is 3, restores the status of the generator.
 *
 *      The status of the generator is represented by Status[0..56].
 *      Make sure you initialize the seed before you get random
 *      numbers.
 ****/
#define MBIG 1000000000
#define MSEED 161803398
#define MZ 0
#define FAC 1.0E-9

double RandomGen(char Type, long Seed, long *Status)
{
static long i1, i2, ma[56];   /* ma[0] is not used. */
long        mj, mk;
short       i, ii;

if (Type == 0) { /* set seed. */
	mj = MSEED - (Seed < 0 ? -Seed : Seed);
	mj %= MBIG;
	ma[55] = mj;
	mk = 1;
	for (i = 1; i <= 54; i++) {
		ii = (21 * i) % 55;
		ma[ii] = mk;
		mk = mj - mk;
		if (mk < MZ)
			mk += MBIG;
		 mj = ma[ii];
		}
	for (ii = 1; ii <= 4; ii++)
		for (i = 1; i <= 55; i++) {
			ma[i] -= ma[1 + (i + 30) % 55];
			if (ma[i] < MZ)
				ma[i] += MBIG;
			}
	i1 = 0;
	i2 = 31;
	} 
else if (Type == 1) {       /* get a number. */
	if (++i1 == 56)
		i1 = 1;
	if (++i2 == 56)
		i2 = 1;
	mj = ma[i1] - ma[i2];
	if (mj < MZ)
		mj += MBIG;
	ma[i1] = mj;
	return (mj * FAC);
	}
else if (Type == 2) {       /* get status. */
	for (i = 0; i < 55; i++)
		Status[i] = ma[i + 1];
	Status[55] = i1;
	Status[56] = i2;
	} 
else if (Type == 3) {       /* restore status. */
	for (i = 0; i < 55; i++)
		ma[i + 1] = Status[i];
	i1 = Status[55];
	i2 = Status[56];
	} 
else
	puts("Wrong parameter to RandomGen().");
return (0);
} 
#undef MBIG
#undef MSEED
#undef MZ
#undef FAC
/*******  end subroutine  ******/


/***********************************************************
 * MEMORY ALLOCATION
 * REPORT ERROR MESSAGE to stderr, then exit the program
 * with signal 1.
 ****/
void nrerror(char error_text[])
{
  fprintf(stderr,"%s\n",error_text);
  fprintf(stderr,"...now exiting to system...\n");
  exit(1);
}

/***********************************************************
 * MEMORY ALLOCATION
 * by Lihong Wang for MCML version 1.0 code, 1992.
 *	ALLOCATE A 1D ARRAY with index from nl to nh inclusive.
 *	Original matrix and vector from Numerical Recipes in C
 *	don't initialize the elements to zero. This will
 *	be accomplished by the following functions. 
 ****/
double *AllocVector(short nl, short nh)
{
double *v;
short i;
v=(double *)malloc((unsigned) (nh-nl+1)*sizeof(double));
if (!v) nrerror("allocation failure in vector()");
v -= nl;
for(i=nl;i<=nh;i++) v[i] = 0.0;	/* init. */
return v;
}

/***********************************************************
 * MEMORY ALLOCATION
 *	ALLOCATE A 2D ARRAY with row index from nrl to nrh 
 *	inclusive, and column index from ncl to nch inclusive.
****/
double **AllocMatrix(short nrl,short nrh,
		short ncl,short nch)
{
short i,j;
double **m;
m=(double **) malloc((unsigned) (nrh-nrl+1) *sizeof(double*));
if (!m) nrerror("allocation failure 1 in matrix()");
	m -= nrl;
for(i=nrl;i<=nrh;i++) {
	m[i]=(double *) malloc((unsigned) (nch-ncl+1) *sizeof(double));
	if (!m[i]) nrerror("allocation failure 2 in matrix()");
	m[i] -= ncl;
	}
for(i=nrl;i<=nrh;i++)
	for(j=ncl;j<=nch;j++) m[i][j] = 0.0;
return m;
}

/***********************************************************
 * MEMORY ALLOCATION
 *	RELEASE MEMORY FOR 1D ARRAY.
 ****/
void FreeVector(double *v,short nl,short nh)
{
free((char*) (v+nl));
}

/***********************************************************
 * MEMORY ALLOCATION
 *	RELEASE MEMORY FOR 2D ARRAY.
 ****/
void FreeMatrix(double **m,short nrl,short nrh, short ncl,short nch)
{
short i;
for(i=nrh;i>=nrl;i--) free((char*) (m[i]+ncl));
free((char*) (m+nrl));
}

