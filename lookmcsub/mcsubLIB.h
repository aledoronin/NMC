/**********************
 * mcsubLIB.h
 *
 * Header file that declares subroutines
 * found in mcsub.c, used by calling program.
 *******/
 
 /********************
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
 *   to files named "Ji.dat" and "Fi.dat" where i = Nfile.
 * Saves "Ji.dat" in following format:
 *   Saves r[ir]  values in first  column, (1:NR,1) = (rows,cols).
 *   Saves Ji[ir] values in second column, (1:NR,2) = (rows,cols).
 *   Last row is overflow bin.
 * Saves "Fi.dat" in following format:
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


