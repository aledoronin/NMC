/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of Neu(t)ralMC: a unified, power-efficient platform platform for photon transport simulations
//  accelerated by Metal/GPU computing and Machine Learning
//  created by Alexander Doronin
//  Source code:    https://github.com/aledoronin
//  Web:            http://www.lighttransport.net/
//  Licence:        BSD-3-Clause, see LICENCE file
//  Contributors:   the respective contributors, as shown by the AUTHORS file
//  Year conceived: 2023
//  This file:      mcio.h, contains routines for loading and saving files in mcxyz and mcsub formats by Steven Jacques
/*---------------------------------------------------------------------------------------------------------------------*/

#ifndef mcio_h
#define mcio_h

void ReadRunParams(const char * argv[], struct RunParams* runParams, struct TissueParams* tissParams)
{
    /* Input/Output */
    char    filename[STRLEN];   // temporary filename for writing output.
    FILE*    fid = NULL;               // file ID pointer
    char    buf[32];                // buffer for reading header.dat

    strcpy(runParams->myname, argv[1]);    // acquire name from argument of function call by user.
    printf("name = %s\n", runParams->myname);

    /**** INPUT FILES *****/
    /* IMPORT myname_H.mci */
    strcpy(filename, runParams->myname);
    strcat(filename, "_H.mci");
    fid = fopen(filename, "r");
    fgets(buf, 32, fid);
    // run parameters
    sscanf(buf, "%f", &runParams->time_min); // desired time duration of run [min]
    fgets(buf, 32, fid);
    sscanf(buf, "%d", &runParams->Nx);  // # of bins
    fgets(buf, 32, fid);
    sscanf(buf, "%d", &runParams->Ny);  // # of bins
    fgets(buf, 32, fid);
    sscanf(buf, "%d", &runParams->Nz);  // # of bins

    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->dx);     // size of bins [cm]
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->dy);     // size of bins [cm]
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->dz);     // size of bins [cm]

    // launch parameters
    fgets(buf, 32, fid);
    sscanf(buf, "%d", &runParams->mcflag);  // mcflag, 0 = uniform, 1 = Gaussian, 2 = iso-pt
    fgets(buf, 32, fid);
    sscanf(buf, "%d", &runParams->launchflag);  // launchflag, 0 = ignore, 1 = manually set
    fgets(buf, 32, fid);
    sscanf(buf, "%d", &runParams->boundaryflag);  // 0 = no boundaries, 1 = escape at all boundaries, 2 = escape at surface only

    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->xs);  // initial launch point
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->ys);  // initial launch point
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->zs);  // initial launch point

    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->xfocus);  // xfocus
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->yfocus);  // yfocus
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->zfocus);  // zfocus

    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->ux0);  // ux trajectory
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->uy0);  // uy trajectory
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->uz0);  // uz trajectory

    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->radius);  // radius
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->waist);  // waist

    // tissue optical properties
    fgets(buf, 32, fid);
    sscanf(buf, "%d", &runParams->Nt);                // # of tissue types in tissue list
    for (int i = 1; i <= runParams->Nt; i++) {
        fgets(buf, 32, fid);
        sscanf(buf, "%f", &tissParams->muav[i]);    // absorption coeff [cm^-1]
        fgets(buf, 32, fid);
        sscanf(buf, "%f", &tissParams->musv[i]);    // scattering coeff [cm^-1]
        fgets(buf, 32, fid);
        sscanf(buf, "%f", &tissParams->gv[i]);        // anisotropy of scatter [dimensionless]
    }
    fclose(fid);

}

char* ImportBinaryTissueFile(struct RunParams* runParams, struct TissueParams* tissParams)
{
    char *v = NULL;
    int NN = runParams->Nx*runParams->Ny*runParams->Nz;
    v = (char*)malloc(NN*sizeof(char));  /* tissue structure */
    char    filename[STRLEN];   // temporary filename for writing output.
    FILE*    fid = NULL;               // file ID pointer

    // read binary file
    strcpy(filename, runParams->myname);
    strcat(filename, "_T.bin");
    fid = fopen(filename, "rb");
    fread(v, sizeof(char), NN, fid);
    fclose(fid);
    
    return(v);
}

void PrintRunParameters(const struct RunParams* runParams, const struct TissueParams* tissParams, char*v)
{
    printf("time_min = %0.2f min\n", runParams->time_min);
    printf("Nx = %d, dx = %0.4f [cm]\n", runParams->Nx, runParams->dx);
    printf("Ny = %d, dy = %0.4f [cm]\n", runParams->Ny, runParams->dy);
    printf("Nz = %d, dz = %0.4f [cm]\n", runParams->Nz, runParams->dz);

    printf("xs = %0.4f [cm]\n", runParams->xs);
    printf("ys = %0.4f [cm]\n", runParams->ys);
    printf("zs = %0.4f [cm]\n", runParams->zs);
    printf("mcflag = %d [cm]\n", runParams->mcflag);
    if (runParams->mcflag == 0) printf("launching uniform flat-field beam\n");
    if (runParams->mcflag == 1) printf("launching Gaissian beam\n");
    if (runParams->mcflag == 2) printf("launching isotropic point source\n");
    printf("xfocus = %0.4f [cm]\n", runParams->xfocus);
    printf("yfocus = %0.4f [cm]\n", runParams->yfocus);
    printf("zfocus = %0.2e [cm]\n", runParams->zfocus);
    if (runParams->launchflag == 1) {
        printf("Launchflag ON, so launch the following:\n");
        printf("ux0 = %0.4f [cm]\n", runParams->ux0);
        printf("uy0 = %0.4f [cm]\n", runParams->uy0);
        printf("uz0 = %0.4f [cm]\n", runParams->uz0);
    }
    else {
        printf("Launchflag OFF, so program calculates launch angles.\n");
        printf("radius = %0.4f [cm]\n", runParams->radius);
        printf("waist  = %0.4f [cm]\n", runParams->waist);
    }
    if (runParams->boundaryflag == 0)
        printf("boundaryflag = 0, so no boundaries.\n");
    else if (runParams->boundaryflag == 1)
        printf("boundaryflag = 1, so escape at all boundaries.\n");
    else if (runParams->boundaryflag == 2)
        printf("boundaryflag = 2, so escape at surface only.\n");
    else {
        printf("improper boundaryflag. quit.\n");
        exit(1);
    }
    printf("# of tissues available, Nt = %d\n", runParams->Nt);
    for (int i = 1; i <= runParams->Nt; i++) {
        printf("muav[%d] = %0.4f [cm^-1]\n", i, tissParams->muav[i]);
        printf("musv[%d] = %0.4f [cm^-1]\n", i, tissParams->musv[i]);
        printf("  gv[%d] = %0.4f [--]\n\n", i, tissParams->gv[i]);
    }

    // Show tissue on screen, along central z-axis, by listing tissue type #'s.
    int iy = runParams->Ny / 2;
    int ix = runParams->Nx / 2;
    printf("central axial profile of tissue types:\n");
    for (int iz = 0; iz<runParams->Nz; iz++) {
        int i = (int)(iz*runParams->Ny*runParams->Nx + ix*runParams->Ny + iy);
        printf("%d", v[i]);
    }
    printf("\n\n");
}

void SaveOpticalProperties(const struct RunParams* runParams, const struct TissueParams* tissParams)
{
    /* Input/Output */
    char    filename[STRLEN];   // temporary filename for writing output.
    FILE*    fid = NULL;               // file ID pointer
    // SAVE optical properties, for later use by MATLAB.
    strcpy(filename, runParams->myname);
    strcat(filename, "_props.m");
    fid = fopen(filename, "w");
    for (long i = 1; i <= runParams->Nt; i++) {
        fprintf(fid, "muav(%ld) = %0.4f;\n", i, tissParams->muav[i]);
        fprintf(fid, "musv(%ld) = %0.4f;\n", i, tissParams->musv[i]);
        fprintf(fid, "gv(%ld) = %0.4f;\n\n", i, tissParams->gv[i]);
    }
    fclose(fid);
}

/***********************************************************
 * SAVE RESULTS TO FILES
***********************************************************/
void SaveFile(int Nfile_ID, float *J, float *F, double S, double A, double E,
    double curr_mua, double curr_mus, double curr_g, double curr_n1, double curr_n2,
    short curr_mcflag, double curr_radius, double curr_waist, double curr_xs, double curr_ys, double curr_zs,
    short curr_NR, short curr_NZ, double curr_dr, double curr_dz, double curr_Nphotons)
{
double    r, z, r1, r2;
long      ir, iz;
char    name[20];
FILE*    target;

sprintf(name, "mcOUT%d.dat", 0);
target = fopen(name, "w+");

/* print run parameters */
fprintf(target, "%0.3e\tmua, absorption coefficient [1/cm]\n", curr_mua);
fprintf(target, "%0.4f\tmus, scattering coefficient [1/cm]\n", curr_mus);
fprintf(target, "%0.4f\tg, anisotropy [-]\n", curr_g);
fprintf(target, "%0.4f\tn1, refractive index of tissue\n", curr_n1);
fprintf(target, "%0.4f\tn2, refractive index of outside medium\n", curr_n2);
fprintf(target, "%d\tmcflag\n", curr_mcflag);
fprintf(target, "%0.4f\tradius, radius of flat beam or 1/e radius of Gaussian beam [cm]\n", curr_radius);
fprintf(target, "%0.4f\twaist, 1/e waist of focus [cm]\n", curr_waist);
fprintf(target, "%0.4f\txs, x position of isotropic source [cm]\n", curr_xs);
fprintf(target, "%0.4f\tys, y\n", curr_ys);
fprintf(target, "%0.4f\tzs, z\n", curr_zs);
fprintf(target, "%d\tNR\n", curr_NR);
fprintf(target, "%d\tNZ\n", curr_NZ);
fprintf(target, "%0.5f\tdr\n", curr_dr);
fprintf(target, "%0.5f\tdz\n", curr_dz);
fprintf(target, "%0.1e\tNphotons\n", (double)curr_Nphotons);

/* print SAE values */
fprintf(target, "%1.6e\tSpecular reflectance\n", S);
fprintf(target, "%1.6e\tAbsorbed fraction\n", A);
fprintf(target, "%1.6e\tEscaping fraction\n", E);

/* print r[ir] to row */
fprintf(target, "%0.1f", 0.0); /* ignore upperleft element of matrix */
for (ir=1; ir<=curr_NR; ir++) {
    r2 = curr_dr*ir;
    r1 = curr_dr*(ir-1);
    r = 2.0/3*(r2*r2 + r2*r1 + r1*r1)/(r1 + r2);
    fprintf(target, "\t%1.5f", r);
    }
fprintf(target, "\n");

/* print J[ir] to next row */
fprintf(target, "%0.1f", 0.0); /* ignore this 1st element of 2nd row */
for (ir=1; ir<=curr_NR; ir++) {
    fprintf(target, "\t%1.12e", J[ir]);
    }
fprintf(target, "\n");

/* printf z[iz], F[iz][ir] to remaining rows */
for (iz=1; iz<=curr_NZ; iz++) {
    z = (iz - 0.5)*curr_dz; /* z values for depth position in 1st column */
    fprintf(target, "%1.5f", z);
    for (ir=1; ir<=curr_NR; ir++)
        fprintf(target, "\t %1.6e", F[ir*curr_NR+iz]);
    fprintf(target, "\n");
    }
fclose(target);
} /******** END SUBROUTINE **********/


#endif /* mcio_h */
