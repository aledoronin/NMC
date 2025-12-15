/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of NMC: a unified, power-efficient platform platform for photon transport simulations
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
    FILE*    fid = NULL;        // file ID pointer
    char    buf[32];            // buffer for reading header.dat

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
    sscanf(buf, "%f", &runParams->dx);     // size of bins [mm]
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->dy);     // size of bins [mm]
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->dz);     // size of bins [mm]

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

    fgets(buf, 32,fid);
    sscanf(buf, "%f", &runParams->zsurf);  // zsurf
    
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->det_radius);  // radius
    
    fgets(buf, 32, fid);
    sscanf(buf, "%f", &runParams->xd);  // separation
    
    fgets(buf, 32,fid);
    sscanf(buf, "%f", &runParams->n1);  /* refractive index of medium */
    
    fgets(buf, 32,fid);
    sscanf(buf, "%f", &runParams->n2);   /* refractive index outside medium */
    
    fgets(buf, 32,fid);
    sscanf(buf, "%f", &runParams->na);   /* numerical apperture */
    
    fgets(buf, 32,fid);
    sscanf(buf, "%f", &runParams->lambda);   /* wavelenght */
    
    fgets(buf, 32,fid);
    sscanf(buf, "%f", &runParams->lc);   /* coherence lenght */
    
    fgets(buf, 32, fid);
    sscanf(buf, "%d", &runParams->speckleflag);  // 0 = no speckles, 1 = calc speckles
    
    fgets(buf, 32, fid);
    sscanf(buf, "%d", &runParams->semiflag);  // 1 - semi-analytical
    
    fgets(buf, 32, fid);
    sscanf(buf, "%d", &runParams->det_state);  // detection state
    
    fgets(buf, 32, fid);
    sscanf(buf, "%d", &runParams->mckernelflag);  // 0 = mcsub, 1 = mcxyz, 2 = mcxyztt
    
    switch (runParams->mckernelflag)
    {
        case SIM_TYPE_MCSUB:
        case SIM_TYPE_MCXYZ:
        case SIM_TYPE_POLARIZATION_CBS_ELECTRIC_FIELDS:
        case SIM_TYPE_RAMAN:
        {
            // tissue optical properties
            fgets(buf, 32, fid);
            sscanf(buf, "%d", &runParams->Nt);                // # of tissue types in tissue list
            for (int i = 1; i <= runParams->Nt; i++) {
                fgets(buf, 32, fid);
                sscanf(buf, "%f", &tissParams->muav[i]);    // absorption coeff [mm^-1]
                fgets(buf, 32, fid);
                sscanf(buf, "%f", &tissParams->musv[i]);    // scattering coeff [mm^-1]
                fgets(buf, 32, fid);
                sscanf(buf, "%f", &tissParams->gv[i]);        // anisotropy of scatter [dimensionless]
            }
        }
        break;
        case SIM_TYPE_MCXYZ_TT:
        {
            // tissue optical properties
            fgets(buf, 32, fid);
            sscanf(buf, "%d", &runParams->Nt);                // # of tissue types in tissue list
            for (int i = 1; i <= runParams->Nt; i++) {
                fgets(buf, 32, fid);
                sscanf(buf, "%f", &tissParams->muav[i]);    // absorption coeff [mm^-1]
                fgets(buf, 32, fid);
                sscanf(buf, "%f", &tissParams->musv[i]);    // scattering coeff [mm^-1]
                fgets(buf, 32, fid);
                sscanf(buf, "%f", &tissParams->gf[i]);        // gf
                fgets(buf, 32, fid);
                sscanf(buf, "%f", &tissParams->af[i]);        // af
                fgets(buf, 32, fid);
                sscanf(buf, "%f", &tissParams->gb[i]);        // gb
                fgets(buf, 32, fid);
                sscanf(buf, "%f", &tissParams->ab[i]);        // ab
                fgets(buf, 32, fid);
                sscanf(buf, "%f", &tissParams->CC[i]);        // CC
            }
        }
        break;
        default:
        {
            printf("Unknown or unsupported kernel. quit.\n");
            exit(1);
        }
        break;
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
    printf("Nx = %d, dx = %0.4f [mm]\n", runParams->Nx, runParams->dx);
    printf("Ny = %d, dy = %0.4f [mm]\n", runParams->Ny, runParams->dy);
    printf("Nz = %d, dz = %0.4f [mm]\n", runParams->Nz, runParams->dz);

    printf("xs = %0.4f [mm]\n", runParams->xs);
    printf("ys = %0.4f [mm]\n", runParams->ys);
    printf("zs = %0.4f [mm]\n", runParams->zs);
    printf("mcflag = %d [mm]\n", runParams->mcflag);
    if (runParams->mcflag == 0) printf("launching uniform flat-field beam\n");
    if (runParams->mcflag == 1) printf("launching Gaissian beam\n");
    if (runParams->mcflag == 2) printf("launching isotropic point source\n");
    printf("xfocus = %0.4f [mm]\n", runParams->xfocus);
    printf("yfocus = %0.4f [mm]\n", runParams->yfocus);
    printf("zfocus = %0.2e [mm]\n", runParams->zfocus);
    if (runParams->launchflag == 1) {
        printf("Launchflag ON, so launch the following:\n");
        printf("ux0 = %0.4f [mm]\n", runParams->ux0);
        printf("uy0 = %0.4f [mm]\n", runParams->uy0);
        printf("uz0 = %0.4f [mm]\n", runParams->uz0);
    }
    else {
        printf("Launchflag OFF, so program calculates launch angles.\n");
        printf("radius = %0.4f [mm]\n", runParams->radius);
        printf("waist  = %0.4f [mm]\n", runParams->waist);
    }
    
    printf("zsurf  = %0.4f [mm]\n",runParams->zsurf);
    
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
    
    switch (runParams->mckernelflag)
    {
        case SIM_TYPE_MCSUB:
        {
            printf("performing mcsub\n");
        }
        break;
        case SIM_TYPE_MCXYZ:
        {
            printf("performing mcxyz\n");
        }
        break;
        case SIM_TYPE_MCXYZ_TT:
        {
            printf("performing mcxyztt\n");
        }
        break;
        case SIM_TYPE_POLARIZATION_CBS_ELECTRIC_FIELDS:
        {
            printf("performing polarization simulations using electric field tracking method\n");
        }
        break;
        case SIM_TYPE_RAMAN:
        {
            printf("performing mcraman\n");
        }
        break;
        default:
        {
            printf("Unknown or unsupported kernel. quit.\n");
            exit(1);
        }
        break;
    }

    printf("# of tissues available, Nt = %d\n", runParams->Nt);
    for (int i = 1; i <= runParams->Nt; i++) {
        printf("muav[%d] = %0.4f [mm^-1]\n", i, tissParams->muav[i]);
        printf("musv[%d] = %0.4f [mm^-1]\n", i, tissParams->musv[i]);
        if (runParams->mckernelflag == SIM_TYPE_MCXYZ_TT) {
            printf("gf(%d) = %0.4f;\n",i,tissParams->gf[i]);
            printf("af(%d) = %0.4f;\n",i,tissParams->af[i]);
            printf("gb(%d) = %0.4f;\n",i,tissParams->gb[i]);
            printf("ab(%d) = %0.4f;\n",i,tissParams->ab[i]);
            printf("CC(%d) = %0.4f;\n",i,tissParams->CC[i]);
        }
        else {
            printf("  gv[%d] = %0.4f [--]\n\n", i, tissParams->gv[i]);
        }
    }
    
    printf("n1 = %0.4f [mm]\n", runParams->n1);
    printf("n2 = %0.4f [mm]\n", runParams->n2);

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
        if (runParams->mckernelflag == SIM_TYPE_MCXYZ_TT){
            fprintf(fid,"gf(%ld) = %0.4f;\n\n",i,tissParams->gf[i]);
            fprintf(fid,"af(%ld) = %0.4f;\n\n",i,tissParams->af[i]);
            fprintf(fid,"gb(%ld) = %0.4f;\n\n",i,tissParams->gb[i]);
            fprintf(fid,"ab(%ld) = %0.4f;\n\n",i,tissParams->ab[i]);
            fprintf(fid,"CC(%ld) = %0.4f;\n\n",i,tissParams->CC[i]);
        }
        else {
            fprintf(fid, "gv(%ld) = %0.4f;\n\n", i, tissParams->gv[i]);
        }
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
fprintf(target, "%0.3e\tmua, absorption coefficient [1/mm]\n", curr_mua);
fprintf(target, "%0.4f\tmus, scattering coefficient [1/mm]\n", curr_mus);
fprintf(target, "%0.4f\tg, anisotropy [-]\n", curr_g);
fprintf(target, "%0.4f\tn1, refractive index of tissue\n", curr_n1);
fprintf(target, "%0.4f\tn2, refractive index of outside medium\n", curr_n2);
fprintf(target, "%d\tmcflag\n", curr_mcflag);
fprintf(target, "%0.4f\tradius, radius of flat beam or 1/e radius of Gaussian beam [mm]\n", curr_radius);
fprintf(target, "%0.4f\twaist, 1/e waist of focus [mm]\n", curr_waist);
fprintf(target, "%0.4f\txs, x position of isotropic source [mm]\n", curr_xs);
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



unsigned int Taus_step( unsigned int *z, int S1, int S2, int S3, unsigned int M )
{
    return *z = ( ( ( *z & M ) << S3 ) ^ ( ( ( *z << S1 ) ^ *z ) >> S2 ) );
}

unsigned int LCG_step( unsigned int *z, unsigned int a, unsigned int c )
{
    return *z = ( a**z + c );
}

float hybrid_Taus( TAUS_SEED *seed )
{
    return 2.3283064365387e-10*( Taus_step( &seed->z1, 13, 19, 12, 4294967294UL )
        ^ Taus_step( &seed->z2, 2, 25, 4, 4294967288UL )
        ^ Taus_step( &seed->z3, 3, 11, 17, 4294967280UL )
        ^ LCG_step( &seed->z4, 1664525, 1013904223UL ) );
}

unsigned int hybrid_Taus_int( TAUS_SEED *seed )
{
    return Taus_step( &seed->z1, 13, 19, 12, 4294967294UL )
        ^ Taus_step( &seed->z2, 2, 25, 4, 4294967288UL )
        ^ Taus_step( &seed->z3, 3, 11, 17, 4294967280UL )
        ^ LCG_step( &seed->z4, 1664525, 1013904223UL );
}

struct timespec timediff(struct timespec tstart, struct timespec tend)
{
    struct timespec temp;
    if ((tend.tv_nsec-tstart.tv_nsec)<0)
    {
        temp.tv_sec = tend.tv_sec-tstart.tv_sec-1;
        temp.tv_nsec = 1000000000+tend.tv_nsec-tstart.tv_nsec;
    } else {
        temp.tv_sec = tend.tv_sec-tstart.tv_sec;
        temp.tv_nsec = tend.tv_nsec-tstart.tv_nsec;
    }

return temp;
    
}


#if( PLOT == NO )
    #if( SRS == YES )
int step_forward_cpu( PHOTON *data, unsigned int *raman_counter, unsigned int *n_raman, unsigned int run, FILE *pump_out_f, FILE *raman_out_f, FILE *pump_out_b, FILE *raman_out_b, float time, unsigned int *n_total );
    #else
int step_forward_cpu( PHOTON *data, unsigned int run, FILE *pump_out_f, FILE *raman_out_f, FILE *pump_out_b, FILE *raman_out_b, float time, unsigned int *n_total );
    #endif
#else
    #if( SRS == YES )
int step_forward_cpu( PHOTON *data, unsigned int *raman_counter, unsigned int *n_raman, unsigned int run, float time, unsigned int *n_total );
    #else
int step_forward_cpu( PHOTON *data, unsigned int run, float time, unsigned int *n_total );
    #endif
#endif

float cos_theta( float xi );
float sin_theta( float xi );

unsigned int N_runs =         10;

unsigned int N_laser =         1E4;
unsigned int N_raman =         0;
unsigned int N_tot;

unsigned int seed =         1;    // Will be overridden if SEED_FROM_CLOCK=YES

float width =                 0.5;    // Width of scattering sample
float index_of_refraction =         1.5;    // Index of refraction of the sample
float g =                 0.6;    // Anisotropy parameter
float r_s =                 0.01;    // Scattering mean free path (1/\mu_s) (inversely proportional to the concentration)
float r_a =                10.0;    // Absorption mean free path (1/\mu_a)
float step_size =             0.001;    // Step size propagated

float raman_prob =             0.01;    // Raman cross section times number density (times fudge factor to account for too few photons being simulated)
float stim_raman_prob =             0.1;    // Probability per mm that a laser photon undergoes stimulated Raman scattering (for each Raman photon inside the interaction distance)
float interaction_distance =         0.02;    // The distance at which two photons are said to be able to interact

float laser_beam_radius =         0.03;
float laser_beam_pulse_width =         0.5;
float laser_beam_pulse_delay =         2.0;

float probe_beam_radius =         0.5;
float probe_beam_pulse_width =         1.0;
float probe_beam_pulse_delay =         5.0;

float time_simulated =             1000.0;

#if(SIDEBOUND==YES)
float cutoff_radius =             3.0;
#endif

float vel, dt, spon_prob, stim_prob;
unsigned int N_steps;
float avg_f = 0.0, sigma_f = 0.0;
TAUS_SEED master_seed, *individual_seed;


float cos_theta( float xi )
{
    float ans, temp;
    if( g == 0.0 )
    {
        ans = 2*xi - 1;
    }
    else
    {
        temp = (1.0-g*g)/(1.0-g+2.0*g*xi);
        ans = 1.0/(2*g)*( 1.0 + g*g - temp*temp );
    }
    return(ans);
}

float sin_theta( float xi )
{
    float ans;
    float c_theta = cos_theta( xi );
    ans = sqrt( 1.0 - c_theta*c_theta );
    return(ans);
}

void ReadRunParamsRaman(int argc, const char * argv[], struct RunParams* runParams, struct TissueParams* tissParams)
{
    
    if( argc > 1 )
    {
        FILE *config;
        char param[1000];
        int error;
        config = fopen(argv[1], "r");
        if(config != NULL)
        {
            error = fscanf(config, "N_runs:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            N_runs = atoi( param );
            error = fscanf(config, "N_laser:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            N_laser = atoi( param );
            error = fscanf(config, "N_raman:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            N_raman = atoi( param );
            error = fscanf(config, "width:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            width = atof( param );
            error = fscanf(config, "index_of_refraction:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            index_of_refraction = atof( param );
            error = fscanf(config, "g:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            g = atof( param );
            error = fscanf(config, "r_s:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            r_s = atof( param );
            error = fscanf(config, "r_a:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            r_a = atof( param );
            error = fscanf(config, "step_size:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            step_size = atof( param );
            error = fscanf(config, "raman_prob:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            raman_prob = atof( param );
            error = fscanf(config, "stim_raman_prob:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            stim_raman_prob = atof( param );
            error = fscanf(config, "interaction_distance:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            interaction_distance = atof( param );
            error = fscanf(config, "laser_beam_radius:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            laser_beam_radius = atof( param );
            error = fscanf(config, "laser_beam_pulse_width:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            laser_beam_pulse_width = atof( param );
            error = fscanf(config, "laser_beam_pulse_delay:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            laser_beam_pulse_delay = atof( param );
            error = fscanf(config, "probe_beam_radius:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            probe_beam_radius = atof( param );
            error = fscanf(config, "probe_beam_pulse_width:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            probe_beam_pulse_width = atof( param );
            error = fscanf(config, "probe_beam_pulse_delay:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            probe_beam_pulse_delay = atof( param );
            error = fscanf(config, "time_simulated:%s\n", param);
            if( error < 0 )
            {
                printf("Error reading config file\n");
            }
            time_simulated = atof( param );
        }
        fclose(config);
    }
    
    // run
    int i;
    int j;
    
    #if (SEED_FROM_CLOCK == YES)
        seed = time(NULL);
    #endif
    
    #if (PLOT == NO)
    FILE *config_out, *new_config_out;
    FILE *pump_out_f[N_runs];
    FILE *raman_out_f[N_runs];
    FILE *pump_out_b[N_runs];
    FILE *raman_out_b[N_runs];
    char filename[1000];
    if( argc == 3 )
    {
        sprintf(filename, "../data/config.dat");
        config_out = fopen(filename,"w");
        sprintf(filename, "../data/config_2.dat");
        new_config_out = fopen(filename,"w");
        for(i = 0; i < N_runs; i++)
        {
            sprintf(filename, "../data/%d_pump_f.dat", i);
            pump_out_f[i] = fopen(filename,"w");
            sprintf(filename, "../data/%d_raman_f.dat", i);
            raman_out_f[i] = fopen(filename,"w");
            sprintf(filename, "../data/%d_pump_b.dat", i);
            pump_out_b[i] = fopen(filename,"w");
            sprintf(filename, "../data/%d_raman_b.dat", i);
            raman_out_b[i] = fopen(filename,"w");
        }
    }
    else if( argc > 3 )
    {
        sprintf(filename, "%s/config.dat", argv[3]);
        config_out = fopen(filename,"w");
        sprintf(filename, "%s/config_2.dat", argv[3]);
        new_config_out = fopen(filename,"w");
        for(i = 0; i < N_runs; i++)
        {
            sprintf(filename, "%s/%d_pump_f.dat", argv[3], i);
            pump_out_f[i] = fopen(filename,"w");
            sprintf(filename, "%s/%d_raman_f.dat", argv[3], i);
            raman_out_f[i] = fopen(filename,"w");
            sprintf(filename, "%s/%d_pump_b.dat", argv[3], i);
            pump_out_b[i] = fopen(filename,"w");
            sprintf(filename, "%s/%d_raman_b.dat", argv[3], i);
            raman_out_b[i] = fopen(filename,"w");
        }
    }
    else
    {
        sprintf(filename, "../data/config.dat");
        config_out = fopen(filename,"w");
        sprintf(filename, "../data/config_2.dat");
        new_config_out = fopen(filename,"w");
        for(i = 0; i < N_runs; i++)
        {
            sprintf(filename, "../data/%d_pump_f.dat", i);
            pump_out_f[i] = fopen(filename,"w");
            sprintf(filename, "../data/%d_raman_f.dat", i);
            raman_out_f[i] = fopen(filename,"w");
            sprintf(filename, "../data/%d_pump_b.dat", i);
            pump_out_b[i] = fopen(filename,"w");
            sprintf(filename, "../data/%d_raman_b.dat", i);
            raman_out_b[i] = fopen(filename,"w");
        }
    }
    if( config_out == NULL || new_config_out == NULL )
    {
        printf("Problem opening output files\n");
    }
    for(i = 0; i < N_runs; i++)
    {
        if( pump_out_f[i] == NULL )
        {
            printf("Problem opening output files\n");
        }
        if( raman_out_f[i] == NULL )
        {
            printf("Problem opening output files\n");
        }
        if( pump_out_b[i] == NULL )
        {
            printf("Problem opening output files\n");
        }
        if( raman_out_b[i] == NULL )
        {
            printf("Problem opening output files\n");
        }
    }
    fprintf(config_out, "Nodes: 1\n");
    fprintf(config_out, "CPU threads per node: %d\n", CPU_THREADS );
    fprintf(config_out, "Runs per node: %d\n", N_runs);
    fprintf(config_out, "Seed from clock: %d\n", SEED_FROM_CLOCK);
    fprintf(config_out, "SRS enabled: %d\n", SRS);
    fprintf(config_out, "Timing enabled: %d\n", TIMING);
    fprintf(config_out, "Number of laser photons: %d\n", N_laser);
    fprintf(config_out, "Number of probe photons: %d\n", N_raman);
    fprintf(config_out, "Seed: %d\n", seed);
    fprintf(config_out, "Width: %lf (mm)\n", width);
    fprintf(config_out, "Anisotropy parameter: %lf\n", g);
    fprintf(config_out, "Scattering mean free path: %lf\n", r_s);
    fprintf(config_out, "Absorption mean free path: %lf\n", r_a);
    fprintf(config_out, "Step Size: %lf\n", step_size);
    fprintf(config_out, "Index of refraction: %lf\n", index_of_refraction);
    fprintf(config_out, "Spontaneous Raman Prob: %lf (mm^-1)\n", raman_prob);
    fprintf(config_out, "Stimulated Raman Prob: %lf (mm^-1)\n", stim_raman_prob);
    fprintf(config_out, "SRS interaction distance: %lf (mm)\n", interaction_distance);
    fprintf(config_out, "Laser beam radius: %lf (mm)\n", laser_beam_radius);
    fprintf(config_out, "Laser beam pulse width: %lf (ps)\n", laser_beam_pulse_width);
    fprintf(config_out, "Laser beam pulse delay: %lf (ps)\n", laser_beam_pulse_delay);
    fprintf(config_out, "Probe beam radius: %lf (mm)\n", probe_beam_radius);
    fprintf(config_out, "Probe beam pulse width: %lf (ps)\n", probe_beam_pulse_width);
    fprintf(config_out, "Probe beam pulse delay: %lf (ps)\n", probe_beam_pulse_delay);
    fflush(config_out);
    
    fprintf(new_config_out, "Nodes: 1\n");
    fprintf(new_config_out, "CPU threads per node: %d\n", CPU_THREADS );
    fprintf(new_config_out, "Runs per node: %d\n", N_runs);
    fprintf(new_config_out, "Number of laser photons: %d\n", N_laser);
    fprintf(new_config_out, "Number of probe photons: %d\n", N_raman);
    fprintf(new_config_out, "Seed: %d\n", seed);
    fprintf(new_config_out, "Step Size: %lf\n", step_size);
    fprintf(new_config_out, "\n");
    fprintf(new_config_out, "Width: %lf (mm)\n", width);
    fprintf(new_config_out, "Index of refraction: %lf\n", index_of_refraction);
    fprintf(new_config_out, "Anisotropy parameter: %lf\n", g);
    fprintf(new_config_out, "Scattering mean free path: %lf\n", r_s);
    fprintf(new_config_out, "Absorption mean free path: %lf\n", r_a);
    fprintf(new_config_out, "Spontaneous Raman Prob: %lf (mm^-1)\n", raman_prob);
    fprintf(new_config_out, "Stimulated Raman Prob: %lf (mm^-1)\n", stim_raman_prob);
    fprintf(new_config_out, "SRS interaction distance: %lf (mm)\n", interaction_distance);
    fprintf(new_config_out, "\n");
    fprintf(new_config_out, "Laser beam radius: %lf (mm)\n", laser_beam_radius);
    fprintf(new_config_out, "Laser beam pulse width: %lf (ps)\n", laser_beam_pulse_width);
    fprintf(new_config_out, "Laser beam pulse delay: %lf (ps)\n", laser_beam_pulse_delay);
    fprintf(new_config_out, "\n");
    fprintf(new_config_out, "Probe beam radius: %lf (mm)\n", probe_beam_radius);
    fprintf(new_config_out, "Probe beam pulse width: %lf (ps)\n", probe_beam_pulse_width);
    fprintf(new_config_out, "Probe beam pulse delay: %lf (ps)\n", probe_beam_pulse_delay);
    fprintf(new_config_out, "\n");
    fprintf(new_config_out, "Seed from clock: %d\n", SEED_FROM_CLOCK);
    fprintf(new_config_out, "Absorption enabled: %d\n", ABSORPTION);
    fprintf(new_config_out, "Raman enabled: %d\n", RAMAN);
    fprintf(new_config_out, "SRS enabled: %d\n", SRS);
    fprintf(new_config_out, "Timing enabled: %d\n", TIMING);
    fprintf(new_config_out, "Side boundary enabled: %d\n", SIDEBOUND);
    fprintf(new_config_out, "Pump detection enabled: %d\n", DETECTPUMP);
    fprintf(new_config_out, "Raman detection enabled: %d\n", DETECTRAMAN);
    fprintf(new_config_out, "Forward detection enabled: %d\n", DETECTFORWARD);
    fprintf(new_config_out, "Backward detection enabled: %d\n", DETECTBACKWARD);
    fprintf(new_config_out, "\n");
    fflush(new_config_out);
    #endif
    
    N_tot = N_laser + N_raman;
    vel = C_RAMAN/index_of_refraction;        // Speed of light in the scattering medium
    dt = step_size*index_of_refraction/C_RAMAN;
    stim_prob = stim_raman_prob*step_size;
    N_steps = (unsigned int)( ceil( C_RAMAN*time_simulated/(index_of_refraction*step_size) ) );
    
    printf("Simulating %u runs using %u CPU threads\n", N_runs, CPU_THREADS);
    printf("%u total photons are being simulated in each run\n", N_tot);
    printf("%u scattering events will be simulated in each run\n", N_steps);
    
    individual_seed = (TAUS_SEED*)malloc( N_runs*sizeof(TAUS_SEED) );
    
    #if (PLOT == NO)
        srand( seed );
        master_seed.z1 = rand();
        master_seed.z2 = rand();
        master_seed.z3 = rand();
        master_seed.z4 = rand();
        
        #if( TIMING == YES )
        struct timespec t_start, t_end, t_diff;
        long tsec, tnsec;
        /*        Start Timing             */
        clock_gettime(CLOCK_REALTIME, &t_start);
        #endif
    
        for( i = 0; i < N_runs; i++ )
        {
            individual_seed[i].z1 = hybrid_Taus_int( &master_seed );
            individual_seed[i].z2 = hybrid_Taus_int( &master_seed );
            individual_seed[i].z3 = hybrid_Taus_int( &master_seed );
            individual_seed[i].z4 = hybrid_Taus_int( &master_seed );
        }
        
        for( i = 0; i < N_runs; i++ )
        {
            PHOTON *data = (PHOTON*)malloc( N_tot*sizeof(PHOTON) );
            unsigned int error = 0;
            unsigned int n_total = 0;
            #if( SRS == YES )
            unsigned int N_Raman = 0;
            unsigned int *N_Raman_Counter = (unsigned int*)malloc( N_tot*sizeof(unsigned int) );
            #endif
            //set_initial_conditions( data, i );
            
            int run = i;
            
            int iPos;
            float theta, xi, R, sigma;
            
            
            // Innitialize laser photons
            for(iPos = 0; iPos < N_laser; iPos++)
            {
                // Initialize seeds for random number generation
                data[iPos].seed.z1 = hybrid_Taus_int( &individual_seed[run] );
                data[iPos].seed.z2 = hybrid_Taus_int( &individual_seed[run] );
                data[iPos].seed.z3 = hybrid_Taus_int( &individual_seed[run] );
                data[iPos].seed.z4 = hybrid_Taus_int( &individual_seed[run] );
                
                // Gaussian spatial profile
                xi = hybrid_Taus( &data[iPos].seed );
                sigma = 0.42466090014400953*laser_beam_radius;    // 1/( 2*sqrt( 2*ln(2) ) )
                R = sqrt( 2.0*sigma*sigma*log( 1.0/(1.0-xi) ) );
                theta = 2.0*PI*hybrid_Taus( &data[iPos].seed );
                data[iPos].r.x = R*cos( theta );
                data[iPos].r.y = R*sin( theta );
                
                // Gaussian temporal profile
                xi = hybrid_Taus( &data[iPos].seed );
                sigma = 0.42466090014400953*laser_beam_pulse_width*C_RAMAN;    // 1/( 2*sqrt( 2*ln(2) ) )
                R = sqrt( 2.0*sigma*sigma*log( 1.0/(1.0-xi) ) );
                theta = 2.0*PI*hybrid_Taus( &data[iPos].seed );
                data[iPos].r.z = R*sin( theta ) - laser_beam_pulse_delay*C_RAMAN;
                // Check to make sure none of the photons are in the sample at the begining
                if( data[iPos].r.z >= 0.0 )
                {
                    printf("Laser photon is in the gain region potential error will be incurred in this simulation\n");
                }
                
                // Initially pulse is incident on the sample
                data[iPos].v.x = 0.0;
                data[iPos].v.y = 0.0;
                data[iPos].v.z = 1.0;
                
                // Set type to be a laser photon that has not yet entered the sample
                data[iPos].type = 3;
            }
            
            // Innitialize probe photons
            for(iPos = N_laser; iPos < N_tot; iPos++)
            {
                // Initialize seeds for random number generation
                data[iPos].seed.z1 = hybrid_Taus_int( &individual_seed[run] );
                data[iPos].seed.z2 = hybrid_Taus_int( &individual_seed[run] );
                data[iPos].seed.z3 = hybrid_Taus_int( &individual_seed[run] );
                data[iPos].seed.z4 = hybrid_Taus_int( &individual_seed[run] );
                
                // Gaussian spatial profile
                xi = hybrid_Taus( &data[iPos].seed );
                sigma = 0.42466090014400953*probe_beam_radius;    // 1/( 2*sqrt( 2*ln(2) ) )
                R = sqrt( 2.0*sigma*sigma*log( 1.0/(1.0-xi) ) );
                theta = 2.0*PI*hybrid_Taus( &data[iPos].seed );
                data[iPos].r.x = R*cos( theta );
                data[iPos].r.y = R*sin( theta );
                
                // Gaussian temporal profile
                xi = hybrid_Taus( &data[iPos].seed );
                sigma = 0.42466090014400953*probe_beam_pulse_width*C_RAMAN;    // 1/( 2*sqrt( 2*ln(2) ) )
                R = sqrt( 2.0*sigma*sigma*log( 1.0/(1.0-xi) ) );
                theta = 2.0*PI*hybrid_Taus( &data[iPos].seed );
                data[iPos].r.z = R*sin( theta ) - probe_beam_pulse_delay*C_RAMAN;
                // Check to make sure none of the photons are in the sample at the begining
                if( data[iPos].r.z >= 0.0 )
                {
                    printf("Probe photon is in the gain region potential error will be incurred in this simulation\n");
                }
                
                // Initially pulse is incident on the sample
                data[iPos].v.x = 0.0;
                data[iPos].v.y = 0.0;
                data[iPos].v.z = 1.0;
                
                // Set type to be a Raman photon that has not yet entered the sample
                data[iPos].type = 4;
            }
            
            
            
            ///
            ///
            ///int step_forward_cpu( PHOTON *data, unsigned int *raman_counter, unsigned int *n_raman, unsigned int run, FILE *pump_out_f, FILE *raman_out_f, FILE *pump_out_b, FILE *raman_out_b, float time, unsigned int *n_total );
            ///
            
            for( j = 0; j < N_steps; j++ )
            {
                #if( SRS == YES )
                error = step_forward_cpu( data, N_Raman_Counter, &N_Raman, i, pump_out_f[i], raman_out_f[i], pump_out_b[i], raman_out_b[i], j*index_of_refraction*step_size/C_RAMAN, &n_total );
                #else
                error = step_forward_cpu( data, i, pump_out_f[i], raman_out_f[i], pump_out_b[i], raman_out_b[i], j*index_of_refraction*step_size/C_RAMAN, &n_total );
                #endif
                if(error != 0)
                {
                    printf("Run %d: Exited early because all photons have exited\n", i );
                    break;
                }
            }
            
            printf("Run %d: Complete\n", i );
            
            free(data);
            #if( SRS == YES )
            free(N_Raman_Counter);
            #endif
        }
    
        #if( TIMING == YES )
        /*        End Timing        */
        clock_gettime(CLOCK_REALTIME, &t_end);
        t_diff = timediff(t_start, t_end);
        tsec = t_diff.tv_sec;
        tnsec = t_diff.tv_nsec;
        printf("Compute Time %lu.%lu seconds\n", tsec, tnsec);
        #if( PLOT == NO )
        fprintf(config_out, "Compute Time %lu.%lu seconds\n", tsec, tnsec);
        #endif
        #endif
        
    #endif
    
    free(individual_seed);
    
    #if( PLOT == NO )
    fclose(config_out);
    for(i = 0; i < N_runs; i++)
    {
        fclose(pump_out_f[i]);
        fclose(raman_out_f[i]);
        fclose(pump_out_b[i]);
        fclose(raman_out_b[i]);
    }
    #endif
    
    printf("Program ran successfully\n");

}


#if( PLOT == NO )
    #if( SRS == YES )
    int step_forward_cpu( PHOTON *data, unsigned int *raman_counter, unsigned int *n_raman, unsigned int run, FILE *pump_out_f, FILE *raman_out_f, FILE *pump_out_b, FILE *raman_out_b, float time, unsigned int *n_total )
    #else
    int step_forward_cpu( PHOTON *data, unsigned int run, FILE *pump_out_f, FILE *raman_out_f, FILE *pump_out_b, FILE *raman_out_b, float time, unsigned int *n_total )
    #endif
#else
    #if( SRS == YES )
    int step_forward_cpu( PHOTON *data, unsigned int *raman_counter, unsigned int *n_raman, unsigned int run, PREC time, unsigned int *n_total )
    #else
    int step_forward_cpu( PHOTON *data, unsigned int run, PREC time, unsigned int *n_total )
    #endif
#endif
{
    int i;
    int n_change[CPU_THREADS];
    
    float temp, xi, phi, theta, c_theta, s_theta, c_phi, s_phi, prob, x_end, y_end, vdt;
    float t = time;
    int det = 0;
    int tid = 1;//omp_get_thread_num();
    n_change[tid] = 0;
    #if(SRS == YES)
    int j, k;
        float dx, dy, dz;
    #endif
        float P_Elastic = 1.0 - exp(-step_size/r_s);
        float P_Raman = 1.0 - exp(-raman_prob*step_size);
        float P_Abs = 1.0 - exp(-step_size/r_a);
        float P_SRS = 1.0 - exp(-stim_raman_prob*step_size);
    PHOTON *data_new = (PHOTON*)malloc( N_tot*sizeof(PHOTON) );
    //printf("Hello from thread %d\n", omp_get_thread_num());
    for( i = 0; i < N_tot; i++ )
    {
        PHOTON ph = data[i];
        data_new[i] = data[i];        // Copy data into the new placeholder
        #if(ABSORPTION == YES)
        if(data[i].type == 1 || data[i].type == 2)
        {
            // Linear Absorption
            prob = hybrid_Taus( &data_new[i].seed );
            if( prob < P_Abs )
            {
                data_new[i].type = 0;
                n_change[tid] --;
            }
        }
        #endif
        if( data_new[i].type != 0 )
        {
            if( data[i].type == 1 )    // Laser photon in region
            {
                vdt = step_size;
                
                #if(RAMAN == YES)
                // Spontaneous Raman
                prob = hybrid_Taus( &data_new[i].seed );
                if( prob < P_Raman )
                {
                    data_new[i].type = 2;
                    // Spontaneous Raman photons are emitted uniformally in 4*PI
                    theta = PI*hybrid_Taus( &data_new[i].seed );
                    phi = 2.0*PI*hybrid_Taus( &data_new[i].seed );
                    data_new[i].v.x = sin(theta)*cos(phi);
                    data_new[i].v.y = sin(theta)*sin(phi);
                    data_new[i].v.z = cos(theta);
                    data_new[i].creation_point = data[i].r;
                }
                
                #if(SRS == YES)
                // Stimulated Raman
                if( data_new[i].type != 2 )    // Ensure this photon has not already undergone spontaneous Raman
                {
                    for( j = 0; j < *n_raman; j++ )
                    {
                        k = raman_counter[j];
                        dx = data[k].r.x - data[i].r.x;
                        dy = data[k].r.y - data[i].r.y;
                        dz = data[k].r.z - data[i].r.z;
                        if( dx*dx + dy*dy + dz*dz < interaction_distance*interaction_distance )
                        {
                            prob = hybrid_Taus( &data_new[i].seed );
                            if( prob < stim_prob )
                            {
                                data_new[i].type = 2;
                                // SRS photons take the direction of Raman photon
                                data_new[i].v = data[k].v;
                                data_new[i].creation_point = data[i].r;
                                break;
                            }
                        }
                    }
                }
                #endif
                #endif
                
                #if(ELASTIC == YES)
                // Elastic Scattering
                if( data_new[i].type != 2 )
                {
                    prob = hybrid_Taus( &data_new[i].seed );
                    if( prob < P_Elastic )
                    {
                        // Find new velocities
                        phi = 2.0*PI*hybrid_Taus( &data_new[i].seed );
                        xi = hybrid_Taus( &data_new[i].seed );
                        c_theta = cos_theta( xi );
                        s_theta = sin_theta( xi );
                        c_phi = cos( phi );
                        s_phi = sin( phi );
                
                        // Check if velocity vector lies along the z axis
                        if( fabs( data[i].v.z ) > 0.99999999 )
                        {
                            data_new[i].v.x = s_theta*c_phi;
                            data_new[i].v.y = s_theta*s_phi;
                            data_new[i].v.z = data[i].v.z*c_theta;
                        }
                        else
                        {
                            temp = sqrt( 1.0 - data[i].v.z*data[i].v.z );
                            data_new[i].v.x = s_theta/temp*( data[i].v.y*s_phi
                                - data[i].v.z*data[i].v.x*c_phi ) + data[i].v.x*c_theta;
                            data_new[i].v.y = s_theta/temp*( -data[i].v.x*s_phi
                                - data[i].v.z*data[i].v.y*c_phi ) + data[i].v.y*c_theta;
                            data_new[i].v.z = s_theta*temp*c_phi + data[i].v.z*c_theta;
                        }
                    }
                }
                #endif
            
                // Find new positions
                data_new[i].r.x = data_new[i].v.x*vdt + data[i].r.x;
                data_new[i].r.y = data_new[i].v.y*vdt + data[i].r.y;
                data_new[i].r.z = data_new[i].v.z*vdt + data[i].r.z;
            }
        
            else if( data[i].type == 2 )    // Raman photon in region
            {
                vdt = step_size;
                
                #if(ELASTIC == YES)
                // Elastic Scattering
                prob = hybrid_Taus( &data_new[i].seed );
                if( prob < P_Elastic )
                {
                    // Find new velocities
                    phi = 2.0*PI*hybrid_Taus( &data_new[i].seed );
                    xi = hybrid_Taus( &data_new[i].seed );
                    c_theta = cos_theta( xi );
                    s_theta = sin_theta( xi );
                    c_phi = cos( phi );
                    s_phi = sin( phi );
                
                    // Check if velocity vector lies along the z axis
                    if( fabs( data[i].v.z ) > 0.99999999 )
                    {
                        data_new[i].v.x = s_theta*c_phi;
                        data_new[i].v.y = s_theta*s_phi;
                        data_new[i].v.z = data[i].v.z*c_theta;
                    }
                    else
                    {
                        temp = sqrt( 1.0 - data[i].v.z*data[i].v.z );
                        data_new[i].v.x = s_theta/temp*( data[i].v.y*s_phi
                            - data[i].v.z*data[i].v.x*c_phi ) + data[i].v.x*c_theta;
                        data_new[i].v.y = s_theta/temp*( -data[i].v.x*s_phi
                            - data[i].v.z*data[i].v.y*c_phi ) + data[i].v.y*c_theta;
                        data_new[i].v.z = s_theta*temp*c_phi + data[i].v.z*c_theta;
                    }
                }
                #endif
            
                // Find new positions
                data_new[i].r.x = data_new[i].v.x*vdt + data[i].r.x;
                data_new[i].r.y = data_new[i].v.y*vdt + data[i].r.y;
                data_new[i].r.z = data_new[i].v.z*vdt + data[i].r.z;
            }
        
            else if( data[i].type == 3 )    // Laser photon before entering region
            {
                vdt = index_of_refraction*step_size;
                // Find new positions
                data_new[i].r.x = data_new[i].v.x*vdt + data[i].r.x;
                data_new[i].r.y = data_new[i].v.y*vdt + data[i].r.y;
                data_new[i].r.z = data_new[i].v.z*vdt + data[i].r.z;
            
                // Check to see if photon has entered region
                if( data_new[i].r.z >= 0.0 )
                {
                    vdt = -data[i].r.z/data[i].v.z + step_size + data[i].r.z/(data[i].v.z*index_of_refraction);
                    data_new[i].type = 1;
                    // Adjust new position to take into account the change in refraction index
                    data_new[i].r.x = data_new[i].v.x*vdt + data[i].r.x;
                    data_new[i].r.y = data_new[i].v.y*vdt + data[i].r.y;
                    data_new[i].r.z = data_new[i].v.z*vdt + data[i].r.z;
                    data_new[i].start_t = t;
                    n_change[tid] ++;
                }
            }
        
            else if( data[i].type == 4 )    // Probe photon before entering region
            {
                vdt = index_of_refraction*step_size;
                // Find new positions
                data_new[i].r.x = data_new[i].v.x*vdt + data[i].r.x;
                data_new[i].r.y = data_new[i].v.y*vdt + data[i].r.y;
                data_new[i].r.z = data_new[i].v.z*vdt + data[i].r.z;
            
                // Check to see if photon has entered region
                if( data_new[i].r.z >= 0.0 )
                {
                    vdt = -data[i].r.z/data[i].v.z + step_size + data[i].r.z/(data[i].v.z*index_of_refraction);
                    data_new[i].type = 2;
                    // Adjust new position to take into account the change in refraction index
                    data_new[i].r.x = data_new[i].v.x*vdt + data[i].r.x;
                    data_new[i].r.y = data_new[i].v.y*vdt + data[i].r.y;
                    data_new[i].r.z = data_new[i].v.z*vdt + data[i].r.z;
                    data_new[i].start_t = t;
                    n_change[tid] ++;
                }
            }
        
            // Check if photon will exit
            if( data_new[i].type == 1 || data_new[i].type == 2 )    // Make sure photon is from region
            {
                if( data_new[i].r.z >= width )
                {
                    #if(PLOT == NO)
                    #if(DETECTFORWARD == YES)
                    #if(DETECTPUMP == YES)
                    if( data_new[i].type == 1 )
                    {
                        // Adjust position vectors to place the photon at the boundary
                        vdt = (width - data[i].r.z)/data_new[i].v.z;
                        data_new[i].r.x = data_new[i].v.x*vdt + data[i].r.x;
                        data_new[i].r.y = data_new[i].v.y*vdt + data[i].r.y;
                        data_new[i].r.z = data_new[i].v.z*vdt + data[i].r.z;
                        if(fabs(data_new[i].r.z - width) > 1.0E-6)
                        {
                            printf("Forward Photon Detection Error %lf\n", fabs(data_new[i].r.z - width));
                        }
                        /*else
                        {
                            printf("Forward successful detection\n");
                        }*/
                        fprintf(pump_out_f,"% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\n", data_new[i].r.x, data_new[i].r.y, data_new[i].v.x, data_new[i].v.y, data_new[i].v.z, t, data_new[i].start_t, data_new[i].creation_point.x, data_new[i].creation_point.y, data_new[i].creation_point.z );
                        
                    }
                    #endif
                    #if(DETECTRAMAN == YES)
                    if( data_new[i].type == 2 )
                    {
                        // Adjust position vectors to place the photon at the boundary
                        vdt = (width - data[i].r.z)/data_new[i].v.z;
                        data_new[i].r.x = data_new[i].v.x*vdt + data[i].r.x;
                        data_new[i].r.y = data_new[i].v.y*vdt + data[i].r.y;
                        data_new[i].r.z = data_new[i].v.z*vdt + data[i].r.z;
                        if(fabs(data_new[i].r.z - width) > 1.0E-6)
                        {
                            printf("Forward Photon Detection Error %lf\n", fabs(data_new[i].r.z - width));
                        }
                        /*else
                        {
                            printf("Forward successful detection\n");
                        }*/
                        fprintf(raman_out_f,"% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\n", data_new[i].r.x, data_new[i].r.y, data_new[i].v.x, data_new[i].v.y, data_new[i].v.z, t, data_new[i].start_t, data_new[i].creation_point.x, data_new[i].creation_point.y, data_new[i].creation_point.z );
                    }
                    #endif
                    #endif
                    #endif
                    data_new[i].type = 0;
                    n_change[tid] --;
                }
                if( data_new[i].r.z <= 0 )
                {
                    #if(PLOT == NO)
                    #if(DETECTBACKWARD == YES)
                    #if(DETECTPUMP == YES)
                    if( data_new[i].type == 1 )
                    {
                        // Adjust position vectors to place the photon at the boundary
                        vdt = -data[i].r.z/data_new[i].v.z;
                        data_new[i].r.x = data_new[i].v.x*vdt + data[i].r.x;
                        data_new[i].r.y = data_new[i].v.y*vdt + data[i].r.y;
                        data_new[i].r.z = data_new[i].v.z*vdt + data[i].r.z;
                        if(fabs(data_new[i].r.z - 0.0) > 1.0E-6)
                        {
                            printf("Backward Photon Detection Error %lf\n", fabs(data_new[i].r.z - 0.0));
                        }
                        /*else
                        {
                            printf("Backward successful detection\n");
                        }*/
                        fprintf(pump_out_b,"% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\n", data_new[i].r.x, data_new[i].r.y, data_new[i].v.x, data_new[i].v.y, data_new[i].v.z, t, data_new[i].start_t, data_new[i].creation_point.x, data_new[i].creation_point.y, data_new[i].creation_point.z );
                    }
                    #endif
                    #if(DETECTRAMAN == YES)
                    if( data_new[i].type == 2 )
                    {
                        // Adjust position vectors to place the photon at the boundary
                        vdt = -data[i].r.z/data_new[i].v.z;
                        data_new[i].r.x = data_new[i].v.x*vdt + data[i].r.x;
                        data_new[i].r.y = data_new[i].v.y*vdt + data[i].r.y;
                        data_new[i].r.z = data_new[i].v.z*vdt + data[i].r.z;
                        if(fabs(data_new[i].r.z - 0.0) > 1.0E-6)
                        {
                            printf("Backward Photon Detection Error %lf\n", fabs(data_new[i].r.z - 0.0));
                        }
                        /*else
                        {
                            printf("Backward successful detection\n");
                        }*/
                        fprintf(raman_out_b,"% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\t% 7.6lf\n", data_new[i].r.x, data_new[i].r.y, data_new[i].v.x, data_new[i].v.y, data_new[i].v.z, t, data_new[i].start_t, data_new[i].creation_point.x, data_new[i].creation_point.y, data_new[i].creation_point.z );
                    }
                    #endif
                    #endif
                    #endif
                    data_new[i].type = 0;
                    n_change[tid] --;
                }
                #if(SIDEBOUND==YES)
                if( data_new[i].r.x*data_new[i].r.x + data_new[i].r.y*data_new[i].r.y > cutoff_radius*cutoff_radius )
                {
                    data_new[i].type = 0;
                    n_change[tid] --;
                }
                #endif
            }
        }
    }
    #if(SRS == YES)
    n_raman = 0;
    #endif
    #pragma omp barrier
    #pragma omp for schedule(SCHEDULE, CHUNK)
    for( i = 0; i < N_tot; i++ )
    {
        #if(SRS == YES)
        if( data_new[i].type == 2 )
        {
            //#pragma omp flush (n_raman)
            
            /*#pragma omp critical
            {
            raman_counter[n_raman] = i;
            }
            
            #pragma omp atomic
            n_raman ++;*/
            
            #pragma omp critical
            {
            raman_counter[*n_raman] = i;
            *n_raman  = *n_raman+1;
            }
        }
        #endif
        data[i] = data_new[i];

    }
    
    //printf("%u\n", n_total);
    //printf("%u\t%u\t%u\t%u\n", data_new[0].seed.z1, data_new[0].seed.z2, data_new[0].seed.z3, data_new[0].seed.z4 );
    
    free( data_new );
    
    for(i = 0; i < CPU_THREADS; i++)
    {
        n_total += n_change[i];
    }
    if(time > laser_beam_pulse_delay)
    {
        //if( n_raman == 0 && ((PREC)(n_total)/(PREC)(N_tot)) < 0.0001 )
        if( n_total == 0 )
        {
            return(1);
        }
        else
        {
            return(0);
        }
    }
    else
    {
        return(0);
    }
}




#endif /* mcio_h */

