/*
See LICENSE folder for this sample’s licensing information.

Abstract:
An app that performs a simple calculation on a GPU.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalMC.h"
#import "definitions.h"


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

sprintf(name, "mcOUT%d.dat", Nfile);
target = fopen(name, "w+");

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
fprintf(target, "%d\tNR\n", BINS);
fprintf(target, "%d\tNZ\n", BINS);
fprintf(target, "%0.5f\tdr\n", dr);
fprintf(target, "%0.5f\tdz\n", dz);
fprintf(target, "%0.1e\tNphotons\n", (double)PHOTON_TOTAL);

/* print SAE values */
fprintf(target, "%1.6e\tSpecular reflectance\n", S);
fprintf(target, "%1.6e\tAbsorbed fraction\n", A);
fprintf(target, "%1.6e\tEscaping fraction\n", E);

/* print r[ir] to row */
fprintf(target, "%0.1f", 0.0); /* ignore upperleft element of matrix */
for (ir=1; ir<=BINS; ir++) {
    r2 = dr*ir;
    r1 = dr*(ir-1);
    r = 2.0/3*(r2*r2 + r2*r1 + r1*r1)/(r1 + r2);
    fprintf(target, "\t%1.5f", r);
    }
fprintf(target, "\n");

/* print J[ir] to next row */
fprintf(target, "%0.1f", 0.0); /* ignore this 1st element of 2nd row */
for (ir=1; ir<=BINS; ir++) {
    fprintf(target, "\t%1.12e", J[ir]);
    }
fprintf(target, "\n");

/* printf z[iz], F[iz][ir] to remaining rows */
for (iz=1; iz<=BINS; iz++) {
    z = (iz - 0.5)*dz; /* z values for depth position in 1st column */
    fprintf(target, "%1.5f", z);
    for (ir=1; ir<=BINS; ir++)
        fprintf(target, "\t %1.6e", F[ir*BINS+iz]);
    fprintf(target, "\n");
    }
fclose(target);
} /******** END SUBROUTINE **********/

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        
        float ResJ_GPU[BINS] = {0};
        float ResF_GPU[BINS*BINS] = {0};
        float ResSAE_GPU[3] = {0};
        
        uint64_t start = mach_absolute_time();
        
        NSArray *devices = MTLCopyAllDevices();
           for (id device in devices) {
               NSLog(@"%@", [device name]);
           }
        
        int NumGPURuns = PHOTON_TOTAL/PHOTON_BATCH;
        
        for (int iRunNum = 1; iRunNum <= NumGPURuns; ++iRunNum)
        {
            
            
        // id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLDevice> device = devices[0];

        // Create the custom object used to encapsulate the Metal code.
        // Initializes objects to communicate with the GPU.
        MetalMC* mc_sim = [[MetalMC alloc] initWithDevice:device];

        // Create buffers to hold data
        [mc_sim prepareData];
            
        [mc_sim setNumRuns: &iRunNum];
        
        // Send a command to the GPU to perform the calculation.
        [mc_sim sendComputeCommand];
            
        [mc_sim getComputeResults: ResJ_GPU :ResF_GPU :ResSAE_GPU];
            
        }
    
        uint64_t  end = mach_absolute_time();
        
        double time = ((double)(end - start)/NSEC_PER_SEC);
        
        
        uint64_t start_Scott = mach_absolute_time();
      //  main_mc321(Csph_CPU, Ccyl_CPU, Cpla_CPU);
        uint64_t  end_Scott = mach_absolute_time();
        
        double time_Scott = ((double)(end_Scott - start_Scott)/NSEC_PER_SEC);
        
        printf("Execution time (GPU):%f\nExecution time (CPU):%f\n",time,time_Scott);
        
        for (int ir=1; ir<=BINS; ir++) {
            float r = (ir - 0.5)*dr;
            ResJ_GPU[ir] /= 2.0*PI*r*dr*PHOTON_TOTAL;                /* flux density */
            for (int iz=1; iz<=BINS; iz++)
                ResF_GPU[ir*BINS + iz] /= 2.0*PI*r*dr*dz*PHOTON_TOTAL*mua; /* fluence rate */
            }
        
        ResSAE_GPU[0] /= PHOTON_TOTAL;
        ResSAE_GPU[1] /= PHOTON_TOTAL;
        ResSAE_GPU[2] /= PHOTON_TOTAL;
        
        SaveFile(    Nfile, ResJ_GPU, ResF_GPU, ResSAE_GPU[0], ResSAE_GPU[1], ResSAE_GPU[2],       // save to "mcOUTi.dat", i = Nfile
                    mua, mus, g, n1, n2,
                    mcflag, radius, waist, xs, ys, zs,
                    BINS, BINS, dr, dz, PHOTON_TOTAL);
   
    }
    return 0;
}

