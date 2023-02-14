/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of Neu(t)ralMC: a unified, power-efficient platform platform for photon transport simulations
//  accelerated by Metal/GPU computing and Machine Learning
//  created by Alexander Doronin
//  Source code:    https://github.com/aledoronin
//  Web:            http://www.lighttransport.net/
//  Licence:        BSD-3-Clause, see LICENCE file
//  Contributors:   the respective contributors, as shown by the AUTHORS file
//  Year conceived: 2023
//  This file:      main.m, is the entry point of Neu(t)ralMC
/*---------------------------------------------------------------------------------------------------------------------*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalMC.h"
#import "definitions.h"
#import "mcio.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        
        // Simulation Parameters and results (initiated based on type of simulation)
        struct RunParams runParams;
        struct TissueParams tissParams;
        float* ResJ_GPU = NULL;
        float* ResF_GPU = NULL;
        float* ResF3D_GPU = NULL;
        char* v = NULL;
        float ResSAE_GPU[3] = {0};
        
        if (mc_kernel_type == 0)
        {
            ResJ_GPU = (float*)malloc(sizeof(float) * BINS);
            ResF_GPU = (float*)malloc(sizeof(float) * BINS*BINS);
            memset(ResJ_GPU, 0x0, sizeof(float) * BINS);
            memset(ResF_GPU, 0x0, sizeof(float) * BINS*BINS);
            
        }
        else if (mc_kernel_type == 1)
        {
            if (argc == 1) {
                printf("assuming you've compiled mcxyz.c as gomcxyz ...\n");
                printf("USAGE: gomcxyz name\n");
                printf("which will load the files name_H.mci and name_T.bin\n");
                printf("and run the Monte Carlo program.\n");
                printf("Yields  name_F.bin, which holds the fluence rate distribution.\n");
                return 0;
            }
            
            ReadRunParams(argv, &runParams, &tissParams);
            v = ImportBinaryTissueFile(&runParams, &tissParams);
            PrintRunParameters(&runParams, &tissParams, v);
            SaveOpticalProperties(&runParams, &tissParams);
            ResF3D_GPU = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny*runParams.Nz);
            memset(ResF3D_GPU, 0x0, sizeof(float) * runParams.Nx*runParams.Ny*runParams.Nz);
        }
        
        // Enumerated devices suppoprting Metal Compute
        NSArray *devices = MTLCopyAllDevices();
        for (id device in devices) {
            NSLog(@"%@", [device name]);
        }
        
        int NumGPURuns = PHOTON_TOTAL/PHOTON_BATCH;
        
        // id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLDevice> device = devices[0];
        
        // Create the custom object used to encapsulate the Metal code.
        // Initializes objects to communicate with the GPU.
        MetalMC* mc_sim = [[MetalMC alloc] initWithDevice:device];
        
        // Create buffers to hold data
        [mc_sim prepareData: &runParams :&tissParams :v ];
        
        uint64_t start = mach_absolute_time();
        
        // perform perallel MC simulations
        for (int iRunNum = 1; iRunNum <= NumGPURuns; ++iRunNum)
        {
            //[mc_sim clearBuffers];
            
            // Send number of runs
            [mc_sim setNumRuns: &iRunNum];
            
            // Send a command to the GPU to perform the calculation.
            [mc_sim sendComputeCommand];
        }
        
        // Get simulations results
        [mc_sim getComputeResults: ResJ_GPU :ResF_GPU :ResF3D_GPU :ResSAE_GPU];
        
        uint64_t end = mach_absolute_time();
        
        // Time elapsed in Mach time units.
        const uint64_t elapsedMTU = end - start;
        
        // Get information for converting from MTU to nanoseconds
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        
        // Get elapsed time in nanoseconds:
        const double elapsedNS = (double)elapsedMTU * (double)info.numer / (double)info.denom;
        
        printf("Total execution time (GPU):%f\nSimulation rate %d [photons/sec]\n", elapsedNS/NSEC_PER_SEC,(int)(PHOTON_TOTAL/(elapsedNS/NSEC_PER_SEC)));
        
        if (mc_kernel_type == 0)
        {
            for (int ir=1; ir<=BINS; ir++)
            {
                float r = (ir - 0.5)*dr_mcsub;
                ResJ_GPU[ir] /= 2.0*PI*r*dr_mcsub*PHOTON_TOTAL;                /* flux density */
                for (int iz=1; iz<=BINS; iz++)
                    ResF_GPU[ir*BINS + iz] /= 2.0*PI*r*dr_mcsub*dz_mcsub*PHOTON_TOTAL*mua_mcsub; /* fluence rate */
            }
            
            ResSAE_GPU[0] /= PHOTON_TOTAL;
            ResSAE_GPU[1] /= PHOTON_TOTAL;
            ResSAE_GPU[2] /= PHOTON_TOTAL;
            
            SaveFile(    Nfile, ResJ_GPU, ResF_GPU, ResSAE_GPU[0], ResSAE_GPU[1], ResSAE_GPU[2],       // save to "mcOUTi.dat", i = Nfile
                     mua_mcsub, mus_mcsub, g_mcsub, n1_mcsub, n2_mcsub,
                     mcflag_mcsub, radius_mcsub, waist_mcsub, xs_mcsub, ys_mcsub, zs_mcsub,
                     BINS, BINS, dr_mcsub, dz_mcsub, PHOTON_TOTAL);
        }
        else if (mc_kernel_type == 1)
        {
            /**** SAVE
             Convert data to relative fluence rate [cm^-2] and save.
             *****/
            // Normalize deposition (A) to yield fluence rate (F).
            float temp = (float)runParams.dx*runParams.dy*runParams.dz*PHOTON_TOTAL;
            for (int i = 0; i<runParams.Nx*runParams.Ny*runParams.Nz; i++)
            {
                float mua = tissParams.muav[v[i]];
                float Val = ResF3D_GPU[i] / (temp*mua);
                if (!isnan(Val) && !isinf(Val))
                    ResF3D_GPU[i] = Val;
                else
                    ResF3D_GPU[i] = 0.0F;
            }
            
            // Save the binary file
            char filename[STRLEN];   // temporary filename for writing output.
            strcpy(filename, "skinvessel");
            strcat(filename, "_F.bin");
            printf("saving %s\n", filename);
            FILE* fid = fopen(filename, "wb");   /* 3D voxel output */
            fwrite(ResF3D_GPU, sizeof(float), runParams.Nx*runParams.Ny*runParams.Nz, fid);
            fclose(fid);
            printf("------------------------------------------------------\n");
        }
        
        // Clenup
        if (ResJ_GPU)
            free(ResJ_GPU);
        if (ResF_GPU)
            free(ResF_GPU);
        if (ResF3D_GPU)
            free(ResF3D_GPU);
        if (v)
            free(v);
    }
    return 0;
}

