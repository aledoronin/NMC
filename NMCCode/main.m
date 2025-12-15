/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of NMC: a unified, power-efficient platform platform for photon transport simulations
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
        float* ResRd_GPU = NULL;
        float* ResRd2D_GPU = NULL;
        float* ResPolElectrciField2D_XX_GPU = NULL;
        float* ResPolElectrciField2D_XY_GPU = NULL;
        float* ResPolElectrciField2D_YX_GPU = NULL;
        float* ResPolElectrciField2D_YY_GPU = NULL;
        float* ResPolElectrciField2D_Phase_GPU0 = NULL;
        float* ResPolElectrciField2D_Phase_GPU1 = NULL;
        float* ResPolElectrciField2D_Phase_GPU2 = NULL;
        float* ResPolElectrciField2D_Phase_GPU3 = NULL;
        int*   ResDetPhotons2D_GPU = NULL;
        char* v = NULL;
        float ResSAE_GPU[3] = {0};
        float ResPolVsScatt[MAX_SCATT_POL*POL_CHANNELS] = {0};
        float t_seed = clock();
        
        //ReadRunParamsRaman(argc, argv, &runParams, &tissParams);
        //return 0;
        
        ReadRunParams(argv, &runParams, &tissParams);
        
        switch (runParams.mckernelflag)
        {
            case SIM_TYPE_MCSUB:
            {
                printf("Welcome to mcsub ...\n");
                ResJ_GPU = (float*)malloc(sizeof(float) * BINS);
                ResF_GPU = (float*)malloc(sizeof(float) * BINS*BINS);
                memset(ResJ_GPU, 0x0, sizeof(float) * BINS);
                memset(ResF_GPU, 0x0, sizeof(float) * BINS*BINS);
                v = ImportBinaryTissueFile(&runParams, &tissParams);
                PrintRunParameters(&runParams, &tissParams, v);
                SaveOpticalProperties(&runParams, &tissParams);
            }
            break;
            case SIM_TYPE_POLARIZATION_CBS_ELECTRIC_FIELDS:
            {
                printf("Welcome to polarization simulation based on electric field tracking method ...\n");
                ResJ_GPU = (float*)malloc(sizeof(float) * BINS);
                ResF_GPU = (float*)malloc(sizeof(float) * BINS*BINS);
                v = ImportBinaryTissueFile(&runParams, &tissParams);
                PrintRunParameters(&runParams, &tissParams, v);
                SaveOpticalProperties(&runParams, &tissParams);
                ResF3D_GPU = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny*runParams.Nz);
                memset(ResF3D_GPU, 0x0, sizeof(float) * runParams.Nx*runParams.Ny*runParams.Nz);
                ResRd_GPU = (float*)malloc(sizeof(float) * PHOTON_BATCH);
                memset(ResRd_GPU, 0x0, sizeof(float) *  PHOTON_BATCH);
                ResRd2D_GPU = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny);
                memset(ResRd2D_GPU, 0x0, sizeof(float) * runParams.Nx*runParams.Ny);
                ResDetPhotons2D_GPU = (int*)malloc(sizeof(int) * runParams.Nx*runParams.Ny);
                memset(ResDetPhotons2D_GPU, 0x0, sizeof(int) * runParams.Nx*runParams.Ny);
                ResPolElectrciField2D_XX_GPU = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny);
                memset(ResPolElectrciField2D_XX_GPU, 0x0, sizeof(float) * runParams.Nx*runParams.Ny);
                ResPolElectrciField2D_XY_GPU = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny);
                memset(ResPolElectrciField2D_XY_GPU, 0x0, sizeof(float) * runParams.Nx*runParams.Ny);
                ResPolElectrciField2D_YX_GPU = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny);
                memset(ResPolElectrciField2D_YX_GPU, 0x0, sizeof(float) * runParams.Nx*runParams.Ny);
                ResPolElectrciField2D_YY_GPU = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny);
                memset(ResPolElectrciField2D_YY_GPU, 0x0, sizeof(float) * runParams.Nx*runParams.Ny);
                ResPolElectrciField2D_Phase_GPU0 = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny);
                memset(ResPolElectrciField2D_Phase_GPU0, 0x0, sizeof(float) * runParams.Nx*runParams.Ny);
                ResPolElectrciField2D_Phase_GPU1 = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny);
                memset(ResPolElectrciField2D_Phase_GPU1, 0x0, sizeof(float) * runParams.Nx*runParams.Ny);
                ResPolElectrciField2D_Phase_GPU2 = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny);
                memset(ResPolElectrciField2D_Phase_GPU2, 0x0, sizeof(float) * runParams.Nx*runParams.Ny);
                ResPolElectrciField2D_Phase_GPU3 = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny);
                memset(ResPolElectrciField2D_Phase_GPU3, 0x0, sizeof(float) * runParams.Nx*runParams.Ny);
             }
            break;
            case SIM_TYPE_RAMAN:
            {
                printf("Welcome to Raman simulation...\n");
             }
            break;
            case SIM_TYPE_MCXYZ:
            case SIM_TYPE_MCXYZ_TT:
            {
                if (runParams.mckernelflag == 1)
                    printf("Welcome to mcxyz...\n");
                else if (runParams.mckernelflag == 2)
                    printf("Welcome to mcxyztt...\n");
                if (argc == 1) {
                    printf("assuming you've compiled the programm as nmc ...\n");
                    printf("USAGE: nmc name\n");
                    printf("which will load the files name_H.mci and name_T.bin\n");
                    printf("and run the Monte Carlo program.\n");
                    printf("Yields name_F.bin, which holds the fluence rate distribution.\n");
                    return 0;
                }
                v = ImportBinaryTissueFile(&runParams, &tissParams);
                PrintRunParameters(&runParams, &tissParams, v);
                SaveOpticalProperties(&runParams, &tissParams);
                ResF3D_GPU = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny*runParams.Nz);
                memset(ResF3D_GPU, 0x0, sizeof(float) * runParams.Nx*runParams.Ny*runParams.Nz);
                ResRd_GPU = (float*)malloc(sizeof(float) * PHOTON_BATCH);
                memset(ResRd_GPU, 0x0, sizeof(float) *  PHOTON_BATCH);
                ResRd2D_GPU = (float*)malloc(sizeof(float) * runParams.Nx*runParams.Ny);
                memset(ResRd2D_GPU, 0x0, sizeof(float) * runParams.Nx*runParams.Ny);
                ResDetPhotons2D_GPU = (int*)malloc(sizeof(int) * runParams.Nx*runParams.Ny);
                memset(ResDetPhotons2D_GPU, 0x0, sizeof(int) * runParams.Nx*runParams.Ny);
            }
            break;
            default:
            {
                printf("Unknown or unsupported kernel. quit.\n");
                exit(1);
            }
            break;
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
        MetalMC* mc_sim = [[MetalMC alloc] initWithDevice:device :runParams.mckernelflag];
        
        // Create buffers to hold data
        [mc_sim prepareData: &runParams :&tissParams :v ];
        
        uint64_t start = mach_absolute_time();
        
        // perform perallel MC simulations
        for (int iRunNum = 1; iRunNum <= NumGPURuns; ++iRunNum)
        {
            uint64_t start_kernel = mach_absolute_time();
            //[mc_sim clearBuffers];
            
            // Send number of runs
            [mc_sim setNumRunsSeed: &iRunNum :&t_seed];
            
            // Send a command to the GPU to perform the calculation.
            [mc_sim sendComputeCommand];
            
            uint64_t end_kernel = mach_absolute_time();
            
            // Time elapsed in Mach time units.
            const uint64_t elapsedMTUKernell = end_kernel - start_kernel;
            
            // Get information for converting from MTU to nanoseconds
            mach_timebase_info_data_t infokernel;
            mach_timebase_info(&infokernel);
            
            // Get elapsed time in nanoseconds:
            const double elapsedNSKernell = (double)elapsedMTUKernell * (double)infokernel.numer / (double)infokernel.denom;
            const double kernalphotonrate = PHOTON_BATCH/(elapsedNSKernell/NSEC_PER_SEC);
            const double fractioncompleted = (1.0*iRunNum)/(1.0*NumGPURuns);
            
            if (iRunNum % 100 == 0)
                printf("Compute kernel successfully executed: %f percent completed; total simulated photons: %d out of %d; projected completion in: %f seconds\n",100.0*fractioncompleted, iRunNum*PHOTON_BATCH, PHOTON_TOTAL, (PHOTON_TOTAL - iRunNum*PHOTON_BATCH)/kernalphotonrate);
            
        }
        
        printf("Getting simulation results\n");
        // Get simulations results
        [mc_sim getComputeResults: ResJ_GPU :ResF_GPU :ResF3D_GPU :ResSAE_GPU :ResRd_GPU :ResRd2D_GPU :ResDetPhotons2D_GPU :ResPolElectrciField2D_XX_GPU  :ResPolElectrciField2D_XY_GPU  :ResPolElectrciField2D_YX_GPU  :ResPolElectrciField2D_YY_GPU :ResPolElectrciField2D_Phase_GPU0 :ResPolElectrciField2D_Phase_GPU1 :ResPolElectrciField2D_Phase_GPU2 :ResPolElectrciField2D_Phase_GPU3 :ResPolVsScatt];
        
        uint64_t end = mach_absolute_time();
        
        // Time elapsed in Mach time units.
        const uint64_t elapsedMTU = end - start;
        
        // Get information for converting from MTU to nanoseconds
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        
        // Get elapsed time in nanoseconds:
        const double elapsedNS = (double)elapsedMTU * (double)info.numer / (double)info.denom;
        
        printf("Total execution time (GPU):%f\nSimulation rate %d [photons/sec]\n", elapsedNS/NSEC_PER_SEC,(int)(PHOTON_TOTAL/(elapsedNS/NSEC_PER_SEC)));
        
        switch (runParams.mckernelflag)
        {
            case SIM_TYPE_MCSUB:
            {
                
                float mua_mcsub = tissParams.muav[FIRST_TISSUE_LAYER_MCSUB];
                float mus_mcsub = tissParams.musv[FIRST_TISSUE_LAYER_MCSUB];
                float g_mcsub = tissParams.gv[FIRST_TISSUE_LAYER_MCSUB];
                float n1_mcsub = runParams.n1;
                float n2_mcsub = runParams.n2;
                
                /*float radius_mcsub = runParams.radius;
                float waist_mcsub = runParams.waist;
                float zs_mcsub = runParams.zs;
                float xs_mcsub = runParams.xs;
                float ys_mcsub = runParams.ys;
              
                int mcflag_mcsub = runParams.mcflag;*/
                
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
                
                float R =  ResSAE_GPU[0];
                float A =  ResSAE_GPU[1];
                float E =  ResSAE_GPU[2];
                
                float total_RD = E + R;
                
                printf("Escaping fraction is: %lg\n", E);
                printf("Absorbed fraction is: %lg\n", A);
                printf("Reflectance is: %lg\n", R);
                printf("Total is: %lg\n", E  + A + R);
                printf("Rd is: %lg\n", total_RD);
                
                char filename[STRLEN];   // temporary filename for writing output.
                strcpy(filename, runParams.myname);
                strcat(filename, "_F.bin");
                strcpy(filename,runParams.myname);
                strcat(filename,"_Rd.dat");
                printf("saving %s\n",filename);
                FILE* fid = fopen(filename,"w");
                fprintf(fid,"%0.4f\n",total_RD);
                fclose(fid);
                
                SaveFile(    Nfile, ResJ_GPU, ResF_GPU, ResSAE_GPU[0], ResSAE_GPU[1], ResSAE_GPU[2],       // save to "mcOUTi.dat", i = Nfile
                         mua_mcsub, mus_mcsub, g_mcsub, n1_mcsub, n2_mcsub,
                         mcflag_mcsub, radius_mcsub, waist_mcsub, xs_mcsub, ys_mcsub, zs_mcsub,
                         BINS, BINS, dr_mcsub, dz_mcsub, PHOTON_TOTAL);
            }
            break;
            case SIM_TYPE_POLARIZATION_CBS_ELECTRIC_FIELDS:
            {
                /**** SAVE
                 Convert data to relative fluence rate [mm^-2] and save.
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
                strcpy(filename, runParams.myname);
                strcat(filename, "_F.bin");
                printf("saving %s\n", filename);
                FILE* fid = fopen(filename, "wb");   /* 3D voxel output */
                fwrite(ResF3D_GPU, sizeof(float), runParams.Nx*runParams.Ny*runParams.Nz, fid);
                fclose(fid);
                printf("------------------------------------------------------\n");
                
                // Gather Rd from all threads
                float total_RD = 0.0;
                for (int iPos = 0; iPos < PHOTON_BATCH; ++iPos)
                {
                    float Rd_photon = ResRd_GPU[iPos];
                    if (!isnan(Rd_photon) && !isinf(Rd_photon))
                        total_RD += Rd_photon;
                }
                
                /* save reflectance */
                //printf("W_Rd = %0.2e\n",total_RD);
                total_RD /= PHOTON_TOTAL;
                printf("Total Rd = %0.8f\n",total_RD);
                strcpy(filename,runParams.myname);
                strcat(filename,"_Rd.dat");
                printf("saving %s\n",filename);
                fid = fopen(filename,"w");
                fprintf(fid,"%0.4f\n",total_RD);
                fclose(fid);
                
                // save R(y,x)
                char filename_rd2d[STRLEN];   // temporary filename for writing output.
                strcpy(filename_rd2d, runParams.myname);
                strcat(filename_rd2d, "_Ryx.dat");
                FILE* fid_rd2d = fopen(filename_rd2d, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_rd2d);
                
                for (int i = 0; i < runParams.Nx; i++)
                {
                    for (int j = 0; j < runParams.Ny ;  j++)
                    {
                        int index = j + i*runParams.Nx;
                        float Val = ResRd2D_GPU[index];
                        if (!isnan(Val) && !isinf(Val))
                            fprintf(fid_rd2d, "\t %1.6e", Val);
                        else
                            fprintf(fid_rd2d, "\t %1.6e", 0.0);
                    }
                    fprintf(fid_rd2d, "\n");
                }
                fclose(fid_rd2d);
                
                // save R(y,x)
                char filename_pol_xx[STRLEN];   // temporary filename for writing output.
                char filename_pol_xy[STRLEN];   // temporary filename for writing output.
                char filename_pol_yx[STRLEN];   // temporary filename for writing output.
                char filename_pol_yy[STRLEN];   // temporary filename for writing output.
                char filename_pol_phase0[STRLEN];   // temporary filename for writing output.
                char filename_pol_phase1[STRLEN];   // temporary filename for writing output.
                char filename_pol_phase2[STRLEN];   // temporary filename for writing output.
                char filename_pol_phase3[STRLEN];   // temporary filename for writing output.
                strcpy(filename_pol_xx, runParams.myname);
                strcat(filename_pol_xx, "_Pol_xx.dat");
                strcpy(filename_pol_xy, runParams.myname);
                strcat(filename_pol_xy, "_Pol_xy.dat");
                strcpy(filename_pol_yx, runParams.myname);
                strcat(filename_pol_yx, "_Pol_yx.dat");
                strcpy(filename_pol_yy, runParams.myname);
                strcat(filename_pol_yy, "_Pol_yy.dat");
                strcpy(filename_pol_phase0, runParams.myname);
                strcat(filename_pol_phase0, "_Pol_phase0.dat");
                strcpy(filename_pol_phase1, runParams.myname);
                strcat(filename_pol_phase1, "_Pol_phase1.dat");
                strcpy(filename_pol_phase2, runParams.myname);
                strcat(filename_pol_phase2, "_Pol_phase2.dat");
                strcpy(filename_pol_phase3, runParams.myname);
                strcat(filename_pol_phase3, "_Pol_phase3.dat");
                
                FILE* fid_xx = fopen(filename_pol_xx, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_pol_xx);
                FILE* fid_xy = fopen(filename_pol_xy, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_pol_xy);
                FILE* fid_yx = fopen(filename_pol_yx, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_pol_yx);
                FILE* fid_yy = fopen(filename_pol_yy, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_pol_yy);
                FILE* fid_phase0 = fopen(filename_pol_phase0, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_pol_phase0);
                FILE* fid_phase1 = fopen(filename_pol_phase1, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_pol_phase1);
                FILE* fid_phase2 = fopen(filename_pol_phase2, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_pol_phase2);
                FILE* fid_phase3 = fopen(filename_pol_phase3, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_pol_phase3);
                
                for (int i = 0; i < runParams.Nx; i++)
                {
                    for (int j = 0; j < runParams.Ny ;  j++)
                    {
                        int index = j + i*runParams.Nx;
                        float Val_xx = ResPolElectrciField2D_XX_GPU[index];
                        float Val_xy = ResPolElectrciField2D_XY_GPU[index];
                        float Val_yx = ResPolElectrciField2D_YX_GPU[index];
                        float Val_yy = ResPolElectrciField2D_YY_GPU[index];
                        float Val_phase0 = ResPolElectrciField2D_Phase_GPU0[index];
                        float Val_phase1 = ResPolElectrciField2D_Phase_GPU1[index];
                        float Val_phase2 = ResPolElectrciField2D_Phase_GPU2[index];
                        float Val_phase3 = ResPolElectrciField2D_Phase_GPU3[index];

                        if (!isnan(Val_xx) && !isinf(Val_xx) && !isnan(Val_xy) && !isinf(Val_xy) && !isnan(Val_yx) && !isinf(Val_yx) && !isnan(Val_yy) && !isinf(Val_yy) && !isnan(Val_phase0) && !isinf(Val_phase0) && !isnan(Val_phase1) && !isinf(Val_phase1) && !isnan(Val_phase2) && !isinf(Val_phase2) && !isnan(Val_phase3) && !isinf(Val_phase3))
                        {
                            fprintf(fid_xx, "\t %1.6e", Val_xx);
                            fprintf(fid_xy, "\t %1.6e", Val_xy);
                            fprintf(fid_yx, "\t %1.6e", Val_yx);
                            fprintf(fid_yy, "\t %1.6e", Val_yy);
                            fprintf(fid_phase0, "\t %1.6e", Val_phase0);
                            fprintf(fid_phase1, "\t %1.6e", Val_phase1);
                            fprintf(fid_phase2, "\t %1.6e", Val_phase2);
                            fprintf(fid_phase3, "\t %1.6e", Val_phase3);
                        }
                        else
                        {
                            fprintf(fid_xx, "\t %1.6e", 0.0);
                            fprintf(fid_xy, "\t %1.6e", 0.0);
                            fprintf(fid_yx, "\t %1.6e", 0.0);
                            fprintf(fid_yy, "\t %1.6e", 0.0);
                            fprintf(fid_phase0, "\t %1.6e", 0.0);
                            fprintf(fid_phase1, "\t %1.6e", 0.0);
                            fprintf(fid_phase2, "\t %1.6e", 0.0);
                            fprintf(fid_phase3, "\t %1.6e", 0.0);
                        }
                    }
                    fprintf(fid_xx, "\n");
                    fprintf(fid_xy, "\n");
                    fprintf(fid_yx, "\n");
                    fprintf(fid_yy, "\n");
                    fprintf(fid_phase0, "\n");
                    fprintf(fid_phase1, "\n");
                    fprintf(fid_phase2, "\n");
                    fprintf(fid_phase3, "\n");
                }
                fclose(fid_xx);
                fclose(fid_xy);
                fclose(fid_yx);
                fclose(fid_yy);
                fclose(fid_phase0);
                fclose(fid_phase1);
                fclose(fid_phase2);
                fclose(fid_phase3);
                
                // save R(y,x)
                char filename_Photons2d[STRLEN];   // temporary filename for writing output.
                strcpy(filename_Photons2d, runParams.myname);
                strcat(filename_Photons2d, "_DetPhot2D.dat");
                FILE* fid_Ph2d = fopen(filename_Photons2d, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_Photons2d);
                
                for (int i = 0; i < runParams.Nx; i++)
                {
                    for (int j = 0; j < runParams.Ny ;  j++)
                    {
                        int index = j + i*runParams.Nx;
                        int Val = ResDetPhotons2D_GPU[index];
                        if (!isnan(Val) && !isinf(Val))
                            fprintf(fid_Ph2d, "\t %d", Val);
                        else
                            fprintf(fid_Ph2d, "\t %d", 0);
                    }
                    fprintf(fid_Ph2d, "\n");
                }
                fclose(fid_Ph2d);
                
                char filename_polscatt[STRLEN];   // temporary filename for writing output.
                strcpy(filename_polscatt, runParams.myname);
                strcat(filename_polscatt, "_PolScatt.dat");
                FILE* fid_polscatt = fopen(filename_polscatt, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_polscatt);

                for (int iPos = 0; iPos <= MAX_SCATT_POL; ++iPos)
                {
                    fprintf(fid_polscatt, "%d %lg %lg %lg %lg \n", iPos, ResPolVsScatt[iPos], ResPolVsScatt[MAX_SCATT_POL + iPos], ResPolVsScatt[2*MAX_SCATT_POL + iPos], ResPolVsScatt[3*MAX_SCATT_POL + iPos]);
                }
                fclose(fid_polscatt);
                
            }
            break;
            case SIM_TYPE_RAMAN:
            {

            }
            break;
            case SIM_TYPE_MCXYZ:
            case SIM_TYPE_MCXYZ_TT:
            {
                /**** SAVE
                 Convert data to relative fluence rate [mm^-2] and save.
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
                strcpy(filename, runParams.myname);
                strcat(filename, "_F.bin");
                printf("saving %s\n", filename);
                FILE* fid = fopen(filename, "wb");   /* 3D voxel output */
                fwrite(ResF3D_GPU, sizeof(float), runParams.Nx*runParams.Ny*runParams.Nz, fid);
                fclose(fid);
                printf("------------------------------------------------------\n");
                
                // Gather Rd from all threads
                float total_RD = 0.0;
                for (int iPos = 0; iPos < PHOTON_BATCH; ++iPos)
                {
                    float Rd_photon = ResRd_GPU[iPos];
                    if (!isnan(Rd_photon) && !isinf(Rd_photon))
                        total_RD += Rd_photon;
                }
                
                /* save reflectance */
                //printf("W_Rd = %0.2e\n",total_RD);
                total_RD /= PHOTON_TOTAL;
                printf("Total Rd = %0.8f\n",total_RD);
                strcpy(filename,runParams.myname);
                strcat(filename,"_Rd.dat");
                printf("saving %s\n",filename);
                fid = fopen(filename,"w");
                fprintf(fid,"%0.4f\n",total_RD);
                fclose(fid);
                
                // save R(y,x)
                char filename_rd2d[STRLEN];   // temporary filename for writing output.
                strcpy(filename_rd2d, runParams.myname);
                strcat(filename_rd2d, "_Ryx.dat");
                FILE* fid_rd2d = fopen(filename_rd2d, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_rd2d);
                
                for (int i = 0; i < runParams.Nx; i++)
                {
                    for (int j = 0; j < runParams.Ny ;  j++)
                    {
                        int index = j + i*runParams.Nx;
                        float Val = ResRd2D_GPU[index];
                        if (!isnan(Val) && !isinf(Val))
                            fprintf(fid_rd2d, "\t %1.6e", Val);
                        else
                            fprintf(fid_rd2d, "\t %1.6e", 0.0);
                    }
                    fprintf(fid_rd2d, "\n");
                }
                fclose(fid_rd2d);
                
                // save R(y,x)
                char filename_Photons2d[STRLEN];   // temporary filename for writing output.
                strcpy(filename_Photons2d, runParams.myname);
                strcat(filename_Photons2d, "_DetPhot2D.dat");
                FILE* fid_Ph2d = fopen(filename_Photons2d, "wb");   /* 3D voxel output */
                printf("saving %s\n",filename_Photons2d);
                
                for (int i = 0; i < runParams.Nx; i++)
                {
                    for (int j = 0; j < runParams.Ny ;  j++)
                    {
                        int index = j + i*runParams.Nx;
                        int Val = ResDetPhotons2D_GPU[index];
                        if (!isnan(Val) && !isinf(Val))
                            fprintf(fid_Ph2d, "\t %d", Val);
                        else
                            fprintf(fid_Ph2d, "\t %d", 0);
                    }
                    fprintf(fid_Ph2d, "\n");
                }
                fclose(fid_Ph2d);
                
            }
            break;
            default:
            {
                printf("Unknown or unsupported kernel. quit.\n");
                exit(1);
            }
            break;
        }
        
        // Clenup
        if (ResJ_GPU)
            free(ResJ_GPU);
        if (ResF_GPU)
            free(ResF_GPU);
        if (ResF3D_GPU)
            free(ResF3D_GPU);
        if (ResRd_GPU)
            free(ResRd_GPU);
        if (ResRd2D_GPU)
            free(ResRd2D_GPU);
        if (ResDetPhotons2D_GPU)
            free(ResDetPhotons2D_GPU);
        if (ResPolElectrciField2D_XX_GPU)
            free(ResPolElectrciField2D_XX_GPU);
        if (ResPolElectrciField2D_XY_GPU)
            free(ResPolElectrciField2D_XY_GPU);
        if (ResPolElectrciField2D_YX_GPU)
            free(ResPolElectrciField2D_YX_GPU);
        if (ResPolElectrciField2D_YY_GPU)
            free(ResPolElectrciField2D_YY_GPU);
        if (ResPolElectrciField2D_Phase_GPU0)
            free(ResPolElectrciField2D_Phase_GPU0);
        if (ResPolElectrciField2D_Phase_GPU1)
            free(ResPolElectrciField2D_Phase_GPU1);
        if (v)
            free(v);
    }
    return 0;
}

