/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of NMC: a unified, power-efficient platform platform for photon transport simulations
//  accelerated by Metal/GPU computing and Machine Learning
//  created by Alexander Doronin
//  Source code:    https://github.com/aledoronin
//  Web:            http://www.lighttransport.net/
//  Licence:        BSD-3-Clause, see LICENCE file
//  Contributors:   the respective contributors, as shown by the AUTHORS file
//  Year conceived: 2023
//  This file:      MetalMC.m, provides the main implemetation of the interface to MC GPU Computing is based on Apple's
//  PerformingCalculations on a GPU example: https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu
//
//  Copyright © 2019 Apple Inc.
//  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
/*---------------------------------------------------------------------------------------------------------------------*/

#import "MetalMC.h"
#include "definitions.h"


// Function to trim leading and trailing whitespace from a string
NSString* trimWhitespace(NSString *str) {
    return [str stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
}

// Function to load data from a comma-separated file into a float array
int loadFloatArrayFromFile(NSString *filename, float array[], int maxSize) {
    NSError *error;
    NSString *fileContents = [NSString stringWithContentsOfFile:filename encoding:NSUTF8StringEncoding error:&error];
    if (error) {
        NSLog(@"Error reading file: %@", [error localizedDescription]);
        return -1;
    }

    NSArray *components = [fileContents componentsSeparatedByString:@","];
    int index = 0;

    for (NSString *component in components) {
        if (index >= maxSize) break;

        NSString *trimmed = trimWhitespace(component);
        array[index++] = [trimmed floatValue];
    }

    return index; // Return the number of elements read
}

@implementation MetalMC
{
    id<MTLDevice> _mDevice;
    
    // The compute pipeline generated from the compute kernel in the .metal shader file.
    id<MTLComputePipelineState> _mMCFunctionPSO;
    
    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _mCommandQueue;
    
    // Buffers to hold data.
    id<MTLBuffer> _mBufferRunParams;
    id<MTLBuffer> _mBufferTissPrams;
    id<MTLBuffer> _mBufferV;
    id<MTLBuffer> _mBufferJ;
    id<MTLBuffer> _mBufferF;
    id<MTLBuffer> _mBufferF3D;
    id<MTLBuffer> _mBufferSAE;
    id<MTLBuffer> _mBufferRd;
    id<MTLBuffer> _mBufferRd2D;
    id<MTLBuffer> _mBufferPhot2D;
   
    id<MTLBuffer> _mBufferPhotonCoordinates;
    id<MTLBuffer> _mBufferPol2DXX;
    id<MTLBuffer> _mBufferPol2DXY;
    id<MTLBuffer> _mBufferPol2DYX;
    id<MTLBuffer> _mBufferPol2DYY;
    id<MTLBuffer> _mBufferPol2DPhase0;
    id<MTLBuffer> _mBufferPol2DPhase1;
    id<MTLBuffer> _mBufferPol2DPhase2;
    id<MTLBuffer> _mBufferPol2DPhase3;
    id<MTLBuffer> _mBufferPolVsScatt;
    id<MTLBuffer> _mBufferPhase02D;
    id<MTLBuffer> _mBufferSpeckleData;
    id<MTLBuffer> _mBufferSpeckleCount;
    id<MTLBuffer> _mBufferRunNum;
    id<MTLBuffer> _mBufferTimeSeed;
    
    id<MTLBuffer> _mBufferIntensity_radial;
    id<MTLBuffer> _mBufferIntensity_azimuthal;
    id<MTLBuffer> _mBufferPhase_radial;
    id<MTLBuffer> _mBufferPhase_azimuthal;
    
    struct RunParams _mRunParams;
}

- (instancetype) initWithDevice: (id<MTLDevice>) device :(enum MC_SIM_TYPE) mckernelflag
{
    self = [super init];
    if (self)
    {
        _mDevice = device;
        
        NSError* error = nil;
        
        // Load the shader files with a .metal file extension in the project

        id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
        if (defaultLibrary == nil)
        {
            NSLog(@"Failed to find the default library.");
            return nil;
        }

        id<MTLFunction> MonteCarloFunction;
        switch (mckernelflag)
        {
            case SIM_TYPE_MCSUB:
            {
                MonteCarloFunction = [defaultLibrary newFunctionWithName:@"MCSubKernel"];
            }
            break;
            case SIM_TYPE_MCXYZ:
            {
                MonteCarloFunction = [defaultLibrary newFunctionWithName:@"MCXYZKernel"];
            }
            break;
            case SIM_TYPE_MCXYZ_TT:
            {
                MonteCarloFunction = [defaultLibrary newFunctionWithName:@"MCXYZKernelTT"];
            }
            break;
            case SIM_TYPE_POLARIZATION_CBS_ELECTRIC_FIELDS:
            {
                MonteCarloFunction = [defaultLibrary newFunctionWithName:@"MCPolCbsElectricFiedsKernel"];
                //MonteCarloFunction = [defaultLibrary newFunctionWithName:@"MCPolCbsElectricFiedsMajoranaKernel"];
            }
            break;
            case SIM_TYPE_RAMAN:
            {
                MonteCarloFunction = [defaultLibrary newFunctionWithName:@"MCRamanKernel"];
            }
            break;
            default:
            {
                NSLog(@"Failed to find the MonteCarlo function.");
                return nil;
            }
            break;
        }
        
        if (MonteCarloFunction == nil)
        {
            NSLog(@"Failed to find the MonteCarlo function.");
            return nil;
        }
        
        // Create a compute pipeline state object.
        _mMCFunctionPSO = [_mDevice newComputePipelineStateWithFunction: MonteCarloFunction error:&error];
        if (_mMCFunctionPSO == nil)
        {
            //  If the Metal API validation is enabled, you can find out more information about what
            //  went wrong.  (Metal API validation is enabled by default when a debug build is run
            //  from Xcode)
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return nil;
        }
        
        _mCommandQueue = [_mDevice newCommandQueue];
        if (_mCommandQueue == nil)
        {
            NSLog(@"Failed to find the command queue.");
            return nil;
        }
    }
    
    return self;
}

- (void) prepareData: (struct RunParams*) run_params :(struct TissueParams*) tissParams :(char*) v
{
    memcpy(&_mRunParams, run_params, sizeof(struct RunParams));

    switch (_mRunParams.mckernelflag)
    {
        case SIM_TYPE_MCSUB:
        {
            // The number of floats in each array, and the size of the arrays in bytes.
            unsigned int bufferSizeJ = BINS * PHOTON_BATCH * sizeof(float);
            unsigned int bufferSizeF = BINS * BINS * PHOTON_BATCH * sizeof(float);
            unsigned int bufferSizeSAE = 3 * PHOTON_BATCH * sizeof(float);
            _mBufferJ = [_mDevice newBufferWithLength:bufferSizeJ options:MTLResourceStorageModeShared];
            _mBufferF = [_mDevice newBufferWithLength:bufferSizeF options:MTLResourceStorageModeShared];
            _mBufferSAE = [_mDevice newBufferWithLength:bufferSizeSAE options:MTLResourceStorageModeShared];
            
            _mBufferRunParams = [_mDevice newBufferWithLength:sizeof(struct RunParams) options:MTLResourceStorageModeShared];
            struct RunParams* rpr = _mBufferRunParams.contents;
            memset(rpr, 0x0, sizeof(struct RunParams));
            memcpy(rpr, run_params, sizeof(struct RunParams));
            
            _mBufferTissPrams = [_mDevice newBufferWithLength:sizeof(struct TissueParams) options:MTLResourceStorageModeShared];
            struct TissueParams* tpr = _mBufferTissPrams.contents;
            memset(tpr, 0x0, sizeof(struct TissueParams));
            memcpy(tpr, tissParams, sizeof(struct TissueParams));
        }
        break;
        case SIM_TYPE_POLARIZATION_CBS_ELECTRIC_FIELDS:
        {
            _mBufferRunParams = [_mDevice newBufferWithLength:sizeof(struct RunParams) options:MTLResourceStorageModeShared];
            struct RunParams* rpr = _mBufferRunParams.contents;
            memset(rpr, 0x0, sizeof(struct RunParams));
            memcpy(rpr, run_params, sizeof(struct RunParams));
            
            _mBufferTissPrams = [_mDevice newBufferWithLength:sizeof(struct TissueParams) options:MTLResourceStorageModeShared];
            struct TissueParams* tpr = _mBufferTissPrams.contents;
            memset(tpr, 0x0, sizeof(struct TissueParams));
            memcpy(tpr, tissParams, sizeof(struct TissueParams));
            
            unsigned long buff_v_size = (long)run_params->Nx*run_params->Ny*run_params->Nz*sizeof(char);
            _mBufferV = [_mDevice newBufferWithLength: buff_v_size options:MTLResourceStorageModeShared];
            char* v_dev = _mBufferV.contents;
            memset(v_dev, 0x0, buff_v_size);
            memcpy(v_dev, v, buff_v_size);
            
            unsigned long bufferSizeF3D = (long)run_params->Nz*run_params->Ny*run_params->Nz * sizeof(float);
            _mBufferF3D = [_mDevice newBufferWithLength:bufferSizeF3D options:MTLResourceStorageModeShared];
            
            unsigned long bufferSizeRd = PHOTON_BATCH * sizeof(float);
            _mBufferRd = [_mDevice newBufferWithLength:bufferSizeRd options:MTLResourceStorageModeShared];
            float* rd_dev = _mBufferRd.contents;
            memset(rd_dev, 0x0, bufferSizeRd);
            
            unsigned long bufferSizeRd2D = (long) run_params->Nx*run_params->Ny * sizeof(float);
            _mBufferRd2D = [_mDevice newBufferWithLength:bufferSizeRd2D options:MTLResourceStorageModeShared];
            float* rd_dev_2d = _mBufferRd2D.contents;
            memset(rd_dev_2d, 0x0, bufferSizeRd2D);
            
            unsigned long bufferSizePhot2D = (long) run_params->Nx*run_params->Ny * sizeof(int);
            _mBufferPhot2D = [_mDevice newBufferWithLength:bufferSizePhot2D options:MTLResourceStorageModeShared];
            int* ph_dev_2d = _mBufferPhot2D.contents;
            memset(ph_dev_2d, 0x0, bufferSizePhot2D);
            
            unsigned long bufferSizePol2D = (long) run_params->Nx*run_params->Ny *  sizeof(float);
            _mBufferPol2DXX = [_mDevice newBufferWithLength:bufferSizePol2D options:MTLResourceStorageModeShared];
            float* pol_dev_2d_xx = _mBufferPol2DXX.contents;
            memset(pol_dev_2d_xx, 0x0, bufferSizePol2D);
            
            _mBufferPol2DXY = [_mDevice newBufferWithLength:bufferSizePol2D options:MTLResourceStorageModeShared];
            float* pol_dev_2d_xy = _mBufferPol2DXY.contents;
            memset(pol_dev_2d_xy, 0x0, bufferSizePol2D);
            
            _mBufferPol2DYX = [_mDevice newBufferWithLength:bufferSizePol2D options:MTLResourceStorageModeShared];
            float* pol_dev_2d_yx = _mBufferPol2DYX.contents;
            memset(pol_dev_2d_yx, 0x0, bufferSizePol2D);
            
            _mBufferPol2DYY = [_mDevice newBufferWithLength:bufferSizePol2D options:MTLResourceStorageModeShared];
            float* pol_dev_2d_yy = _mBufferPol2DYY.contents;
            memset(pol_dev_2d_yy, 0x0, bufferSizePol2D);
            
            _mBufferPol2DPhase0 = [_mDevice newBufferWithLength:bufferSizePol2D options:MTLResourceStorageModeShared];
            _mBufferPol2DPhase1 = [_mDevice newBufferWithLength:bufferSizePol2D options:MTLResourceStorageModeShared];
            _mBufferPol2DPhase2 = [_mDevice newBufferWithLength:bufferSizePol2D options:MTLResourceStorageModeShared];
            _mBufferPol2DPhase3 = [_mDevice newBufferWithLength:bufferSizePol2D options:MTLResourceStorageModeShared];
            float* pol_dev_2d_phase = _mBufferPol2DPhase0.contents;
            memset(pol_dev_2d_phase, 0x0, bufferSizePol2D);
            pol_dev_2d_phase = _mBufferPol2DPhase1.contents;
            memset(pol_dev_2d_phase, 0x0, bufferSizePol2D);
            pol_dev_2d_phase = _mBufferPol2DPhase2.contents;
            memset(pol_dev_2d_phase, 0x0, bufferSizePol2D);
            pol_dev_2d_phase = _mBufferPol2DPhase3.contents;
            memset(pol_dev_2d_phase, 0x0, bufferSizePol2D);
            
            unsigned long bufferSizePolScatt = (long) POL_CHANNELS * PHOTON_BATCH * MAX_SCATT_POL * sizeof(float);
            _mBufferPolVsScatt = [_mDevice newBufferWithLength:bufferSizePolScatt options:MTLResourceStorageModeShared];
            float* pol_scatt = _mBufferPolVsScatt.contents;
            memset(pol_scatt, 0x0, bufferSizePolScatt);
            
            unsigned long bufferSizePhotonCoords = (long) 3 * PHOTON_BATCH * MAX_SCATT * sizeof(float) ;
            _mBufferPhotonCoordinates = [_mDevice newBufferWithLength:bufferSizePhotonCoords options:MTLResourceStorageModeShared];
            float* ph_coords = _mBufferPhotonCoordinates.contents;
            memset(ph_coords, 0x0, bufferSizePhotonCoords);
            
            unsigned long bufferSizePol_initial = (long) run_params->Nx*run_params->Ny *  sizeof(float);
            
            _mBufferPhase02D = [_mDevice newBufferWithLength:bufferSizePol_initial options:MTLResourceStorageModeShared];
            float* phase_distrib = _mBufferPhase02D.contents;
            NSString *filename_phase = @"Phase_wide.txt";
            int numElements = loadFloatArrayFromFile(filename_phase, phase_distrib, rpr->Nx * rpr->Ny);
            if (numElements == -1) exit(1);
            
            _mBufferIntensity_radial = [_mDevice newBufferWithLength:bufferSizePol_initial options:MTLResourceStorageModeShared];
            float* int_distrib_rad = _mBufferIntensity_radial.contents;
            NSString *filename_int_distrib_rad = @"Intensity_LG3_norm.txt";
            numElements = loadFloatArrayFromFile(filename_int_distrib_rad, int_distrib_rad, rpr->Nx * rpr->Ny);
            if (numElements == -1) exit(1);
            
            _mBufferIntensity_azimuthal = [_mDevice newBufferWithLength:bufferSizePol_initial options:MTLResourceStorageModeShared];
            float* int_distrib_azim = _mBufferIntensity_azimuthal.contents;
            NSString *filename_int_distrib_azim = @"Intensity_LG3_norm.txt";
            numElements = loadFloatArrayFromFile(filename_int_distrib_azim, int_distrib_azim, rpr->Nx * rpr->Ny);
            if (numElements == -1) exit(1);
            
            _mBufferPhase_radial = [_mDevice newBufferWithLength:bufferSizePol_initial options:MTLResourceStorageModeShared];
            float* phase_distrib_rad = _mBufferPhase_radial.contents;
            NSString *filename_phase_distrib_rad = @"Phase_LG3_norm.txt";
            numElements = loadFloatArrayFromFile(filename_phase_distrib_rad, phase_distrib_rad, rpr->Nx * rpr->Ny);
            if (numElements == -1) exit(1);
            
            _mBufferPhase_azimuthal = [_mDevice newBufferWithLength:bufferSizePol_initial options:MTLResourceStorageModeShared];
            float* phase_distrib_azim = _mBufferPhase_azimuthal.contents;
            NSString *filename_phase_distrib_azim = @"Phase_LG3_norm.txt";
            numElements = loadFloatArrayFromFile(filename_phase_distrib_azim, phase_distrib_azim, rpr->Nx * rpr->Ny);
            if (numElements == -1) exit(1);
            
            if (run_params->speckleflag)
            {
                NSLog(@"This simulation computes speckle and mutual interference and therefore resource-demanding\n");
                unsigned long bufferSizeSpeckleData = (long)MAX_SPECKLES_PER_PIXEL * run_params->Nx*run_params->Ny *  sizeof(struct SPECKLE);
                _mBufferSpeckleData = [_mDevice newBufferWithLength:bufferSizeSpeckleData options:MTLResourceStorageModeShared];
                struct SPECKLE* SpeckleData = _mBufferSpeckleData.contents;
                memset(SpeckleData, 0x0, bufferSizeSpeckleData);
                
                unsigned long bufferSizeSpeckleCount = (long)run_params->Nx*run_params->Ny * sizeof(int);
                _mBufferSpeckleCount = [_mDevice newBufferWithLength:bufferSizeSpeckleCount options:MTLResourceStorageModeShared];
                int* SpeckleCounts = _mBufferSpeckleCount.contents;
                memset(SpeckleCounts, 0x0, bufferSizeSpeckleCount);
            }
        }
        break;
        case SIM_TYPE_RAMAN:
        {
            
        }
        break;
        case SIM_TYPE_MCXYZ:
        case SIM_TYPE_MCXYZ_TT:
        {
            _mBufferRunParams = [_mDevice newBufferWithLength:sizeof(struct RunParams) options:MTLResourceStorageModeShared];
            struct RunParams* rpr = _mBufferRunParams.contents;
            memset(rpr, 0x0, sizeof(struct RunParams));
            memcpy(rpr, run_params, sizeof(struct RunParams));
            
            _mBufferTissPrams = [_mDevice newBufferWithLength:sizeof(struct TissueParams) options:MTLResourceStorageModeShared];
            struct TissueParams* tpr = _mBufferTissPrams.contents;
            memset(tpr, 0x0, sizeof(struct TissueParams));
            memcpy(tpr, tissParams, sizeof(struct TissueParams));
            
            unsigned int buff_v_size = run_params->Nx*run_params->Ny*run_params->Nz*sizeof(char);
            _mBufferV = [_mDevice newBufferWithLength: buff_v_size options:MTLResourceStorageModeShared];
            char* v_dev = _mBufferV.contents;
            memset(v_dev, 0x0, buff_v_size);
            memcpy(v_dev, v, run_params->Nx*run_params->Ny*run_params->Nz*sizeof(char));
            
            unsigned long bufferSizeF3D = (long)run_params->Nz*run_params->Ny*run_params->Nz * sizeof(float);
            _mBufferF3D = [_mDevice newBufferWithLength:bufferSizeF3D options:MTLResourceStorageModeShared];
            
            unsigned long bufferSizeRd = PHOTON_BATCH * sizeof(float);
            _mBufferRd = [_mDevice newBufferWithLength:bufferSizeRd options:MTLResourceStorageModeShared];
            float* rd_dev = _mBufferRd.contents;
            memset(rd_dev, 0x0, bufferSizeRd);
            
            unsigned long bufferSizeRd2D = (long)PHOTON_BATCH * run_params->Nz*run_params->Ny * sizeof(float);
            _mBufferRd2D = [_mDevice newBufferWithLength:bufferSizeRd2D options:MTLResourceStorageModeShared];
            float* rd_dev_2d = _mBufferRd2D.contents;
            memset(rd_dev_2d, 0x0, bufferSizeRd2D);
            
            unsigned long bufferSizePhot2D = (long)PHOTON_BATCH *run_params->Nz*run_params->Ny * sizeof(int);
            _mBufferPhot2D = [_mDevice newBufferWithLength:bufferSizePhot2D options:MTLResourceStorageModeShared];
            int* ph_dev_2d = _mBufferPhot2D.contents;
            memset(ph_dev_2d, 0x0, bufferSizePhot2D);
            
        }
        break;
        default:
        {
            NSLog(@"Failed to find the MonteCarlo function.");
        }
        break;
    }
        
    _mBufferRunNum = [_mDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    _mBufferTimeSeed = [_mDevice newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

}

- (void) sendComputeCommand
{
    // Create a command buffer to hold commands.
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);
    
    // Start a compute pass.
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);
    
    [self encodeMCCommand:computeEncoder];
    
    // End the compute pass.
    [computeEncoder endEncoding];
    
    // Execute the command.
    [commandBuffer commit];
    
    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    [commandBuffer waitUntilCompleted];

}

- (void)encodeMCCommand:(id<MTLComputeCommandEncoder>)computeEncoder {
    
    // Encode the pipeline state object and its parameters.
    [computeEncoder setComputePipelineState:_mMCFunctionPSO];
    switch (_mRunParams.mckernelflag)
    {
        case SIM_TYPE_MCSUB:
        {
            [computeEncoder setBuffer:_mBufferRunParams offset:0 atIndex:0];
            [computeEncoder setBuffer:_mBufferTissPrams offset:0 atIndex:1];
            [computeEncoder setBuffer:_mBufferJ offset:0 atIndex:2];
            [computeEncoder setBuffer:_mBufferF offset:0 atIndex:3];
            [computeEncoder setBuffer:_mBufferSAE offset:0 atIndex:4];
            [computeEncoder setBuffer:_mBufferRunNum offset:0 atIndex:5];
        }
        break;
        case SIM_TYPE_POLARIZATION_CBS_ELECTRIC_FIELDS:
        {
            [computeEncoder setBuffer:_mBufferRunParams offset:0 atIndex:0];
            [computeEncoder setBuffer:_mBufferTissPrams offset:0 atIndex:1];
            [computeEncoder setBuffer:_mBufferV offset:0 atIndex:2];
            [computeEncoder setBuffer:_mBufferF3D offset:0 atIndex:3];
            [computeEncoder setBuffer:_mBufferRd offset:0 atIndex:4];
            [computeEncoder setBuffer:_mBufferRd2D offset:0 atIndex:5];
            [computeEncoder setBuffer:_mBufferPhot2D offset:0 atIndex:6];
            [computeEncoder setBuffer:_mBufferPol2DXX offset:0 atIndex:7];
            [computeEncoder setBuffer:_mBufferPol2DXY offset:0 atIndex:8];
            [computeEncoder setBuffer:_mBufferPol2DYX offset:0 atIndex:9];
            [computeEncoder setBuffer:_mBufferPol2DYY offset:0 atIndex:10];
            [computeEncoder setBuffer:_mBufferPol2DPhase0 offset:0 atIndex:11];
            [computeEncoder setBuffer:_mBufferPol2DPhase1 offset:0 atIndex:12];
            [computeEncoder setBuffer:_mBufferPol2DPhase2 offset:0 atIndex:13];
            [computeEncoder setBuffer:_mBufferPol2DPhase3 offset:0 atIndex:14];
            [computeEncoder setBuffer:_mBufferPolVsScatt offset:0 atIndex:15];
            [computeEncoder setBuffer:_mBufferPhotonCoordinates offset:0 atIndex:16];
            [computeEncoder setBuffer:_mBufferPhase02D offset:0 atIndex:17];
            [computeEncoder setBuffer:_mBufferSpeckleData offset:0 atIndex:18];
            [computeEncoder setBuffer:_mBufferSpeckleCount offset:0 atIndex:19];
            [computeEncoder setBuffer:_mBufferRunNum offset:0 atIndex:20];
            
            [computeEncoder setBuffer:_mBufferIntensity_radial offset:0 atIndex:21];
            [computeEncoder setBuffer:_mBufferIntensity_azimuthal offset:0 atIndex:22];
            [computeEncoder setBuffer:_mBufferPhase_radial offset:0 atIndex:23];
            [computeEncoder setBuffer:_mBufferPhase_azimuthal offset:0 atIndex:24];
            

        }
        break;
        case SIM_TYPE_RAMAN:
        {

        }
        break;
        case SIM_TYPE_MCXYZ:
        case SIM_TYPE_MCXYZ_TT:
        {
            [computeEncoder setBuffer:_mBufferRunParams offset:0 atIndex:0];
            [computeEncoder setBuffer:_mBufferTissPrams offset:0 atIndex:1];
            [computeEncoder setBuffer:_mBufferV offset:0 atIndex:2];
            [computeEncoder setBuffer:_mBufferF3D offset:0 atIndex:3];
            [computeEncoder setBuffer:_mBufferRd offset:0 atIndex:4];
            [computeEncoder setBuffer:_mBufferRd2D offset:0 atIndex:5];
            [computeEncoder setBuffer:_mBufferPhot2D offset:0 atIndex:6];
            [computeEncoder setBuffer:_mBufferRunNum offset:0 atIndex:7];
            [computeEncoder setBuffer:_mBufferTimeSeed offset:0 atIndex:8];
        }
        break;
        default:
        {
            NSLog(@"Failed to find the MonteCarlo function.");
        }
        break;
    }
    
    MTLSize gridSize = MTLSizeMake(PHOTON_BATCH, 1, 1);
    
    // Calculate a threadgroup size.
    NSUInteger threadGroupSize = _mMCFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > PHOTON_BATCH)
    {
        threadGroupSize = PHOTON_BATCH;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

- (void) setNumRunsSeed: (int*) numRuns :(float*) timeSeed;
{
    int* numruns = _mBufferRunNum.contents;
    *numruns = *numRuns;
    
    float* time_seed = _mBufferTimeSeed.contents;
    *time_seed = *timeSeed;
}


- (void)  getComputeResults: (float*) resJ  :(float*) resF :(float*) resF3D :(float*) resSAE :(float*) resRD :(float*) resRd2D :(int*) ResDetPhotons2D :(float*) ResPolXX2D :(float*) ResPolXY2D :(float*) ResPolYX2D :(float*) ResPolYY2D :(float*) ResPolPhase2D0 :(float*) ResPolPhase2D1 :(float*) ResPolPhase2D2 :(float*) ResPolPhase2D3 :(float*) PolVsScatt;
{
    switch (_mRunParams.mckernelflag)
    {
        case SIM_TYPE_MCSUB:
        {
            float* resultJ = _mBufferJ.contents;
            float* resultF = _mBufferF.contents;
            float* resultSAE = _mBufferSAE.contents;
            for (unsigned long index = 0; index < PHOTON_BATCH; index++)
            {
                for (unsigned long ipos = 1; ipos <= BINS ; ipos++)
                {
                    resJ[ipos]  += resultJ[index*BINS  + ipos];
                }
                
                for (unsigned long ipos = 1; ipos <= BINS ; ipos++)
                {
                    for (unsigned long ipos2 = 1; ipos2 <= BINS; ipos2++)
                    {
                        resF[ipos*BINS + ipos2]  += resultF[index* BINS*BINS  + ipos*BINS + ipos2];
                    }
                }
                
                for (unsigned long ipos = 0; ipos < 3; ipos++)
                {
                    resSAE[ipos]  += resultSAE[index*3 + ipos];
                }
            }
        }
        break;
        case SIM_TYPE_POLARIZATION_CBS_ELECTRIC_FIELDS:
        {
            float* resultF3D = _mBufferF3D.contents;
            struct RunParams* run_params = _mBufferRunParams.contents;
            for (unsigned long iposx = 0; iposx < run_params->Nx; iposx++)
            {
                for (unsigned long iposy = 0; iposy < run_params->Ny; iposy++)
                {
                    for (unsigned long iposz = 0; iposz < run_params->Nz; iposz++)
                    {
                        long iPosArray = (long)(iposx*run_params->Ny*run_params->Nz + iposy*run_params->Nz + iposz);
                        resF3D[iPosArray]  += resultF3D[iPosArray];
                    }
                }
            }
            
            float* resultRD = _mBufferRd.contents;
            for (unsigned long index = 0; index < PHOTON_BATCH; index++)
            {
                resRD[index] += resultRD[index];
            }
            
            if (!run_params->speckleflag)
            {
                float* resultRd2D = _mBufferRd2D.contents;
                int* resultPhot2D = _mBufferPhot2D.contents;
                float* resultsPol2DXX = _mBufferPol2DXX.contents;
                float* resultsPol2DXY = _mBufferPol2DXY.contents;
                float* resultsPol2DYX = _mBufferPol2DYX.contents;
                float* resultsPol2DYY = _mBufferPol2DYY.contents;
                float* resultsPol2DPhase0 = _mBufferPol2DPhase0.contents;
                float* resultsPol2DPhase1 = _mBufferPol2DPhase1.contents;
                float* resultsPol2DPhase2 = _mBufferPol2DPhase2.contents;
                float* resultsPol2DPhase3 = _mBufferPol2DPhase3.contents;
                
               // for (unsigned long index = 0; index < PHOTON_BATCH; index++)
               // {
                    for (unsigned long iposx = 0; iposx < run_params->Nx; iposx++)
                    {
                        for (unsigned long iposy = 0; iposy < run_params->Ny; iposy++)
                        {
                             long iPosArrayGPU = (long)(/*index*run_params->Ny*run_params->Nx +*/iposx*run_params->Ny + iposy);
                             long iPosArrayCPU = (long)(iposx*run_params->Ny + iposy);
                             resRd2D[iPosArrayCPU]  += resultRd2D[iPosArrayGPU];
                             ResDetPhotons2D[iPosArrayCPU]  += resultPhot2D[iPosArrayGPU];
                             ResPolXX2D[iPosArrayCPU] += resultsPol2DXX[iPosArrayGPU];
                             ResPolXY2D[iPosArrayCPU] += resultsPol2DXY[iPosArrayGPU];
                             ResPolYX2D[iPosArrayCPU] += resultsPol2DYX[iPosArrayGPU];
                             ResPolYY2D[iPosArrayCPU] += resultsPol2DYY[iPosArrayGPU];
                             ResPolPhase2D0[iPosArrayCPU] += resultsPol2DPhase0[iPosArrayGPU];
                             ResPolPhase2D1[iPosArrayCPU] += resultsPol2DPhase1[iPosArrayGPU];
                             ResPolPhase2D2[iPosArrayCPU] += resultsPol2DPhase2[iPosArrayGPU];
                             ResPolPhase2D3[iPosArrayCPU] += resultsPol2DPhase3[iPosArrayGPU];
                        }
                    }
               // }
                
                
                
            }
            else
            {
                int* resultPhot2D = _mBufferPhot2D.contents;
                float* resultRd2D = _mBufferRd2D.contents;
                float* resultsPol2DXX = _mBufferPol2DXX.contents;
                float* resultsPol2DXY = _mBufferPol2DXY.contents;
                float* resultsPol2DYX = _mBufferPol2DYX.contents;
                float* resultsPol2DYY = _mBufferPol2DYY.contents;
                float* resultsPol2DPhase0 = _mBufferPol2DPhase0.contents;
                float* resultsPol2DPhase1 = _mBufferPol2DPhase1.contents;
                float* resultsPol2DPhase2 = _mBufferPol2DPhase2.contents;
                float* resultsPol2DPhase3 = _mBufferPol2DPhase3.contents;
                
                /*
                for (unsigned long iposx = 0; iposx < run_params->Nx; iposx++)
                {
                    for (unsigned long iposy = 0; iposy < run_params->Ny; iposy++)
                    {
                            long iPosArray = (long)(iposx*run_params->Ny + iposy);
                            resRd2D[iPosArray]  += resultRd2D[iPosArray];
                            ResDetPhotons2D[iPosArray]  += resultPhot2D[iPosArray];
                            ResPolXX2D[iPosArray] += resultsPol2DXX[iPosArray];
                            ResPolXY2D[iPosArray] += resultsPol2DXY[iPosArray];
                            ResPolYX2D[iPosArray] += resultsPol2DYX[iPosArray];
                            ResPolYY2D[iPosArray] += resultsPol2DYY[iPosArray];
                            ResPolPhase2D0[iPosArray] += resultsPol2DPhase0[iPosArray];
                            ResPolPhase2D1[iPosArray] += resultsPol2DPhase1[iPosArray];
                            ResPolPhase2D2[iPosArray] += resultsPol2DPhase2[iPosArray];
                            ResPolPhase2D3[iPosArray] += resultsPol2DPhase3[iPosArray];
                    }
                }*/
                    
                int* resultSpeckleCount = _mBufferSpeckleCount.contents;
                struct SPECKLE* SpeckleData = _mBufferSpeckleData.contents;
                NSLog(@"Starting speckle computations:\n");
                    
                dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
                dispatch_apply(run_params->Nx, queue, ^(size_t iposx) {
                for (unsigned long iposy = 0; iposy < run_params->Ny; iposy++)
                {
                    long iPosArrayCPU = (long)(iposx * run_params->Ny + iposy);
                    int SpeckleCount = resultSpeckleCount[iposx * run_params->Ny + iposy];
                    NSLog(@"Working on pixel: %zu, %d, total speckles: %d\n", iposx, (int)iposy, SpeckleCount);
                            
                    float XX_accum_temp = resultsPol2DXX[iPosArrayCPU];
                    float XY_accum_temp = resultsPol2DXY[iPosArrayCPU];
                    float norm_sum = XX_accum_temp + XY_accum_temp;

                    float XX_accum = (norm_sum > 0.0f) ? (XX_accum_temp / norm_sum) : 0.0f;
                    float XY_accum = (norm_sum > 0.0f) ? (XY_accum_temp / norm_sum) : 0.0f;
                    
                    float Ryx_accum = resultRd2D[iPosArrayCPU] / resultPhot2D[iPosArrayCPU];
                    float Phase_accum = resultsPol2DPhase0[iPosArrayCPU] / resultPhot2D[iPosArrayCPU];
                    float Path_accum = resultsPol2DPhase1[iPosArrayCPU] / resultPhot2D[iPosArrayCPU];
                    float Scatt_accum = resultsPol2DPhase2[iPosArrayCPU] / resultPhot2D[iPosArrayCPU];
                    float Total_accum = resultsPol2DPhase3[iPosArrayCPU] / resultPhot2D[iPosArrayCPU];
                            
                    XX_accum     = (!isfinite(XX_accum))     ? 0.0f : XX_accum;
                    XY_accum     = (!isfinite(XY_accum))     ? 0.0f : XY_accum;
                    Ryx_accum    = (!isfinite(Ryx_accum))    ? 0.0f : Ryx_accum;
                    Phase_accum  = (!isfinite(Phase_accum))  ? 0.0f : Phase_accum;
                    Path_accum   = (!isfinite(Path_accum))   ? 0.0f : Path_accum;
                    Scatt_accum  = (!isfinite(Scatt_accum))  ? 0.0f : Scatt_accum;
                    Total_accum  = (!isfinite(Total_accum))  ? 0.0f : Total_accum;
                           
                    // Mutual accumulation variables
                    float XX_mutual = XX_accum;
                    float XY_mutual = XY_accum;
                    float Ryx_mutual = Ryx_accum;
                    float Phase_mutual_sum = Phase_accum;
                    float Path_mutual_sum = Path_accum;
                    float Scatt_mutual_sum = Scatt_accum;
                    float Total_mutual_sum = Total_accum;
                    float N_times_interferred = MC_ONE;
                                
                    for (unsigned long index_current = 0; index_current < SpeckleCount; index_current++)
                    {
                        long iPosArraySpeckle_current = (long)(index_current * run_params->Ny * run_params->Nx + iposx * run_params->Ny + iposy);
                                    
                        float XX_current_temp = SpeckleData[iPosArraySpeckle_current].XX;
                        float XY_current_temp = SpeckleData[iPosArraySpeckle_current].XY;
                        float norm_sum_current = XX_current_temp + XY_current_temp;

                        float XX_current = (norm_sum_current > 0.0f) ? (XX_current_temp / norm_sum_current) : 0.0f;
                        float XY_current = (norm_sum_current > 0.0f) ? (XY_current_temp / norm_sum_current) : 0.0f;
                        float Ryx_current = SpeckleData[iPosArraySpeckle_current].Ryx;

                        float Phase_current = SpeckleData[iPosArraySpeckle_current].Phase;
                        float Path_current = SpeckleData[iPosArraySpeckle_current].Path;
                        float Scatt_current = SpeckleData[iPosArraySpeckle_current].Scatt;

                        // Sanitize values
                        XX_current = (!isfinite(XX_current)) ? 0.0f : XX_current;
                        XY_current = (!isfinite(XY_current)) ? 0.0f : XY_current;
                        Ryx_current = (!isfinite(Ryx_current)) ? 0.0f : Ryx_current;
                        Phase_current = (!isfinite(Phase_current)) ? 0.0f : Phase_current;
                        Path_current = (!isfinite(Path_current)) ? 0.0f : Path_current;
                        Scatt_current = (!isfinite(Scatt_current)) ? 0.0f : Scatt_current;
                        
                        // --- PHASE DIFFERENCES (wrapped)
                        float dPhase = atan2(sin(Phase_accum - Phase_current), cos(Phase_accum - Phase_current));
                        float dPath  = atan2(sin(Path_accum  - Path_current),  cos(Path_accum  - Path_current));
                        float dScatt = atan2(sin(Scatt_accum - Scatt_current), cos(Scatt_accum - Scatt_current));

                        // Wrapped total phase difference
                        float phase_total_diff_lg     = atan2(sin(dPhase + dPath + dScatt), cos(dPhase + dPath + dScatt));
                        float phase_total_diff_linear = atan2(sin(dPath + dScatt), cos(dPath + dScatt));


                        // Interference terms
                        float interference_term_lg     = sin(phase_total_diff_lg);       // for polarised fields
                        float interference_term_linear = sin(phase_total_diff_linear);   // for intensity-like channel

                        // Longitudinal coherence envelope
                        float path_accum_mm  = (Path_accum * run_params->lambda) / (MC_TWO * PI);
                        float path_current_mm = (Path_current * run_params->lambda) / (MC_TWO * PI);
                        float delta_path = (path_current_mm - path_accum_mm) / run_params->lc;

                        float lc_contribution = exp(-delta_path * delta_path);
                        lc_contribution = (!isfinite(lc_contribution) || run_params->lc <= 0.01f) ? 0.0f : lc_contribution;

                        // Mutual intensity calculation
                        float I_xx = (XX_accum + XX_current + MC_TWO * sqrt(XX_accum * XX_current) * interference_term_linear * lc_contribution) / 4.0f;
                        float I_xy = (XY_accum + XY_current + MC_TWO * sqrt(XY_accum * XY_current) * interference_term_linear * lc_contribution) / 4.0f;
                        float I_ryx = (Ryx_accum + Ryx_current + MC_TWO * sqrt(Ryx_accum * Ryx_current) * interference_term_linear * lc_contribution) / 4.0f;

                        // Accumulate results
                        XX_mutual += I_xx;
                        XY_mutual += I_xy;
                        Ryx_mutual += I_ryx;

                        Phase_mutual_sum += Phase_current;
                        Path_mutual_sum  += Path_current;
                        Scatt_mutual_sum += Scatt_current;

                        float total_phase_current = Phase_current + Path_current + Scatt_current;
                        total_phase_current = atan2(sin(total_phase_current), cos(total_phase_current));
                        Total_mutual_sum += total_phase_current;

                        N_times_interferred += 1.0f;
                    }
                    
                    // Store final values
                    resRd2D[iPosArrayCPU]         += Ryx_mutual;
                    ResDetPhotons2D[iPosArrayCPU] += N_times_interferred;
                    ResPolXX2D[iPosArrayCPU]      += XX_mutual;
                    ResPolXY2D[iPosArrayCPU]      += XY_mutual;
                    ResPolPhase2D0[iPosArrayCPU]  += Phase_mutual_sum;
                    ResPolPhase2D1[iPosArrayCPU]  += Path_mutual_sum;
                    ResPolPhase2D2[iPosArrayCPU]  += Scatt_mutual_sum;
                    ResPolPhase2D3[iPosArrayCPU]  += Total_mutual_sum;
                }
                });
            }
            
            float* resultPolVsScatt = _mBufferPolVsScatt.contents;
            for (unsigned long index = 0; index < PHOTON_BATCH; index++)
            {
                for (unsigned long ipos = 0; ipos <= MAX_SCATT_POL; ipos++)
                {
                    long iPosArrayGPU = (long)(index*MAX_SCATT_POL);
                    PolVsScatt[ipos]  += resultPolVsScatt[iPosArrayGPU + ipos];
                    PolVsScatt[MAX_SCATT_POL + ipos]  += resultPolVsScatt[PHOTON_BATCH*MAX_SCATT_POL + iPosArrayGPU + ipos];
                    PolVsScatt[2*MAX_SCATT_POL + ipos]  += resultPolVsScatt[2*PHOTON_BATCH*MAX_SCATT_POL + iPosArrayGPU + ipos];
                    PolVsScatt[3*MAX_SCATT_POL + ipos]  += resultPolVsScatt[3*PHOTON_BATCH*MAX_SCATT_POL + iPosArrayGPU + ipos];
                }
            }
            

        }
        break;
        case SIM_TYPE_RAMAN:
        {
        }
        break;
        case SIM_TYPE_MCXYZ:
        case SIM_TYPE_MCXYZ_TT:
        {
            
            float* resultF3D = _mBufferF3D.contents;
            struct RunParams* run_params = _mBufferRunParams.contents;
            for (unsigned long iposx = 0; iposx < run_params->Nx; iposx++)
            {
                for (unsigned long iposy = 0; iposy < run_params->Ny; iposy++)
                {
                    for (unsigned long iposz = 0; iposz < run_params->Nz; iposz++)
                    {
                        long iPosArray = (long)(iposx*run_params->Ny*run_params->Nz + iposy*run_params->Nz + iposz);
                        resF3D[iPosArray]  += resultF3D[iPosArray];
                    }
                }
            }
            
            float* resultRD = _mBufferRd.contents;
            for (unsigned long index = 0; index < PHOTON_BATCH; index++)
            {
                resRD[index] += resultRD[index];
            }
            
            float* resultRd2D = _mBufferRd2D.contents;
            int* resultPhot2D = _mBufferPhot2D.contents;
            
            for (unsigned long index = 0; index < PHOTON_BATCH; index++)
            {
                for (unsigned long iposx = 0; iposx < run_params->Nx; iposx++)
                {
                    for (unsigned long iposy = 0; iposy < run_params->Ny; iposy++)
                    {
                        long iPosArrayGPU = (long)(index*run_params->Ny*run_params->Nx + iposx*run_params->Ny + iposy);
                        long iPosArrayCPU = (long)(iposx*run_params->Ny + iposy);
                        resRd2D[iPosArrayCPU]  += resultRd2D[iPosArrayGPU];
                        ResDetPhotons2D[iPosArrayCPU]  += resultPhot2D[iPosArrayGPU];
                    }
                }
            }
            
        }
        break;
        default:
        {
            NSLog(@"Failed to find the MonteCarlo function.");
        }
        break;
    }
    
}
@end
