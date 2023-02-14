/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of Neu(t)ralMC: a unified, power-efficient platform platform for photon transport simulations
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
    id<MTLBuffer> _mBufferRunNum;
}

- (instancetype) initWithDevice: (id<MTLDevice>) device
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
        if (mc_kernel_type == 0)
            MonteCarloFunction = [defaultLibrary newFunctionWithName:@"MCSubKernel"];
        else if (mc_kernel_type == 1)
            MonteCarloFunction = [defaultLibrary newFunctionWithName:@"MCXYZKernel"];
        
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
    if (mc_kernel_type == 0)
    {
        // The number of floats in each array, and the size of the arrays in bytes.
        unsigned int bufferSizeJ = BINS * PHOTON_BATCH * sizeof(float);
        unsigned int bufferSizeF = BINS * BINS * PHOTON_BATCH * sizeof(float);
        unsigned int bufferSizeSAE = 3 * PHOTON_BATCH * sizeof(float);
        _mBufferJ = [_mDevice newBufferWithLength:bufferSizeJ options:MTLResourceStorageModeShared];
        _mBufferF = [_mDevice newBufferWithLength:bufferSizeF options:MTLResourceStorageModeShared];
        _mBufferSAE = [_mDevice newBufferWithLength:bufferSizeSAE options:MTLResourceStorageModeShared];
    }
    else if (mc_kernel_type == 1)
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

    }
        
    _mBufferRunNum = [_mDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

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
    if (mc_kernel_type == 0)
    {
        [computeEncoder setBuffer:_mBufferJ offset:0 atIndex:0];
        [computeEncoder setBuffer:_mBufferF offset:0 atIndex:1];
        [computeEncoder setBuffer:_mBufferSAE offset:0 atIndex:2];
        [computeEncoder setBuffer:_mBufferRunNum offset:0 atIndex:3];
    }
    else if (mc_kernel_type == 1)
    {
        [computeEncoder setBuffer:_mBufferRunParams offset:0 atIndex:0];
        [computeEncoder setBuffer:_mBufferTissPrams offset:0 atIndex:1];
        [computeEncoder setBuffer:_mBufferV offset:0 atIndex:2];
        [computeEncoder setBuffer:_mBufferF3D offset:0 atIndex:3];
        [computeEncoder setBuffer:_mBufferRunNum offset:0 atIndex:4];
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

- (void) setNumRuns: (int*) numRuns;
{
    int* numruns = _mBufferRunNum.contents;
    *numruns = *numRuns;
}

- (void) getComputeResults: (float*) resJ  :(float*) resF :(float*) resF3D :(float*) resSAE;
{
    if (mc_kernel_type == 0)
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
    else if (mc_kernel_type == 1)
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
    }
    
}
@end
