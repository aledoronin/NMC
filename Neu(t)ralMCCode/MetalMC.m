/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A class to manage all of the Metal objects this app creates.
*/

#import "MetalMC.h"

#include "definitions.h"

// The number of floats in each array, and the size of the arrays in bytes.
const unsigned int arrayLength = BINS;
const unsigned int photons_batch = PHOTON_BATCH;
const unsigned int bufferSizeJ = arrayLength * photons_batch * sizeof(float);
const unsigned int bufferSizeF = arrayLength * arrayLength * photons_batch * sizeof(float);
const unsigned int bufferSizeSAE = 3 * photons_batch * sizeof(float);

@implementation MetalMC
{
    id<MTLDevice> _mDevice;
    
    // The compute pipeline generated from the compute kernel in the .metal shader file.
    id<MTLComputePipelineState> _mMCFunctionPSO;
    
    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _mCommandQueue;
    
    // Buffers to hold data.
    id<MTLBuffer> _mBufferJ;
    id<MTLBuffer> _mBufferF;
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

        id<MTLFunction> MonteCarloFunction = [defaultLibrary newFunctionWithName:@"MonteCarloKernel"];
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

- (void) prepareData
{
    // Allocate three buffers to hold our initial data and the result.
    _mBufferJ = [_mDevice newBufferWithLength:bufferSizeJ options:MTLResourceStorageModeShared];
    _mBufferF = [_mDevice newBufferWithLength:bufferSizeF options:MTLResourceStorageModeShared];
    _mBufferSAE = [_mDevice newBufferWithLength:bufferSizeSAE options:MTLResourceStorageModeShared];
    _mBufferRunNum = [_mDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
    
    [self clearResultsBuffer];
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
    [computeEncoder setBuffer:_mBufferJ offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferF offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferSAE offset:0 atIndex:2];
    [computeEncoder setBuffer:_mBufferRunNum offset:0 atIndex:3];
    
    MTLSize gridSize = MTLSizeMake(photons_batch, 1, 1);
    
    // Calculate a threadgroup size.
    NSUInteger threadGroupSize = _mMCFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > photons_batch)
    {
        threadGroupSize = photons_batch;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

- (void) clearResultsBuffer
{
    float* dataPtrJ = _mBufferJ.contents;
    float* dataPtrF = _mBufferF.contents;
    float* dataPtrSAE = _mBufferSAE.contents;
    
    memset(dataPtrJ, 0x0, bufferSizeJ);
    memset(dataPtrF, 0x0, bufferSizeF);
    memset(dataPtrSAE, 0x0, bufferSizeSAE);
}

- (void) setNumRuns: (int*) numRuns;
{
    int* numruns = _mBufferRunNum.contents;
    *numruns = *numRuns;
}


- (void) getComputeResults: (float*) resJ  :(float*) resF :(float*) resSAE;
{
    float* resultJ = _mBufferJ.contents;
    float* resultF = _mBufferF.contents;
    float* resultSAE = _mBufferSAE.contents;
    
    for (unsigned long index = 0; index < PHOTON_BATCH; index++)
    {
        for (unsigned long ipos = 1; ipos <= BINS; ipos++)
        {
            resJ[ipos]  += resultJ[index*BINS + ipos];
        }
        
        for (unsigned long ipos = 1; ipos <= BINS; ipos++)
        {
            for (unsigned long ipos2 = 1; ipos2 <= BINS; ipos2++)
            {
                resF[ipos*BINS + ipos2]  += resultF[index*BINS*BINS + ipos*BINS + ipos2];
            }
        }
        
        for (unsigned long ipos = 0; ipos < 3; ipos++)
        {
            resSAE[ipos]  += resultSAE[index*3 + ipos];
        }
    }
    
}
@end
