/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A class to manage all of the Metal objects this app creates.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


NS_ASSUME_NONNULL_BEGIN

@interface MetalMC : NSObject
- (instancetype) initWithDevice: (id<MTLDevice>) device;
- (void) prepareData;
- (void) sendComputeCommand;
- (void) getComputeResults: (float*) Csph :(float*) Ccyl :(float*) Cpla;
- (void) setNumRuns: (int*) NumRuns;
@end


NS_ASSUME_NONNULL_END
