/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of NMC: a unified, power-efficient platform platform for photon transport simulations
//  accelerated by Metal/GPU computing and Machine Learning
//  created by Alexander Doronin
//  Source code:    https://github.com/aledoronin
//  Web:            http://www.lighttransport.net/
//  Licence:        BSD-3-Clause, see LICENCE file
//  Contributors:   the respective contributors, as shown by the AUTHORS file
//  Year conceived: 2023
//  This file:      MetalMC.h, provides the main interface to MC GPU Computing is based on Apple's Performing Calculations on
//                  a GPU example: https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu
//
//  Copyright © 2019 Apple Inc.
//  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
/*---------------------------------------------------------------------------------------------------------------------*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "definitions.h"

NS_ASSUME_NONNULL_BEGIN

@interface MetalMC : NSObject
- (instancetype) initWithDevice: (id<MTLDevice>) device :(enum MC_SIM_TYPE) kernel_type ;
- (void) prepareData: (struct RunParams*) run_params :(struct TissueParams*) tissParams :(char*) v;
- (void) sendComputeCommand;
- (void) getComputeResults: (float*) resJ :(float*) resF :(float*) resF3D :(float*) resSAE :(float*) Rd :(float*) Rd2D :(int*) ResDetPhotons2D0 :(float*) RdPolXX2D :(float*) RdPolXY2D :(float*) RdPolYX2D :(float*) RdPolYY2D :(float*) RdPolPhase2D0 :(float*) ResPolPhase2D1 :(float*) ResPolPhase2D2 :(float*) ResPolPhase2D3 :(float*) PolVsScatt;
- (void) setNumRunsSeed: (int*) NumRuns :(float*) TimeSeed;
@end

NS_ASSUME_NONNULL_END
