/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of NMC: a unified, power-efficient platform platform for photon transport simulations
//  accelerated by Metal/GPU computing and Machine Learning
//  created by Alexander Doronin
//  Source code:    https://github.com/aledoronin
//  Web:            http://www.lighttransport.net/
//  Licence:        BSD-3-Clause, see LICENCE file
//  Contributors:   the respective contributors, as shown by the AUTHORS file
//  Year conceived: 2023
//  This file:      helpers.metal, contains Metal versions of routines for handling voxelized represination of
//  scattering media, Fresnel reflectance and floats comparison by Steven Jacques, Ting LI, Alexander Doronin
//  Random Number Generator implemetation is based on "Efficient pseudo-random number generation for monte-carlo simulations
//  using graphic processors" by Siddhant Mohanty et al 2012 J. Phys.: Conf. Ser. 368 012024 and the Loki package by Youssef Victor
//  https://github.com/YoussefV/Loki

#include <metal_stdlib>
#include "definitions.h"
using namespace metal;

/* Compare two floats in GPU memory */
inline bool FltEq(float dbl1, float dbl2, int error = 1)
{
    float errorDbl2 = dbl2 * FLT_EPSILON * (1 + error);
    return (dbl1 >= dbl2 - errorDbl2) && (dbl1 <= dbl2 + errorDbl2);
}

inline bool SameVoxel(float x1, float y1, float z1, float x2, float y2, float z2, float dx, float dy, float dz)
{
    float xmin = min((floor)(x1 / dx), (floor)(x2 / dx))*dx;
    float ymin = min((floor)(y1 / dy), (floor)(y2 / dy))*dy;
    float zmin = min((floor)(z1 / dz), (floor)(z2 / dz))*dz;
    float xmax = xmin + dx;
    float ymax = ymin + dy;
    float zmax = zmin + dz;
    bool sv = 0;
    
    sv = (x1 <= xmax && x2 <= xmax && y1 <= ymax && y2 <= ymax && z1<zmax && z2 <= zmax);
    return (sv);
}


/* Function to numerically solve for cos(theta) using Newton-Raphson method */
inline float SolveCosThetaRayleigh(float r) {
    float cos_theta = 0.0; // Initial guess
    float f, f_prime;
    int max_iter = 100;
    float tolerance = 1e-6;

    for (int i = 0; i < max_iter; i++) {
        // Evaluate the CDF and its derivative
        f = 0.5 * (cos_theta + (cos_theta * cos_theta * cos_theta) / 3.0 + 1.0) - r;
        f_prime = 0.5 * (1.0 + cos_theta * cos_theta);

        // Update cos_theta using Newton-Raphson
        float delta = f / f_prime;
        cos_theta -= delta;

        // Check for convergence
        if (fabs(delta) < tolerance) {
            return cos_theta;
        }
    }

    // If no convergence, return fallback
    return cos_theta;
}


inline float FindVoxelFace3(float x1, float y1, float z1, float dx, float dy, float dz, float ux, float uy, float uz) {
    int ix1 = (int)floor(x1 / dx);
    int iy1 = (int)floor(y1 / dy);
    int iz1 = (int)floor(z1 / dz);

    int ix2 = ux >= 0.0 ? ix1 + 1 : ix1;
    int iy2 = uy >= 0.0 ? iy1 + 1 : iy1;
    int iz2 = uz >= 0.0 ? iz1 + 1 : iz1;

    float xs = ux != 0.0 ? fabs((ix2 * dx - x1) / ux) : INFINITY;
    float ys = uy != 0.0 ? fabs((iy2 * dy - y1) / uy) : INFINITY;
    float zs = uz != 0.0 ? fabs((iz2 * dz - z1) / uz) : INFINITY;

    return min3(xs, ys, zs);
}

inline float FindVoxelFace2(float x1, float y1, float z1, float x2, float y2, float z2, float dx, float dy, float dz, float ux, float uy, float uz)
{
    int ix1 = (int)floor(x1 / dx);
    int iy1 = (int)floor(y1 / dy);
    int iz1 = (int)floor(z1 / dz);

    int ix2, iy2, iz2;
    if (ux >= 0.0)
        ix2 = ix1 + 1;
    else
        ix2 = ix1;

    if (uy >= 0.0)
        iy2 = iy1 + 1;
    else
        iy2 = iy1;

    if (uz >= 0.0)
        iz2 = iz1 + 1;
    else
        iz2 = iz1;

    float xs = fabs((ix2*dx - x1) / ux);
    float ys = fabs((iy2*dy - y1) / uy);
    float zs = fabs((iz2*dz - z1) / uz);

    float s = min3(xs, ys, zs);

    return (s);
}

inline float FindVoxelFace(float x1,float y1,float z1, float x2, float y2, float z2,float dx,float dy,float dz, float ux, float uy, float uz)
{
    float x_1 = x1/dx;
    float y_1 = y1/dy;
    float z_1 = z1/dz;
    float x_2 = x2/dx;
    float y_2 = y2/dy;
    float z_2 = z2/dz;
    float fx_1 = floor(x_1) ;
    float fy_1 = floor(y_1) ;
    float fz_1 = floor(z_1) ;
    float fx_2 = floor(x_2) ;
    float fy_2 = floor(y_2) ;
    float fz_2 = floor(z_2) ;
    float x=0, y=0, z=0, x0=0, y0=0, z0=0, s=0;
    
    if ((fx_1 != fx_2) && (fy_1 != fy_2) && (fz_1 != fz_2) ) { //#10
        fx_2=fx_1+SIGN(fx_2-fx_1);//added
        fy_2=fy_1+SIGN(fy_2-fy_1);//added
        fz_2=fz_1+SIGN(fz_2-fz_1);//added
        
        x = (max(fx_1,fx_2)-x_1)/ux;
        y = (max(fy_1,fy_2)-y_1)/uy;
        z = (max(fz_1,fz_2)-z_1)/uz;
        if (x == min3(x,y,z)) {
            x0 = max(fx_1,fx_2);
            y0 = (x0-x_1)/ux*uy+y_1;
            z0 = (x0-x_1)/ux*uz+z_1;
        }
        else if (y == min3(x,y,z)) {
            y0 = max(fy_1,fy_2);
            x0 = (y0-y_1)/uy*ux+x_1;
            z0 = (y0-y_1)/uy*uz+z_1;
        }
        else {
            z0 = max(fz_1,fz_2);
            y0 = (z0-z_1)/uz*uy+y_1;
            x0 = (z0-z_1)/uz*ux+x_1;
        }
    }
    else if ( (fx_1 != fx_2) && (fy_1 != fy_2) ) { //#2
        fx_2=fx_1+SIGN(fx_2-fx_1);//added
        fy_2=fy_1+SIGN(fy_2-fy_1);//added
        x = (max(fx_1,fx_2)-x_1)/ux;
        y = (max(fy_1,fy_2)-y_1)/uy;
        if (x == min(x,y)) {
            x0 = max(fx_1,fx_2);
            y0 = (x0-x_1)/ux*uy+y_1;
            z0 = (x0-x_1)/ux*uz+z_1;
        }
        else {
            y0 = max(fy_1, fy_2);
            x0 = (y0-y_1)/uy*ux+x_1;
            z0 = (y0-y_1)/uy*uz+z_1;
        }
    }
    else if ( (fy_1 != fy_2) &&(fz_1 != fz_2) ) { //#3
        fy_2=fy_1+SIGN(fy_2-fy_1);//added
        fz_2=fz_1+SIGN(fz_2-fz_1);//added
        y = (max(fy_1,fy_2)-y_1)/uy;
        z = (max(fz_1,fz_2)-z_1)/uz;
        if (y == min(y,z)) {
            y0 = max(fy_1,fy_2);
            x0 = (y0-y_1)/uy*ux+x_1;
            z0 = (y0-y_1)/uy*uz+z_1;
        }
        else {
            z0 = max(fz_1, fz_2);
            x0 = (z0-z_1)/uz*ux+x_1;
            y0 = (z0-z_1)/uz*uy+y_1;
        }
    }
    else if ( (fx_1 != fx_2) && (fz_1 != fz_2) ) { //#4
        fx_2=fx_1+SIGN(fx_2-fx_1);//added
        fz_2=fz_1+SIGN(fz_2-fz_1);//added
        x = (max(fx_1,fx_2)-x_1)/ux;
        z = (max(fz_1,fz_2)-z_1)/uz;
        if (x == min(x,z)) {
            x0 = max(fx_1,fx_2);
            y0 = (x0-x_1)/ux*uy+y_1;
            z0 = (x0-x_1)/ux*uz+z_1;
        }
        else {
            z0 = max(fz_1, fz_2);
            x0 = (z0-z_1)/uz*ux+x_1;
            y0 = (z0-z_1)/uz*uy+y_1;
        }
    }
    else if (fx_1 != fx_2) { //#5
        fx_2=fx_1+SIGN(fx_2-fx_1);//added
        x0 = max(fx_1,fx_2);
        y0 = (x0-x_1)/ux*uy+y_1;
        z0 = (x0-x_1)/ux*uz+z_1;
    }
    else if (fy_1 != fy_2) { //#6
        fy_2=fy_1+SIGN(fy_2-fy_1);//added
        y0 = max(fy_1, fy_2);
        x0 = (y0-y_1)/uy*ux+x_1;
        z0 = (y0-y_1)/uy*uz+z_1;
    }
    else { //#7
        z0 = max(fz_1, fz_2);
        fz_2=fz_1+SIGN(fz_2-fz_1);//added
        x0 = (z0-z_1)/uz*ux+x_1;
        y0 = (z0-z_1)/uz*uy+y_1;
    }
    //s = ( (x0-fx_1)*dx + (y0-fy_1)*dy + (z0-fz_1)*dz )/3;
    //s = sqrt( SQR((x0-x_1)*dx) + SQR((y0-y_1)*dy) + SQR((z0-z_1)*dz) );
    //s = sqrt(SQR(x0-x_1)*SQR(dx) + SQR(y0-y_1)*SQR(dy) + SQR(z0-z_1)*SQR(dz));
    s = sqrt( SQR((x0-x_1)*dx) + SQR((y0-y_1)*dy) + SQR((z0-z_1)*dz));
    return (s);
}

class Fresnel {
    
public:
    /// This is a Metal Shading Language (MSL) function equivalent to the add_arrays() C function, used to perform the calculation on a GPU.
    
    /************************************************************
     *    FRESNEL REFLECTANCE
     * Computes reflectance as photon passes from medium 1 to
     * medium 2 with refractive indices n1,n2. Incident
     * angle a1 is specified by cosine value ca1 = cos(a1).
     * Program returns value of transmitted angle a1 as
     * value in *ca2_Ptr = cos(a2).
     ****/
    
    static float RFresnel(float n1_par,        /* incident refractive index.*/
                   float n2_par,        /* transmit refractive index.*/
                   float ca1,        /* cosine of the incident */
                   /* angle a1, 0<a1<90 degrees. */
                   thread float *ca2_Ptr)     /* pointer to the cosine */
    /* of the transmission */
    /* angle a2, a2>0. */
    {
        
        float r = MC_ZERO;
        
        if(FltEq(n1_par,n2_par)) { /** matched boundary. **/
            *ca2_Ptr = ca1;
            r = MC_ZERO;
        }
        else if(ca1 > COSZERO) { /** normal incidence. **/
            *ca2_Ptr = ca1;
            r = (n2_par-n1_par)/(n2_par+n1_par);
            r *= r;
        }
        else if(ca1 < COS90D)  {    /** very slanted. **/
            *ca2_Ptr = MC_ZERO;
            r = MC_ONE;
        }
        else  {                      /** general. **/
            float sa1, sa2; /* sine of incident and transmission angles. */
            float ca2;      /* cosine of transmission angle. */
            sa1 = sqrt(MC_ONE - ca1*ca1);
            sa2 = n1_par * sa1 / n2_par;
            if(sa2 >= MC_ONE) {
                /* double check for total internal reflection. */
                *ca2_Ptr = MC_ZERO;
                r = MC_ONE;
            }
            else {
                float cap, cam;    /* cosines of sum ap or diff am of the two */
                /* angles: ap = a1 + a2, am = a1 - a2. */
                float sap, sam;    /* sines. */
                *ca2_Ptr = ca2 = sqrt(MC_ONE - sa2*sa2);
                cap = ca1*ca2 - sa1*sa2; /* c+ = cc - ss. */
                cam = ca1*ca2 + sa1*sa2; /* c- = cc + ss. */
                sap = sa1*ca2 + ca1*sa2; /* s+ = sc + cs. */
                sam = sa1*ca2 - ca1*sa2; /* s- = sc - cs. */
                r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam);
                /* rearranged for speed. */
            }
        }
        return(r);
    } /******** END SUBROUTINE **********/
    
    static void Snell(float n1, float n2, thread float *dux, thread float *duy, thread float *duz)
    {
        float dux_loc = *dux;
        float duy_loc = *duy;
        float duz_loc = *duz;
        
        float snell_calc = MC_ONE - (MC_ONE - duz_loc*duz_loc) * n1 * n1 / n2 / n2;
        if (snell_calc < MC_ZERO)
        {
            if (fabs(duz_loc) > MC_ONE)
            {
                if (duz_loc < MC_ZERO)
                    *duz = -MC_ONE;
                else
                    *duz = MC_ONE;
            }
            return;
        }
        *dux = dux_loc * n1 / n2;
        *duy = duy_loc * n1 / n2;
        *duz = sqrt(snell_calc);
    }
    
};


class RandomGen {
private:
    thread unsigned int state1, state2, state3, state4; // Separate state variables

    // Improved Tausworthe step with better bit scrambling
    unsigned int TausStep(unsigned int z, int s1, int s2, int s3, unsigned int M) {
        unsigned int b = ((z << s1) ^ z) >> s2;
        return ((z & M) << s3) ^ b;
    }

    // XORShift step for additional entropy mixing
    unsigned int XorShift(unsigned int x) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

public:
    // Optimized constructor with high-entropy seed initialization
    thread RandomGen(unsigned int seed1, unsigned int seed2, unsigned int seed3) {
        // Strong XORShift mixing on seeds
        seed1 = XorShift(seed1);
        seed2 = XorShift(seed2);
        seed3 = XorShift(seed3);

        // Initialize states using Tausworthe and LCG steps
        state1 = TausStep(seed1, 13, 19, 12, 429496729UL);
        state2 = TausStep(seed2, 2, 25, 4, 4294967288UL);
        state3 = TausStep(seed3, 3, 11, 17, 429496280UL);
        state4 = XorShift(seed1 + seed2 + seed3) ^ 1013904223UL; // Strong mixed state

        // Final XOR mix to avoid correlated states
        state1 ^= state4;
        state2 ^= state1;
        state3 ^= state2;
    }

    // Generate a high-quality random float in range [0, 1)
    thread float rand() {
        // Apply improved Tausworthe steps
        state1 = TausStep(state1, 13, 19, 12, 429496729UL);
        state2 = TausStep(state2, 2, 25, 4, 4294967288UL);
        state3 = TausStep(state3, 3, 11, 17, 429496280UL);
        state4 = XorShift(state4); // Strong XOR shift step

        // Final XOR mix to further decorrelate outputs
        unsigned int result = state1 ^ state2 ^ state3 ^ state4;

        // Convert to high-precision float in [0, 1) using 24-bit mantissa
        return fma((result >> 8), (1.0f / 16777216.0f), 0.0f);  // Ensures full 24-bit precision
    }
};


/*
class RandomGen {
private:
    thread float seed;
    unsigned TausStep(const unsigned z, const int s1, const int s2, const int s3, const unsigned M)
    {
        unsigned b=(((z << s1) ^ z) >> s2);
        return (((z & M) << s3) ^ b);
    }

public:
    thread RandomGen(const unsigned seed1, const unsigned seed2, const unsigned seed3) {
        unsigned seed = seed1 * 1099087573UL;
        unsigned seedb = seed2 * 1099087573UL;
        unsigned seedc = seed3 * 1099087573UL;

        // Round 1: Randomise seed
        unsigned z1 = TausStep(seed,13,19,12,429496729UL);
        unsigned z2 = TausStep(seed,2,25,4,4294967288UL);
        unsigned z3 = TausStep(seed,3,11,17,429496280UL);
        unsigned z4 = (1664525*seed + 1013904223UL);

        // Round 2: Randomise seed again using second seed
        unsigned r1 = (z1^z2^z3^z4^seedb);

        z1 = TausStep(r1,13,19,12,429496729UL);
        z2 = TausStep(r1,2,25,4,4294967288UL);
        z3 = TausStep(r1,3,11,17,429496280UL);
        z4 = (1664525*r1 + 1013904223UL);

        // Round 3: Randomise seed again using third seed
        r1 = (z1^z2^z3^z4^seedc);

        z1 = TausStep(r1,13,19,12,429496729UL);
        z2 = TausStep(r1,2,25,4,4294967288UL);
        z3 = TausStep(r1,3,11,17,429496280UL);
        z4 = (1664525*r1 + 1013904223UL);

        this->seed = (z1^z2^z3^z4) * 2.3283064365387e-10;
    }

    thread float rand() {
        unsigned hashed_seed = this->seed * 1099087573UL;

        unsigned z1 = TausStep(hashed_seed,13,19,12,429496729UL);
        unsigned z2 = TausStep(hashed_seed,2,25,4,4294967288UL);
        unsigned z3 = TausStep(hashed_seed,3,11,17,429496280UL);
        unsigned z4 = (1664525*hashed_seed + 1013904223UL);

        thread float old_seed = this->seed;

        this->seed = (z1^z2^z3^z4) * 2.3283064365387e-10;

        return old_seed;
    }
};


*/
