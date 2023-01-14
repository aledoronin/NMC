/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Metal Banchmark for
 
 * Monte Carlo simulation of fluence rate F and escaping flux J
 * in a semi-infinite medium such as biological tissue,
 * with an external_medium/tissue surface boundary.
 
 https://omlc.org/software/mc/mcsub/
 
*/

#include <metal_stdlib>
#include "definitions.h"
using namespace metal;

#import "loki_header.metal"


/// This is a Metal Shading Language (MSL) function equivalent to the add_arrays() C function, used to perform the calculation on a GPU.

/************************************************************
 *    FRESNEL REFLECTANCE
 * Computes reflectance as photon passes from medium 1 to
 * medium 2 with refractive indices n1,n2. Incident
 * angle a1 is specified by cosine value ca1 = cos(a1).
 * Program returns value of transmitted angle a1 as
 * value in *ca2_Ptr = cos(a2).
 ****/
float RFresnel(float n1_par,        /* incident refractive index.*/
               float n2_par,        /* transmit refractive index.*/
               float ca1,        /* cosine of the incident */
                                    /* angle a1, 0<a1<90 degrees. */
               thread float *ca2_Ptr)     /* pointer to the cosine */
                                    /* of the transmission */
                                    /* angle a2, a2>0. */
{
    
    float r;

if(n1==n2) { /** matched boundary. **/
    *ca2_Ptr = ca1;
    r = 0.0;
    }
else if(ca1>(1.0 - 1.0e-12)) { /** normal incidence. **/
    *ca2_Ptr = ca1;
    r = (n2-n1)/(n2+n1);
    r *= r;
    }
else if(ca1< 1.0e-6)  {    /** very slanted. **/
    *ca2_Ptr = 0.0;
    r = 1.0;
    }
else  {                      /** general. **/
    float sa1, sa2; /* sine of incident and transmission angles. */
    float ca2;      /* cosine of transmission angle. */
    sa1 = sqrt(1-ca1*ca1);
    sa2 = n1*sa1/n2;
    if(sa2>=1.0) {
        /* double check for total internal reflection. */
        *ca2_Ptr = 0.0;
        r = 1.0;
        }
    else {
        float cap, cam;    /* cosines of sum ap or diff am of the two */
                            /* angles: ap = a1 + a2, am = a1 - a2. */
        float sap, sam;    /* sines. */
        *ca2_Ptr = ca2 = sqrt(1-sa2*sa2);
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

kernel void MonteCarloKernel(
                       device float* J,
                       device float* F,
                       device float* SAE,
                       device int* iRunNum,
                       uint index [[thread_position_in_grid]])
{

    
    /* Variable parameters */
    float    mut, albedo, absorb, rsp, Rsptot, Atot;
    float    rnd, xfocus, S, A, E;
    float    x,y,z, ux,uy,uz,uz1, uxx,uyy,uzz, s,r,W,temp;
    float    psi,costheta,sintheta,cospsi,sinpsi;
    long     ir, iz, CNT;
    short    photon_status;
    int numRuns = *iRunNum+1;
    
    CNT = 0;
    mut    = mua + mus;
    albedo = mus/mut;
    Rsptot = 0.0; /* accumulate specular reflectance per photon */
    Atot   = 0.0; /* accumulate absorbed photon weight */
    
    
    Loki rng_gen = Loki(numRuns*index, numRuns*1234, index*1234);
    
    if (mcflag == 0) {
        /* UNIFORM COLLIMATED BEAM INCIDENT AT z = zs */
        /* Launch at (r,zz) = (radius*sqrt(rnd), 0).
         * Due to cylindrical symmetry, radial launch position is
         * assigned to x while y = 0.
         * radius = radius of uniform beam. */
        /* Initial position */
        rnd = rng_gen.rand();
        x = radius*sqrt(rnd);
        y = 0;
        z = zs;
        /* Initial trajectory as cosines */
        ux = 0;
        uy = 0;
        uz = 1;
        /* specular reflectance */
        temp   = n1/n2; /* refractive index mismatch, internal/external */
        temp   = (1.0 - temp)/(1.0 + temp);
        rsp    = temp*temp; /* specular reflectance at boundary */
        }
    else if (mcflag == 1) {
        /* GAUSSIAN BEAM AT SURFACE */
        /* Launch at (r,z) = (radius*sqrt(-log(rnd)), 0).
         * Due to cylindrical symmetry, radial launch position is
         * assigned to x while y = 0.
         * radius = 1/e radius of Gaussian beam at surface.
         * waist  = 1/e radius of Gaussian focus.
         * zfocus = depth of focal point. */
        /* Initial position */
        while ((rnd = rng_gen.rand()) <= 0.0); /* avoids rnd = 0 */
        x = radius*sqrt(-log(rnd));
        y = 0.0;
        z = 0.0;
        /* Initial trajectory as cosines */
        /* Due to cylindrical symmetry, radial launch trajectory is
         * assigned to ux and uz while uy = 0. */
        while ((rnd = rng_gen.rand()) <= 0.0); /* avoids rnd = 0 */
        xfocus = waist*sqrt(-log(rnd));
        temp = sqrt((x - xfocus)*(x - xfocus) + zfocus*zfocus);
        sintheta = -(x - xfocus)/temp;
        costheta = zfocus/temp;
        ux = sintheta;
        uy = 0.0;
        uz = costheta;
        /* specular reflectance and refraction */
        rsp = RFresnel(n2, n1, costheta, &uz); /* new uz */
        ux  = sqrt(1.0 - uz*uz); /* new ux */
        }
    else if  (mcflag == 2) {
        /* ISOTROPIC POINT SOURCE AT POSITION xs,ys,zs */
        /* Initial position */
        x = xs;
        y = ys;
        z = zs;
        /* Initial trajectory as cosines */
        costheta = 1.0 - 2.0*rng_gen.rand();
        sintheta = sqrt(1.0 - costheta*costheta);
        psi = 2.0*PI*rng_gen.rand();
        cospsi = cos(psi);
        if (psi < PI)
            sinpsi = sqrt(1.0 - cospsi*cospsi);
        else
            sinpsi = -sqrt(1.0 - cospsi*cospsi);
        ux = sintheta*cospsi;
        uy = sintheta*sinpsi;
        uz = costheta;
        /* specular reflectance */
        rsp = 0.0;
        }
    
    W             = 1.0 - rsp;  /* set photon initial weight */
    Rsptot       += rsp; /* accumulate specular reflectance per photon */
    photon_status = ALIVE;
    
    /*
    ******************************************
    ****** HOP_ESCAPE_SPINCYCLE **************
    * Propagate one photon until it dies by ESCAPE or ROULETTE.
    *******************************************/
    
    do {

    /**** HOP
     * Take step to new position
     * s = stepsize
     * ux, uy, uz are cosines of current photon trajectory
     *****/
        while ((rnd = rng_gen.rand()) <= 0.0);   /* avoids rnd = 0 */
        s = -log(rnd)/mut;   /* Step size.  Note: log() is base e */
        x += s*ux;           /* Update positions. */
        y += s*uy;
        z += s*uz;

        /* Does photon ESCAPE at surface? ... z <= 0? */
         if ( (boundaryflag == 1) & (z <= 0)) {
            rnd = rng_gen.rand();
            /* Check Fresnel reflectance at surface boundary */
            if (rnd > RFresnel(n1, n2, -uz, &uz1)) {
                /* Photon escapes at external angle, uz1 = cos(angle) */
                x -= s*ux;       /* return to original position */
                y -= s*uy;
                z -= s*uz;
                s  = fabs(z/uz); /* calculate stepsize to reach surface */
                x += s*ux;       /* partial step to reach surface */
                y += s*uy;
                r = sqrt(x*x + y*y);   /* find radial position r */
                ir = (long)(r/dr) + 1; /* round to 1 <= ir */
                if (ir > BINS) ir = BINS;  /* ir = NR is overflow bin */
                long thread_offset_J = index*BINS;
                J[ir + thread_offset_J] += W;      /* increment escaping flux */
                photon_status = DEAD;
                }
            else {
                z = -z;   /* Total internal reflection. */
                uz = -uz;
                }
            }

    if (photon_status  == ALIVE) {
        /*********************************************
         ****** SPINCYCLE = DROP_SPIN_ROULETTE ******
         *********************************************/

        /**** DROP
         * Drop photon weight (W) into local bin.
         *****/
        absorb = W*(1 - albedo);      /* photon weight absorbed at this step */
        W -= absorb;                  /* decrement WEIGHT by amount absorbed */
        Atot += absorb;               /* accumulate absorbed photon weight */
        /* deposit power in cylindrical coordinates z,r */
        r  = sqrt(x*x + y*y);         /* current cylindrical radial position */
        ir = (long)(r/dr) + 1;        /* round to 1 <= ir */
        iz = (long)(fabs(z)/dz) + 1;  /* round to 1 <= iz */
        if (ir >= BINS) ir = BINS;        /* last bin is for overflow */
        if (iz >= BINS) iz = BINS;        /* last bin is for overflow */
        long thread_offset = index*BINS*BINS;
        F[thread_offset + ir*BINS + iz] += absorb;          /* DROP absorbed weight into bin */

        
        /**** SPIN
         * Scatter photon into new trajectory defined by theta and psi.
         * Theta is specified by cos(theta), which is determined
         * based on the Henyey-Greenstein scattering function.
         * Convert theta and psi into cosines ux, uy, uz.
         *****/
        /* Sample for costheta */
        rnd = rng_gen.rand();
        if (g == 0.0)
            costheta = 2.0*rnd - 1.0;
        else if (g == 1.0)
            costheta = 1.0;
        else {
            temp = (1.0 - g*g)/(1.0 - g + 2*g*rnd);
            costheta = (1.0 + g*g - temp*temp)/(2.0*g);
            }
        sintheta = sqrt(1.0 - costheta*costheta);/*sqrt faster than sin()*/

        /* Sample psi. */
        psi = 2.0*PI*rng_gen.rand();
        cospsi = cos(psi);
        if (psi < PI)
            sinpsi = sqrt(1.0 - cospsi*cospsi); /*sqrt faster */
        else
            sinpsi = -sqrt(1.0 - cospsi*cospsi);

        /* New trajectory. */
        if (1 - fabs(uz) <= 1.0e-12) {  /* close to perpendicular. */
            uxx = sintheta*cospsi;
            uyy = sintheta*sinpsi;
            uzz = costheta*((uz)>=0 ? 1:-1);
            }
        else {   /* usually use this option */
            temp = sqrt(1.0 - uz*uz);
            uxx = sintheta*(ux*uz*cospsi - uy*sinpsi)/temp + ux*costheta;
            uyy = sintheta*(uy*uz*cospsi + ux*sinpsi)/temp + uy*costheta;
            uzz = -sintheta*cospsi*temp + uz*costheta;
            }

        /* Update trajectory */
        ux = uxx;
        uy = uyy;
        uz = uzz;

        /**** CHECK ROULETTE
         * If photon weight below THRESHOLD, then terminate photon using
         * Roulette technique. Photon has CHANCE probability of having
         * its weight increased by factor of 1/CHANCE,
         * and 1-CHANCE probability of terminating.
         *****/
        if (W < THRESHOLD) {
            rnd = rng_gen.rand();
            if (rnd <= CHANCE)
                W /= CHANCE;
            else photon_status = DEAD;
            }

        }/**********************************************
          **** END of SPINCYCLE = DROP_SPIN_ROULETTE *
          **********************************************/

    }
    while (photon_status == ALIVE);
    
    temp = 0.0;
    for (ir=1; ir<=BINS; ir++) {
        r = (ir - 0.5)*dr;
        long thread_offset_J = index*BINS;
        temp += J[thread_offset_J + ir];    /* accumulate total escaped photon weight */
    }
    
    long thread_offset_J = index*3;
    SAE[thread_offset_J + 0] = S = Rsptot;
    SAE[thread_offset_J + 1] = A = Atot;
    SAE[thread_offset_J + 2] = E = temp;
    
}



