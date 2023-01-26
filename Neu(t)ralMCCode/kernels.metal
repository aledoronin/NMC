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

#import "helpers.metal"

class Photon
{
public:
    thread Photon() {
        m_pos = {0};
        m_dir = {0};
    }
    
    thread Photon(float3 pos, float3 dir) {
        m_pos = pos;
        m_dir = dir;
    }
private:
    float3 m_pos;
    float3 m_dir;
};

kernel void MonteCarloKernel(
                       device float* J,
                       device float* F,
                       device float* SAE,
                       device int* iRunNum,
                       uint index [[thread_position_in_grid]])
{

    
    Photon photon;
    
    float3 pos = {0};
    float3 dir  = {0};

    /* Variable parameters */
    float    mut, albedo, absorb, rsp, Rsptot, Atot;
    float    rnd, xfocus, S, A, E;
    float    uz1, uxx,uyy,uzz, s,r,W,temp;
    float    psi,costheta,sintheta,cospsi,sinpsi;
    long     ir, iz, CNT;
    short    photon_status;
    int numRuns = *iRunNum+1;
    RandomGen rng_gen = RandomGen(numRuns*index, numRuns*1234, index*1234);
    
    CNT = 0;
    mut    = mua + mus;
    albedo = mus/mut;
    Rsptot = 0.0; /* accumulate specular reflectance per photon */
    Atot   = 0.0; /* accumulate absorbed photon weight */
    
    if (mcflag == 0) {
        /* UNIFORM COLLIMATED BEAM INCIDENT AT z = zs */
        /* Launch at (r,zz) = (radius*sqrt(rnd), 0).
         * Due to cylindrical symmetry, radial launch position is
         * assigned to x while y = 0.
         * radius = radius of uniform beam. */
        /* Initial position */
        rnd = rng_gen.rand();
        pos.xyz = float3(radius*sqrt(rnd), 0.0, zs);
        /* Initial trajectory as cosines */
        dir.xyz = float3(0.0, 0.0, 1.0);
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
        pos.xyz = float3(radius*sqrt(-log(rnd)), 0.0, 0.0);
        /* Initial trajectory as cosines */
        /* Due to cylindrical symmetry, radial launch trajectory is
         * assigned to ux and uz while uy = 0. */
        while ((rnd = rng_gen.rand()) <= 0.0); /* avoids rnd = 0 */
        xfocus = waist*sqrt(-log(rnd));
        temp = sqrt((pos.x - xfocus)*(pos.x - xfocus) + zfocus*zfocus);
        sintheta = -(pos.x - xfocus)/temp;
        costheta = zfocus/temp;
        dir.xyz = float3(sintheta, 0.0, costheta);
        /* specular reflectance and refraction */
        float uz = dir.z;
        rsp = Fresnel::RFresnel(n2, n1, costheta, &uz); /* new uz */
        dir.z = uz;
        dir.x  = sqrt(1.0 -  dir.z*dir.z); /* new ux */
        }
    else if  (mcflag == 2) {
        /* ISOTROPIC POINT SOURCE AT POSITION xs,ys,zs */
        /* Initial position */
        pos.xyz = float3(xs, ys, zs);
        /* Initial trajectory as cosines */
        costheta = 1.0 - 2.0*rng_gen.rand();
        sintheta = sqrt(1.0 - costheta*costheta);
        psi = 2.0*PI*rng_gen.rand();
        cospsi = cos(psi);
        if (psi < PI)
            sinpsi = sqrt(1.0 - cospsi*cospsi);
        else
            sinpsi = -sqrt(1.0 - cospsi*cospsi);
        dir.xyz = float3(sintheta*cospsi, sintheta*sinpsi, costheta);
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
        pos.x += s*dir.x;           /* Update positions. */
        pos.y += s*dir.y;
        pos.z += s*dir.z;

        /* Does photon ESCAPE at surface? ... z <= 0? */
         if ( (boundaryflag == 1) & (pos.z <= 0)) {
            rnd = rng_gen.rand();
            /* Check Fresnel reflectance at surface boundary */
             if (rnd > Fresnel::RFresnel(n1, n2, -dir.z, &uz1)) {
                /* Photon escapes at external angle, uz1 = cos(angle) */
                pos.x -= s*dir.x;       /* return to original position */
                pos.y -= s*dir.y;
                pos.z -= s*dir.z;
                s  = fabs(pos.z/dir.z); /* calculate stepsize to reach surface */
                pos.x += s*dir.x;       /* partial step to reach surface */
                pos.y += s*dir.y;
                r = sqrt(pos.x*pos.x + pos.y*pos.y);   /* find radial position r */
                ir = (long)(r/dr) + 1; /* round to 1 <= ir */
                if (ir > BINS) ir = BINS;  /* ir = NR is overflow bin */
                long thread_offset_J = index*BINS;
                J[ir + thread_offset_J] += W;      /* increment escaping flux */
                photon_status = DEAD;
                }
            else {
                pos.z = -pos.z;   /* Total internal reflection. */
                dir.z = -dir.z;
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
        r  = sqrt(pos.x*pos.x + pos.y*pos.y);         /* current cylindrical radial position */
        ir = (long)(r/dr) + 1;        /* round to 1 <= ir */
        iz = (long)(fabs(pos.z)/dz) + 1;  /* round to 1 <= iz */
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
        if (1 - fabs(dir.z) <= 1.0e-12) {  /* close to perpendicular. */
            uxx = sintheta*cospsi;
            uyy = sintheta*sinpsi;
            uzz = costheta*((dir.z)>=0 ? 1:-1);
            }
        else {   /* usually use this option */
            temp = sqrt(1.0 - dir.z*dir.z);
            uxx = sintheta*(dir.x*dir.z*cospsi - dir.y*sinpsi)/temp + dir.x*costheta;
            uyy = sintheta*(dir.y*dir.z*cospsi + dir.x*sinpsi)/temp + dir.y*costheta;
            uzz = -sintheta*cospsi*temp +  dir.z*costheta;
            }

        /* Update trajectory */
        dir.x = uxx;
        dir.y = uyy;
        dir.z = uzz;

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



