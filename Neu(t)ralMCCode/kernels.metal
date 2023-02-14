/*---------------------------------------------------------------------------------------------------------------------*/
//  This file is a part of Neu(t)ralMC: a unified, power-efficient platform platform for photon transport simulations
//  accelerated by Metal/GPU computing and Machine Learning
//  created by Alexander Doronin
//  Source code:    https://github.com/aledoronin
//  Web:            http://www.lighttransport.net/
//  Licence:        BSD-3-Clause, see LICENCE file
//  Contributors:   the respective contributors, as shown by the AUTHORS file
//  Year conceived: 2023
//  This file:      kernels.metal, contains Metal Banchmarks for Monte Carlo simulation of fluence rate F and escaping flux J,
//                  in a semi-infinite medium such as biological tissue, with an external_medium/tissue surface boundary from
//                  https://omlc.org/software/mc/mcsub/ and mcxyz https://omlc.org/software/mc/mcxyz/index.html
/*---------------------------------------------------------------------------------------------------------------------*/

#include <metal_stdlib>
#include "definitions.h"
#import "helpers.metal"
using namespace metal;

/* Propagation parameters */
typedef struct tagPhoton
{
public:
    thread tagPhoton()
    {
        x = MC_ZERO; y = MC_ZERO; z = MC_ZERO;
        ux = MC_ZERO; uy = MC_ZERO; uz = MC_ZERO;
        uxx = MC_ZERO; uyy = MC_ZERO; uzz = MC_ZERO;
        s = MC_ZERO; sleft = MC_ZERO; costheta = MC_ZERO;
        sintheta = MC_ZERO; cospsi = MC_ZERO; sinpsi = MC_ZERO;
        psi = MC_ZERO; num_scatt = 0; W = MC_ONE; absorb = MC_ZERO;
        photon_status = ALIVE; sv = false; tiss_type = -1;
    };
    float   x, y, z;        /* photon position */
    float   ux, uy, uz;     /* photon trajectory as cosines */
    float   uxx, uyy, uzz;  /* temporary values used during SPIN */
    float   s;              /* step sizes. s = -log(RND)/mus [cm] */
    float   sleft;          /* dimensionless */
    float   costheta;       /* cos(theta) */
    float   sintheta;       /* sin(theta) */
    float   cospsi;         /* cos(psi) */
    float   sinpsi;         /* sin(psi) */
    float   psi;            /* azimuthal angle */
    float   W;              /* photon weight */
    float   absorb;         /* weighted deposited in a step due to absorption */
    bool    photon_status;  /* flag = ALIVE=1 or DEAD=0 */
    bool    sv;             /* Are they in the same voxel? */
    int     num_scatt;      /* current number of scatttering even */
    int     tiss_type;      /* current tissue type */
} Photon;


kernel void MCSubKernel(
                       device float* J,
                       device float* F,
                       device float* SAE,
                       device int* iRunNum,
                       uint index [[thread_position_in_grid]])
{
    float3 pos = {0};
    float3 dir  = {0};

    /* Variable parameters */
    float    mut, albedo, absorb, rsp, Rsptot, Atot;
    float    rnd, xfocus, S = 0, A = 0, E = 0;
    float    uz1, uxx,uyy,uzz, s,r,W,temp;
    float    psi,costheta,sintheta,cospsi,sinpsi;
    long     ir, iz, CNT;
    short    photon_status;
    int numRuns = *iRunNum+1;
    int nums_scatt = 0;
    RandomGen rng_gen = RandomGen(numRuns*index, numRuns*1234, index*1234);
    
    CNT = 0;
    mut    = mua_mcsub + mus_mcsub;
    albedo = mus_mcsub/mut;
    Rsptot = 0.0; /* accumulate specular reflectance per photon */
    Atot   = 0.0; /* accumulate absorbed photon weight */
    rsp = 0.0;
    
    if (mcflag_mcsub == 0) {
        /* UNIFORM COLLIMATED BEAM INCIDENT AT z = zs */
        /* Launch at (r,zz) = (radius*sqrt(rnd), 0).
         * Due to cylindrical symmetry, radial launch position is
         * assigned to x while y = 0.
         * radius = radius of uniform beam. */
        /* Initial position */
        rnd = rng_gen.rand();
        pos.xyz = float3(radius_mcsub*sqrt(rnd), 0.0, zs_mcsub);
        /* Initial trajectory as cosines */
        dir.xyz = float3(0.0, 0.0, 1.0);
        /* specular reflectance */
        temp   = n1_mcsub/n2_mcsub; /* refractive index mismatch, internal/external */
        temp   = (1.0 - temp)/(1.0 + temp);
        rsp    = temp*temp; /* specular reflectance at boundary */
        }
    else if (mcflag_mcsub == 1) {
        /* GAUSSIAN BEAM AT SURFACE */
        /* Launch at (r,z) = (radius*sqrt(-log(rnd)), 0).
         * Due to cylindrical symmetry, radial launch position is
         * assigned to x while y = 0.
         * radius = 1/e radius of Gaussian beam at surface.
         * waist  = 1/e radius of Gaussian focus.
         * zfocus = depth of focal point. */
        /* Initial position */
        while ((rnd = rng_gen.rand()) <= 0.0); /* avoids rnd = 0 */
        pos.xyz = float3(radius_mcsub*sqrt(-log(rnd)), 0.0, 0.0);
        /* Initial trajectory as cosines */
        /* Due to cylindrical symmetry, radial launch trajectory is
         * assigned to ux and uz while uy = 0. */
        while ((rnd = rng_gen.rand()) <= 0.0); /* avoids rnd = 0 */
        xfocus = waist_mcsub*sqrt(-log(rnd));
        temp = sqrt((pos.x - xfocus)*(pos.x - xfocus) + zfocus_mcsub*zfocus_mcsub);
        sintheta = -(pos.x - xfocus)/temp;
        costheta = zfocus_mcsub/temp;
        dir.xyz = float3(sintheta, 0.0, costheta);
        /* specular reflectance and refraction */
        float uz = dir.z;
        rsp = Fresnel::RFresnel(n2_mcsub, n1_mcsub, costheta, &uz); /* new uz */
        dir.z = uz;
        dir.x  = sqrt(1.0 -  dir.z*dir.z); /* new ux */
        }
    else if  (mcflag_mcsub == 2) {
        /* ISOTROPIC POINT SOURCE AT POSITION xs,ys,zs */
        /* Initial position */
        pos.xyz = float3(xs_mcsub, ys_mcsub, zs_mcsub);
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
         if ( (boundaryflag_mcsub == 1) & (pos.z <= 0)) {
            rnd = rng_gen.rand();
            /* Check Fresnel reflectance at surface boundary */
             if (rnd > Fresnel::RFresnel(n1_mcsub, n2_mcsub, -dir.z, &uz1)) {
                /* Photon escapes at external angle, uz1 = cos(angle) */
                pos.x -= s*dir.x;       /* return to original position */
                pos.y -= s*dir.y;
                pos.z -= s*dir.z;
                s  = fabs(pos.z/dir.z); /* calculate stepsize to reach surface */
                pos.x += s*dir.x;       /* partial step to reach surface */
                pos.y += s*dir.y;
                r = sqrt(pos.x*pos.x + pos.y*pos.y);   /* find radial position r */
                ir = (long)(r/dr_mcsub) + 1; /* round to 1 <= ir */
                if (ir > BINS) ir = BINS;  /* ir = NR is overflow bin */
                long thread_offset_J = index*BINS;
                J[thread_offset_J + ir] += W;      /* increment escaping flux */
                E += W;
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
        ir = (long)(r/dr_mcsub) + 1;        /* round to 1 <= ir */
        iz = (long)(fabs(pos.z)/dz_mcsub) + 1;  /* round to 1 <= iz */
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
        if (FltEq(g_mcsub,0.0))
            costheta = 2.0*rnd - 1.0;
        else if (FltEq(g_mcsub,1.0))
            costheta = 1.0;
        else {
            temp = (1.0 - g_mcsub*g_mcsub)/(1.0 - g_mcsub + 2.0*g_mcsub*rnd);
            costheta = (1.0 + g_mcsub*g_mcsub - temp*temp)/(2.0*g_mcsub);
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
        
        nums_scatt++;

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
        
      //  if (nums_scatt > MAX_SCATT)
     //       photon_status = DEAD;

        }/**********************************************
          **** END of SPINCYCLE = DROP_SPIN_ROULETTE *
          **********************************************/

    }
    while (photon_status == ALIVE);
    
    long thread_offset_J = index*3;
    S = Rsptot;
    A = Atot;

    SAE[thread_offset_J + 0] += S;
    SAE[thread_offset_J + 1] += A;
    SAE[thread_offset_J + 2] += E;
    
}


kernel void MCXYZKernel(
                       device RunParams* run_params,
                       device TissueParams* tissParams,
                       device char* V,
                       device float* F3D,
                       device int* iRunNum,
                       uint index [[thread_position_in_grid]])
{

    int numRuns = *iRunNum+1;
    Photon photon;
    RunParams runParamsG = *run_params;
    TissueParams tissParamsG = *tissParams;
    RandomGen rng_gen = RandomGen(numRuns*index, numRuns*1234, index*1234);

    /**** LAUNCH
    Initialize photon position and trajectory.
    *****/

    photon.num_scatt = 0;
    photon.W = 1.0;                    /* set photon weight to one */
    photon.photon_status = ALIVE;      /* Launch an ALIVE photon */
    photon.tiss_type = -1;
    float rnd = 0;
    
    /**** SET SOURCE
     * Launch collimated beam at x,y center.
     ****/
    
    /****************************/
    /* Initial position. */
    
    /* trajectory */
    if (runParamsG.launchflag==1) { // manually set launch
        photon.x    = runParamsG.xs;
        photon.y    = runParamsG.ys;
        photon.z    = runParamsG.zs;
        photon.ux   = runParamsG.ux0;
        photon.uy   = runParamsG.uy0;
        photon.uz   = runParamsG.uz0;
    }
    else { // use mcflag
        if (runParamsG.mcflag==0) { // uniform beam
            // set launch point and width of beam
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            float r        = runParamsG.radius*sqrt(rnd); // radius of beam at launch point
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            float phi       = rnd*2.0*PI;
            photon.x        = runParamsG.xs + r*cos(phi);
            photon.y        = runParamsG.ys + r*sin(phi);
            photon.z        = runParamsG.zs;
            // set trajectory toward focus
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            r        = runParamsG.waist*sqrt(rnd); // radius of beam at focus
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            phi        = rnd*2.0*PI;
            float xfocus    = r*cos(phi);
            float yfocus    = r*sin(phi);
            float temp    = sqrt((photon.x - xfocus)*(photon.x - xfocus) + (photon.y - yfocus)*(photon.y - yfocus) + runParamsG.zfocus*runParamsG.zfocus);
            photon.ux        = -(photon.x - xfocus)/temp;
            photon.uy        = -(photon.y - yfocus)/temp;
            photon.uz        = sqrt(1.0 - photon.ux*photon.ux - photon.uy*photon.uy);
        }
        else if (runParamsG.mcflag==2) { // isotropic pt source
            photon.costheta = 1.0 - 2.0*RandomNum;
            photon.sintheta = sqrt(1.0 - photon.costheta*photon.costheta);
            float psi = 2.0*PI*RandomNum;
            photon.cospsi = cos(psi);
            if (psi < PI)
                photon.sinpsi = sqrt(1.0 - photon.cospsi*photon.cospsi);
            else
                photon.sinpsi = -sqrt(1.0 - photon.cospsi*photon.cospsi);
            photon.x = runParamsG.xs;
            photon.y = runParamsG.ys;
            photon.z = runParamsG.zs;
            photon.ux = photon.sintheta*photon.cospsi;
            photon.uy = photon.sintheta*photon.sinpsi;
            photon.uz = photon.costheta;
        }
        else if (runParamsG.mcflag==3) { // rectangular source collimated
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            photon.x = runParamsG.radius*(rnd*2.0-1.0); // use radius to specify x-halfwidth of rectangle
            while ((rnd = RandomNum) <= 0.0); // avoids rnd = 0
            photon.y = runParamsG.radius*(rnd*2.0-1.0); // use radius to specify y-halfwidth of rectangle
            photon.z = runParamsG.zs;
            photon.ux = 0.0;
            photon.uy = 0.0;
            photon.uz = 1.0; // collimated beam
        }
    } // end  use mcflag
    /****************************/

    /* Get tissue voxel properties of launchpoint.
        * If photon beyond outer edge of defined voxels,
        * the tissue equals properties of outermost voxels.
        * Therefore, set outermost voxels to infinite background value.
        */
    /* Added. Used to track photons */
    int ix = (int)(runParamsG.Nx / 2 + photon.x / runParamsG.dx);
    int iy = (int)(runParamsG.Ny / 2 + photon.y / runParamsG.dy);
    int iz = (int)(photon.z / runParamsG.dz);
    if (ix >= runParamsG.Nx) ix = runParamsG.Nx - 1;
    if (iy >= runParamsG.Ny) iy = runParamsG.Ny - 1;
    if (iz >= runParamsG.Nz) iz = runParamsG.Nz - 1;
    if (ix<0)   ix = 0;
    if (iy<0)   iy = 0;
    if (iz<0)   iz = 0;
    /* Get the tissue type of located voxel */
    long i = (long)(iz*runParamsG.Ny*runParamsG.Nx + ix*runParamsG.Ny + iy);
    photon.tiss_type = V[i];
    float mua = tissParamsG.muav[photon.tiss_type];
    float mus = tissParamsG.musv[photon.tiss_type];
    float g = tissParamsG.gv[photon.tiss_type];
    int bflag = 1; // initialize as 1 = inside volume, but later check as photon propagates.
    
    /* HOP_DROP_SPIN_CHECK
     Propagate one photon until it dies as determined by ROULETTE.
     *******/
    
    do {
        
      /**** HOP
         Take step to new position
         s = dimensionless stepsize
         x, uy, uz are cosines of current photon trajectory
         *****/
        
      while ((rnd = RandomNum) <= 0.0);   /* yields 0 < rnd <= 1 */
      photon.sleft = -log(rnd);                /* dimensionless step */
      int curr_depth = 0;
            
      do {  // while sleft>0 or maximum tracing depth achived
            photon.s = photon.sleft / mus;                /* Step size [cm].*/
            float    tempx, tempy, tempz; /* temporary variables, used during photon step. */
            tempx = photon.x + photon.s*photon.ux;                /* Update positions. [cm] */
            tempy = photon.y + photon.s*photon.uy;
            tempz = photon.z + photon.s*photon.uz;
            
            photon.sv = SameVoxel(photon.x, photon.y, photon.z, tempx, tempy, tempz, runParamsG.dx, runParamsG.dy, runParamsG.dz);
            
            if (photon.sv) /* photon in same voxel */
            {
                photon.x = tempx;                    /* Update positions. */
                photon.y = tempy;
                photon.z = tempz;

                /**** DROP
                Drop photon weight (W) into local bin.
                *****/
                photon.absorb = photon.W*(1.0 - exp(-mua*photon.s));    /* photon weight absorbed at this step */
                photon.W -= photon.absorb;                    /* decrement WEIGHT by amount absorbed */
                // If photon within volume of heterogeneity, deposit energy in F[].
                // Normalize F[] later, when save output.
                if (bflag)
                    F3D[i] += photon.absorb;  // only save data if blag==1, i.e., photon inside simulation cube
                /* Update sleft */
                photon.sleft = 0.0;        /* dimensionless step remaining */
            }
            else /* photon has crossed voxel boundary */
            {
                
                /* step to voxel face + "littlest step" so just inside new voxel. */
                photon.s = ls + FindVoxelFace2(photon.x, photon.y, photon.z, tempx, tempy, tempz, runParamsG.dx, runParamsG.dy, runParamsG.dz, photon.ux, photon.uy, photon.uz);

                /**** DROP
                Drop photon weight (W) into local bin.
                *****/
                photon.absorb = photon.W*(1.0 - exp(-mua*photon.s));   /* photon weight absorbed at this step */
                photon.W -= photon.absorb;                  /* decrement WEIGHT by amount absorbed */
                // If photon within volume of heterogeneity, deposit energy in F[].
                // Normalize F[] later, when save output.
                if (bflag)
                    F3D[i] += photon.absorb;  // only save data if blag==1, i.e., photon inside simulation cube

                /* Update sleft */
                photon.sleft -= photon.s*mus;  /* dimensionless step remaining */
                if (photon.sleft <= ls) photon.sleft = 0.0;

                /* Update positions. */
                photon.x += photon.s*photon.ux;
                photon.y += photon.s*photon.uy;
                photon.z += photon.s*photon.uz;

                // pointers to voxel containing optical properties
                ix = (int)(runParamsG.Nx / 2 + photon.x / runParamsG.dx);
                iy = (int)(runParamsG.Ny / 2 + photon.y / runParamsG.dy);
                iz = (int)(photon.z / runParamsG.dz);

                bflag = 1;  // Boundary flag. Initialize as 1 = inside volume, then check.
                if (runParamsG.boundaryflag == 0) { // Infinite medium.
                            // Check if photon has wandered outside volume.
                            // If so, set tissue type to boundary value, but let photon wander.
                            // Set blag to zero, so DROP does not deposit energy.
                    if (iz>=runParamsG.Nz) {iz=runParamsG.Nz-1; bflag = 0;}
                    if (ix>=runParamsG.Nx) {ix=runParamsG.Nx-1; bflag = 0;}
                    if (iy>=runParamsG.Ny) {iy=runParamsG.Ny-1; bflag = 0;}
                    if (iz<0)   {iz=0;    bflag = 0;}
                    if (ix<0)   {ix=0;    bflag = 0;}
                    if (iy<0)   {iy=0;    bflag = 0;}
                }
                else if (runParamsG.boundaryflag==1) { // Escape at boundaries
                    if (iz>=runParamsG.Nz) {iz=runParamsG.Nz-1; photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (ix>=runParamsG.Nx) {ix=runParamsG.Nx-1; photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (iy>=runParamsG.Ny) {iy=runParamsG.Ny-1; photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (iz<0)   {iz=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (ix<0)   {ix=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (iy<0)   {iy=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                }
                else if (runParamsG.boundaryflag==2) { // Escape at top surface, no x,y bottom z boundaries
                    if (iz>=runParamsG.Nz) {iz=runParamsG.Nz-1; bflag = 0.0;}
                    if (ix>=runParamsG.Nx) {ix=runParamsG.Nx-1; bflag = 0.0;}
                    if (iy>=runParamsG.Ny) {iy=runParamsG.Ny-1; bflag = 0.0;}
                    if (iz<0)   {iz=0;    photon.photon_status = DEAD; photon.sleft=0.0;}
                    if (ix<0)   {ix=0;    bflag = 0;}
                    if (iy<0)   {iy=0;    bflag = 0;}
                }

                // update pointer to tissue type
                i = (long)(iz*runParamsG.Ny*runParamsG.Nx + ix*runParamsG.Ny + iy);
                photon.tiss_type = V[i];
                mua = tissParamsG.muav[photon.tiss_type];
                mus = tissParamsG.musv[photon.tiss_type];
                g = tissParamsG.gv[photon.tiss_type];


            } //(sv) /* same voxel */
          curr_depth++;
          photon.num_scatt++;
       } while (curr_depth <= RAY_DEPTH && photon.sleft>0.0); //do...while

        
        /**** SPIN
        Scatter photon into new trajectory defined by theta and psi.
        Theta is specified by cos(theta), which is determined
        based on the Henyey-Greenstein scattering function.
        Convert theta and psi into cosines ux, uy, uz.
        *****/
        /* Sample for costheta */
        rnd = RandomNum;
        if (FltEq(g, 0.0))
            photon.costheta = 2.0*rnd - 1.0;
        else if (FltEq(g, 1.0))
            photon.costheta = 1.0;
        else {
            float temp = (1.0 - g*g) / (1.0 - g + 2.0 * g*rnd);
            photon.costheta = (1.0 + g*g - temp*temp) / (2.0*g);
        }
        photon.sintheta = sqrt(1.0 - photon.costheta*photon.costheta); /* sqrt() is faster than sin(). */

        /* Sample psi. */
        photon.psi = 2.0*PI*RandomNum;
        photon.cospsi = cos(photon.psi);
        if (photon.psi < PI)
            photon.sinpsi = sqrt(1.0 - photon.cospsi*photon.cospsi);     /* sqrt() is faster than sin(). */
        else
            photon.sinpsi = -sqrt(1.0 - photon.cospsi*photon.cospsi);

        /* New trajectory. */
        if (1.0 - fabs(photon.uz) <= ONE_MINUS_COSZERO) {      /* close to perpendicular. */
            photon.uxx = photon.sintheta * photon.cospsi;
            photon.uyy = photon.sintheta * photon.sinpsi;
            photon.uzz = photon.costheta * SIGN(photon.uz);   /* SIGN() is faster than division. */
        }
        else {              /* usually use this option */
            float temp = sqrt(1.0 - photon.uz * photon.uz);
            photon.uxx = photon.sintheta * (photon.ux * photon.uz * photon.cospsi - photon.uy * photon.sinpsi) / temp + photon.ux * photon.costheta;
            photon.uyy = photon.sintheta * (photon.uy * photon.uz * photon.cospsi + photon.ux * photon.sinpsi) / temp + photon.uy * photon.costheta;
            photon.uzz = -photon.sintheta * photon.cospsi * temp + photon.uz * photon.costheta;
        }

        /* Update trajectory */
        photon.ux = photon.uxx;
        photon.uy = photon.uyy;
        photon.uz = photon.uzz;
        photon.num_scatt++;
            
        /**** CHECK ROULETTE
        If photon weight below THRESHOLD, then terminate photon using Roulette technique.
        Photon has CHANCE probability of having its weight increased by factor of 1/CHANCE,
        and 1-CHANCE probability of terminating.
        *****/
        if (photon.W < THRESHOLD) {
            if (RandomNum <= CHANCE)
                photon.W /= CHANCE;
            else photon.photon_status = DEAD;
        }
            
        // Russian roulette with a semi-automatic
        if (photon.W < MIN_VALUE || photon.num_scatt > MAX_SCATT)
            photon.photon_status = DEAD;

        } while (photon.photon_status == ALIVE);  /* end STEP_CHECK_HOP_SPIN */
        /* if ALIVE, continue propagating */
        /* If photon DEAD, then launch new photon. */
}



