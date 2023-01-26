//
//  helper_func.metal
//  Neu(t)ralMC
//
//  Created by alexd on 19/01/23.
//  Copyright © 2023 Apple. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;


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
        
        float r;
        
        if(n1_par==n2_par) { /** matched boundary. **/
            *ca2_Ptr = ca1;
            r = 0.0;
        }
        else if(ca1>(1.0 - 1.0e-12)) { /** normal incidence. **/
            *ca2_Ptr = ca1;
            r = (n2_par-n1_par)/(n2_par+n1_par);
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
            sa2 = n1_par*sa1/n2_par;
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
    
};

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
