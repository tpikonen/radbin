// Pure C implementation of radial rebinning
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

extern int binsearch(double val, const double *table, int len);

// FIXME: do checking for small bin areas so that only the necessary
// rectangular portion of the matrix goes through the inner loop.

// Do radial rebinning, return:
// 0 if success
// 1 if malloc fails
// 2 if total phirange is larger than 2*pi
#ifdef MAKE_TABLE
// Return two arrays of size (xdim, ydim) containing indices to radial
// and phi bins. The pixels which are outside of bins or masked are given
// index (0,0). For this reason, the radial indices are offset by one,
// i.e. a pixel with distance < 0.5 from the center is mapped to radial
// index 1.
int make_radtable(int xdim, int ydim, double x, double y, \
    const double *radrange_in, int radlen, \
    const double *phirange_in, int philen, const uint8_t *mask,\
    uint16_t *rad_out, uint16_t *phi_out)
#else
// Return img radially binned into array a_out of size (radlen, philen).
// Also return the number of pixels in each output bin in n_out.
int do_rebin(const double *img, int xdim, int ydim, double x, double y, \
    const double *radrange_in, int radlen, \
    const double *phirange_in, int philen, int norm, const uint8_t *mask,\
    double *a_out, int *n_out)
#endif
{
    int i, p, q;
    int rind, pind, o_ind, i_ind, rbins, phibins;
    double phi, phibias, r2, r, xrel, yrel;
    double *rad2range, *phirange;

    if(!(rad2range = (double *) malloc(radlen*sizeof(double))))
        return 1;
    if(!(phirange = (double *) malloc(philen*sizeof(double))))
        return 1;

    for(p = 0; p < radlen; p++) {
	rad2range[p] = radrange_in[p]*radrange_in[p];
    }
    // rotate phirange to start from 0 (to avoid negative angles)
    phibias = -1.0*phirange_in[0];
    for(i = 0; i < philen; i++) {
	phirange[i] = phirange_in[i] + phibias;
    }
    if(phirange[philen -1] > 2*M_PI) {
	printf("PHIRANGE should not span more than 360 degrees\n");
        return 2;
    }
    phibias = phibias - 2*M_PI*floor(phibias / (2*M_PI));

    rbins = radlen - 1;
    phibins = philen - 1;

#ifdef MAKE_TABLE

    // Initialize output tables to zero. Non-binned values go to bin
    // (rad, phi) = (0,0)
    for(i = 0; i < xdim*ydim; i++) {
	rad_out[i] = 0;
	phi_out[i] = 0;
    }
#else
    for(q = 0; q < phibins; q++) {
        o_ind = q*rbins;
	for(p = 0; p < rbins; p++) {
	    a_out[o_ind] = 0.0;
	    n_out[o_ind] = 0;
            o_ind++;
	}
    }
#endif

    rind = 0;
    pind = 0;
    for(p = 0; p < ydim; p++) {
        i_ind = p*xdim - 1;
        for(q = 0; q < xdim; q++) {
            i_ind++;
	    // discard non-roi area
	    if(mask[i_ind] != 1) {
		continue;
	    }
	    xrel = (q + 0.5) - x;
	    // y grows downwards in the matrix coordinates, opposite
	    // from the normal direction, thus the negation
	    yrel = y - (p + 0.5);

	    // rule out the cases where the pixel is not in the bin-area
	    r2 = (xrel*xrel + yrel*yrel);
	    if( (r2 >= rad2range[rbins]) || (r2 < rad2range[0]) ) {
		continue;
	    }
	    r = sqrt(r2);
	    if(yrel >= 0) {
		phi = acos(xrel/r);
	    }
	    else {
		phi = 2.0*M_PI - acos(xrel/r);
	    }
	    // modular addition of phibias
	    phi += phibias;
	    if(phi >= 2.0*M_PI) {
		phi -= 2.0*M_PI;
	    } else if(phi < 0) {
		phi += 2.0*M_PI;
	    }
	    if(phi >= phirange[phibins]) {
		continue;
	    }
	    // avoid binsearch on consecutive cells
	    if( r2 >= rad2range[rind+1] || r2 < rad2range[rind] ) {
		rind = binsearch(r2, rad2range, radlen);
	    }
	    if( phi >= phirange[pind+1] || phi < phirange[pind] ) {
		pind = binsearch(phi, phirange, philen);
	    }
	    // indices are okay now
#ifdef MAKE_TABLE
            rad_out[i_ind] = (uint16_t) rind+1; // Make room for masked pixels
            phi_out[i_ind] = (uint16_t) pind;
#else
            o_ind = pind*rbins + rind;
	    a_out[o_ind] += img[i_ind];
	    n_out[o_ind]++;
#endif
	}
    }

#ifndef MAKE_TABLE
    if(norm) {
	// normalize
	for(p = 0; p < rbins*phibins; p++) {
	    if(n_out[p] != 0) {
		a_out[p] = a_out[p] / ((double) n_out[p]);
	    }
	}
    }
#endif

    return 0;
}
