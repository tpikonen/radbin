// Pure C implementation of radial rebinning
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

extern int binsearch(double val, const double *table, int len);

// FIXME: do checking for small bin areas so that only the necessary
// rectangular portion of the matrix goes through the inner loop.

// Do radial rebinning, return:
// 0 if success
// 1 if malloc fails
// 2 if total phirange is larger than 2*pi
int do_rebin(const double *img, int xdim, int ydim, double x, double y, \
    const double *radrange_in, int radlen, \
    const double *phirange_in, int philen, int norm, const uchar *mask,\
    double *a_out, int *n_out)
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

    for(q = 0; q < phibins; q++) {
        o_ind = q*rbins;
	for(p = 0; p < rbins; p++) {
	    a_out[o_ind] = 0.0;
	    n_out[o_ind] = 0;
            o_ind++;
	}
    }

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
	    // indices should be okay now, just the summing is left
            o_ind = pind*rbins + rind;
	    a_out[o_ind] += img[i_ind];
	    n_out[o_ind]++;
	}
    }

    if(norm) {
	// normalize
	for(p = 0; p < rbins*phibins; p++) {
	    if(n_out[p] != 0) {
		a_out[p] = a_out[p] / ((double) n_out[p]);
	    }
	}
    }

    return 0;
}
