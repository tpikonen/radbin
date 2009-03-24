#include "Python.h"
#include "arrayobject.h"
#include <stdbool.h>

typedef unsigned char uchar;

const char docstring[] = "\
radbin(img, c_x, c_y, blah)\n\
";


/* Forward declaration of methods for the method table */
static PyObject *radbin(PyObject *self, PyObject *args, PyObject *kwds);

/* Table of methods in the extension */
static PyMethodDef cradbinMethods[] = {
    {"radbin", (PyCFunction) radbin, METH_VARARGS|METH_KEYWORDS, docstring},
    {NULL, NULL, 0, NULL}     /* Sentinel - marks the end of this structure */
};

/* Initialization of the extension */
void initcradbin()  {
    /* name string must match the name of the .so file */
    Py_InitModule("cradbin", cradbinMethods);
    /* Must be called after Py_InitModule when using Numpy Arrays */
    import_array();
}

// binary search of index of val in monotonically increasing array table
// returns the index of the table or
// -1 if val < table[0]
// -2 if val > table[len-1]
int binsearch(double val, const double *table, int len)
{
    int left = 0;
    int right = len;
    int mid;
    while((right - left) > 1) {
	mid = (left + right) >> 1;
	if(val >= table[mid]) {
	    left = mid;
	} else {
	    right = mid;
	}
    }
        
    return left;
}

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


static PyObject *radbin(PyObject *self, PyObject *args, PyObject *kwds)
//static PyObject *radbin(PyObject *self, PyObject *args)
{
    // Arguments
    PyArrayObject *img = NULL;
    double c_x = -1.0;
    double c_y = -1.0;
    PyArrayObject *radrange = NULL;
    PyArrayObject *phirange = NULL;
    int norm = 1;
    PyArrayObject *mask = NULL;
    // Outputs
    PyArrayObject *bin = NULL;
    PyArrayObject *num = NULL;
    // PyObject parsing
    char *keywords[] = { "image", "x", "y", "radrange", "phirange", 
        "normalize", "mask", NULL };
    PyObject *img_obj = NULL;
    PyObject *radrange_obj = NULL;
    PyObject *phirange_obj = NULL;
    PyObject *mask_obj = NULL;
    int width, height;
    int radlen, philen;
    int *maskp = NULL;
    int retval;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Odd|OOiO", keywords,
        &img_obj, &c_x, &c_y, &radrange_obj, &phirange_obj, &norm, &mask_obj))  
        return NULL;

    // Input array conversions
    img = (PyArrayObject *) PyArray_FROM_OTF(img_obj, NPY_DOUBLE, 
        NPY_IN_ARRAY);
    if(!img || img->nd != 2) {
        PyErr_SetString(PyExc_TypeError, "First argument must be a 2D array");
        goto fail;
    }
    width = img->dimensions[0];
    height = img->dimensions[1];

    if(radrange_obj) {
        radrange = (PyArrayObject *) PyArray_FROM_OTF(radrange_obj, 
            NPY_DOUBLE, NPY_IN_ARRAY);
    } else {
        npy_intp *dims = &radlen;
        int i;

        radlen = (int) (sqrt((double)(width*width + height*height))/2.0 + 1.0);
        radrange = (PyArrayObject *) PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
        for(i = 0; i < radlen; i++) {
            ((double *)radrange->data)[i] = (double) i;
        }
    }
    if(!radrange || radrange->nd != 1) { 
        PyErr_SetString(PyExc_TypeError, "radrange must be a 1D array");
        goto fail;
    }
    radlen = radrange->dimensions[0];

    if(phirange_obj) {
        phirange = (PyArrayObject *) PyArray_FROM_OTF(phirange_obj, 
            NPY_DOUBLE, NPY_IN_ARRAY);
    } else {
        npy_intp *dims = &philen;
        int i;

        philen = 2;
        phirange = (PyArrayObject *) PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
        for(i = 0; i < philen; i++) {
            ((double *)phirange->data)[i] 
                = (((double) i)/((double) philen-1))*(2.0*M_PI);
        }
    }
    if(!phirange || phirange->nd != 1) { 
        PyErr_SetString(PyExc_TypeError, "phirange must be a 1D array");
        goto fail;
    }
    philen = phirange->dimensions[0];

    if(mask_obj) {
        mask = (PyArrayObject *) PyArray_FROM_OTF(mask_obj, NPY_UBYTE, 
            NPY_IN_ARRAY);
    } else {
        npy_intp dims[] = { width, height };
        int i;

        mask = (PyArrayObject *) PyArray_SimpleNew(2, dims, PyArray_UBYTE);
        for(i = 0; i < width*height; i++) {
            ((npy_bool*) mask->data)[i] = 1;
        }
    }
    if(mask && (mask->nd != 2 || 
        mask->dimensions[0] != width || mask->dimensions[1] != height)) {
        PyErr_SetString(PyExc_TypeError, "mask dimensions must match image");
        goto fail;
    }

    // Output arrays
    {
        npy_intp dims[] = { philen -1, radlen-1 };
        bin = (PyArrayObject*) PyArray_SimpleNew(2, dims, PyArray_DOUBLE);
        num = (PyArrayObject*) PyArray_SimpleNew(2, dims, PyArray_INT);
    }

    retval = do_rebin((double *) img->data, width, height, c_x, c_y,
        (double *) radrange->data, radlen, (double *) phirange->data, philen,
        norm, (uchar *) mask->data, (double *) bin->data, (int *) num->data);
  
    if(retval) {
        PyErr_SetString(PyExc_ValueError, 
            "Something went wrong with rebinning");
        goto fail;
    }

    Py_XDECREF(img);
    Py_XDECREF(radrange);
    Py_XDECREF(phirange);
    Py_XDECREF(mask);
    return Py_BuildValue("(O,O)", (PyObject*) bin, (PyObject*) num);
        

fail:
    Py_XDECREF(img);
    Py_XDECREF(radrange);
    Py_XDECREF(phirange);
    Py_XDECREF(mask);
    return NULL;
}
