#include "Python.h"
#include "arrayobject.h"
#include <stdbool.h>

typedef unsigned char uchar;

extern int do_rebin(const double *img, int xdim, int ydim, double x, double y, \
    const double *radrange_in, int radlen, \
    const double *phirange_in, int philen, int norm, const uchar *mask,\
    double *a_out, int *n_out);

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
