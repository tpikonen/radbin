import numpy as np
cimport numpy as np
#DTYPE = np.double
#ctypedef np.double_t DTYPE_t
ctypedef char uchar

cdef extern int do_rebin(double *img, int xdim, int ydim, \
    double x, double y, double *radrange_in, int radlen, \
    double *phirange_in, int philen, int norm, uchar *mask_in,\
    double *a_out, int *n_out)

def radbin(np.ndarray image, double c_x=np.nan, double c_y=np.nan, radrange=None, phirange=None, int norm=1, mask=None):
    """radbin(np.ndarray image, double c_x=np.nan, double c_y=np.nan,
        radrange=None, phirange=None, int norm=1, mask=None)

    Radial rebinning of (SAS) data.

    :Parameters:
     -
    """
    # Check Python args
    if image.ndim != 2:
        raise ValueError
    cdef np.ndarray image_arr = image.astype('double')
    if np.isnan(c_x):
        c_x = image_arr.shape[0] / 2.0
    if np.isnan(c_y):
        c_y = image_arr.shape[1] / 2.0

    cdef np.ndarray radrange_arr
    if radrange == None:
        maxdist = max(map(np.linalg.norm, [[c_x, c_y], [image.shape[0] - c_x, c_y], [c_x, image.shape[1] - c_y], [image.shape[0] - c_x, image.shape[1] - c_y]]))
        radrange_arr = np.arange(0.0, np.ceil(maxdist) + 1.0).astype('double')
    else:
        if radrange.ndim != 1:
            raise ValueError
        radrange_arr = radrange.astype('double')

    cdef np.ndarray phirange_arr
    if phirange == None:
        phirange_arr = np.linspace(0.0, 2.0*np.pi, num=2).astype('double')
    else:
        if phirange.ndim != 1:
            raise ValueError
        phirange_arr = phirange.astype('double')

    cdef np.ndarray mask_arr
    if mask == None:
        mask_arr = np.ones_like(image).astype('uint8')
    else:
        if(mask.shape[0] != image.shape[0]
            or mask.shape[1] != image.shape[1]):
            raise ValueError
        mask_arr = mask.astype('uint8')

    # Input args to C function call
    cdef double *img = <double *> image_arr.data
    cdef int xdim = image_arr.shape[0]
    cdef int ydim = image_arr.shape[1]
    cdef double xo = c_x
    cdef double yo = c_y
    cdef double *radrange_in = <double *> radrange_arr.data
    cdef int radlen = radrange_arr.shape[0]
    cdef double *phirange_in = <double *> phirange_arr.data
    cdef int philen = phirange_arr.shape[0]
    cdef int normp = norm
    cdef uchar *mask_in = <uchar *> mask_arr.data

    # Output args to C function
    cdef np.ndarray a = np.zeros([phirange_arr.shape[0]-1, radrange_arr.shape[0]-1], dtype='double')
    cdef double *a_out = <double *> a.data
    cdef np.ndarray n = np.zeros([phirange_arr.shape[0]-1, radrange_arr.shape[0]-1], dtype='int')
    cdef int *n_out = <int *> n.data

    retval = do_rebin(img, xdim, ydim, xo, yo, radrange_in, radlen, phirange_in, philen, normp, mask_in, a_out, n_out)

    a = a.squeeze()
    n = n.squeeze()
    return a, n
