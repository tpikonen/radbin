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
    The coordinate system of the image is such that the upper left corner
    of the image is at position [0.0, 0.0], i.e. the centers of the pixels
    are at positions [n+0.5, m+0.5].

    :Parameters:
     - image: A 2D array
     - c_x: coordinate of the center in horizontal direction
     - c_y: coordinate of the center in vertical direction
     - radrange: array containing the edges of radial bins
     - phirange: array containing the edges of angular bins
     - norm: Do we normalize counts with number of pixels in the bin
     - mask: mask file, same size as image, image values where mask != 1 are ignored
    """
    # Check Python args
    if image.ndim != 2:
        raise ValueError
    cdef np.ndarray image_arr = image.astype('double')
    if np.isnan(c_x):
        c_x = image_arr.shape[1] / 2.0
    if np.isnan(c_y):
        c_y = image_arr.shape[0] / 2.0

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
        if(mask.ndim != 2 or mask.shape[0] != image.shape[0]
            or mask.shape[1] != image.shape[1]):
            raise ValueError
        mask_arr = mask.astype('uint8')

    # Input args to C function call
    cdef double *img = <double *> image_arr.data
    if image_arr.strides[0] > image_arr.strides[1]:
        # First index jumps over columns / rows
        center_x = c_x
        center_y = c_y
        xshape = image_arr.shape[1]
        yshape = image_arr.shape[0]
    else:
        # Second index jumps over columns / rows
        center_x = c_y
        center_y = c_x
        xshape = image_arr.shape[0]
        yshape = image_arr.shape[1]
    cdef int xdim = xshape
    cdef int ydim = yshape
    cdef double xo = center_x
    cdef double yo = center_y
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
