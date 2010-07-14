import numpy as np
cimport numpy as np
cimport cython

cdef extern from "stdint.h":
    ctypedef unsigned char uint8_t
    ctypedef unsigned short uint16_t

cdef extern int do_rebin(double *img, int xdim, int ydim, \
    double x, double y, double *radrange_in, int radlen, \
    double *phirange_in, int philen, int norm, uint8_t *mask_in,\
    double *a_out, int *n_out)

cdef extern int make_radtable(int xdim, int ydim, double x, double y, \
    double *radrange_in, int radlen, \
    double *phirange_in, int philen, uint8_t *mask,\
    uint16_t *rad_out, uint16_t *phi_out)


def __check_radrange(radrange, imshape, c_x, c_y):
    if radrange is None:
        maxdist = max(map(np.linalg.norm, [[c_x, c_y], [imshape[0] - c_x, c_y], [c_x, imshape[1] - c_y], [imshape[0] - c_x, imshape[1] - c_y]]))
        radrange_arr = np.arange(0.0, np.ceil(maxdist) + 1.0).astype('double')
    else:
        if radrange.ndim != 1:
            raise ValueError
        radrange_arr = radrange.astype('double')
    return radrange_arr


def __check_phirange(phirange):
    if phirange is None:
        phirange_arr = np.linspace(0.0, 2.0*np.pi, num=2).astype('double')
    else:
        if phirange.ndim != 1:
            raise ValueError
        phirange_arr = phirange.astype('double')
    return phirange_arr


def __check_mask(mask, imshape):
    if mask is None:
        mask_arr = np.ones(imshape).astype('uint8')
    else:
        if(mask.ndim != 2 or mask.shape[0] != imshape[0]
            or mask.shape[1] != imshape[1]):
            raise ValueError
        mask_arr = mask.astype('uint8')
    return mask_arr


def radbin(image, double c_x=np.nan, double c_y=np.nan, radrange=None, phirange=None, int norm=1, mask=None):
    """radbin(np.ndarray image, double c_x=np.nan, double c_y=np.nan,
        radrange=None, phirange=None, int norm=1, mask=None)

    Radial rebinning of a 2D image array.
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
    radrange_arr = __check_radrange(radrange, image.shape, c_x, c_y)

    cdef np.ndarray phirange_arr
    phirange_arr = __check_phirange(phirange)

    cdef np.ndarray mask_arr
    mask_arr = __check_mask(mask, image.shape)

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
    cdef uint8_t *mask_in = <uint8_t *> mask_arr.data

    # Output args to C function
    cdef np.ndarray a = np.zeros([phirange_arr.shape[0]-1, radrange_arr.shape[0]-1], dtype='double')
    cdef double *a_out = <double *> a.data
    cdef np.ndarray n = np.zeros([phirange_arr.shape[0]-1, radrange_arr.shape[0]-1], dtype='int')
    cdef int *n_out = <int *> n.data

    retval = do_rebin(img, xdim, ydim, xo, yo, radrange_in, radlen, phirange_in, philen, normp, mask_in, a_out, n_out)

    a = a.squeeze()
    n = n.squeeze()
    return a, n


def bincenters(arr):
    """Calculate centers of bins defined by `arr`.

    Return value has length len(arr)-1.
    """
    return np.array([(arr[i] + arr[i+1])/2.0 for i in range(len(arr)-1)])


def make_radmap(table_shape, center, radrange=None, phirange=None, mask=None):
    """make_radmap(table_shape, center=None, radrange=None, phirange=None, mask=None)

    Return tables of radial and angular indices to a in image with a given
    shape and center.

    The coordinate system of the image is such that the upper left corner
    of the image is at position [0.0, 0.0], i.e. the centers of the pixels
    are at positions [n+0.5, m+0.5].

    Input parameters
        `center`: coordinates of the center in horizontal, vertical
        `radrange`: array containing the edges of radial bins
        `phirange`: array containing the edges of angular bins
        `mask`: mask file, same size as image, image values where mask != 1 are ignored

    Output is a dictionary with the following keys
        `center`: Numpy array with the center coordinates.
        `pbins`: Edges of the phi bins (equal to `phirange`, if given)
        `qbins`: Edges of the radial bins (equal to `radrange`, if given)
        `pcens`: Centers of the phi bins (array of length len(phibins)-1)
        `qcens`: Centers of the radial bins (array of length len(radbins)-1)
        `outshape`: Shape of the resulting array when the map is applied.
        `map`: Numpy array with ndim=3. map[:,:,0] is an array mapping each
            pixel to its phi bin and map[:,:,1] is the radial bin mapping.
    """
    # Check Python args
    if len(table_shape) != 2:
        raise ValueError
    cdef int xdim = table_shape[1]
    cdef int ydim = table_shape[0]

    cdef double c_x, c_y
    if len(center) != 2:
        raise ValueError
    c_x = center[0]
    c_y = center[1]

    cdef np.ndarray radrange_arr
    radrange_arr = __check_radrange(radrange, table_shape, c_x, c_y)

    cdef np.ndarray phirange_arr
    phirange_arr = __check_phirange(phirange)

    cdef np.ndarray mask_arr
    mask_arr = __check_mask(mask, table_shape)

    # Input args to C function call
    cdef double xo = c_x
    cdef double yo = c_y
    cdef double *radrange_in = <double *> radrange_arr.data
    cdef int radlen = radrange_arr.shape[0]
    cdef double *phirange_in = <double *> phirange_arr.data
    cdef int philen = phirange_arr.shape[0]
    cdef uint8_t *mask_in = <uint8_t *> mask_arr.data

    # Output args to C function
    cdef np.ndarray rad = np.zeros(table_shape, dtype=np.uint16)
    cdef uint16_t *rad_out = <uint16_t *> rad.data
    cdef np.ndarray phi = np.zeros(table_shape, dtype=np.uint16)
    cdef uint16_t *phi_out = <uint16_t *> phi.data

    retval = make_radtable(xdim, ydim, xo, yo, radrange_in, radlen, phirange_in, philen, mask_in, rad_out, phi_out)

    maparr = np.zeros((table_shape[0], table_shape[1], 2), dtype=np.uint16)
    maparr[...,0] = phi
    maparr[...,1] = rad
    phicens = bincenters(phirange_arr)
    radcens = bincenters(radrange_arr)
    outd = {"center" : np.array([c_x, c_y]),
            "pbins" : phirange_arr,
            "qbins" : radrange_arr,
            "pcens" : phicens,
            "qcens" : radcens,
            "outshape" : (len(phicens), len(radcens)),
            "map" : maparr }
    return outd


#def map_bin(mapd, frame, norm=True, masked=False):
#    """Apply the map dictionary `mapd` to `frame`, return the rebinned array.
#
#    If `norm` is true, normalize the output array with the number of values
#    summed to each bin.
#
#    If `masked` is true, masked values are in the [0,0] position
#    in the resulting array and values [1:,0] are zeros. Otherwise
#    the first row of the result array is discarded.
#
#    This function is very slow, use maparr2indices and index_bin instead.
#    """
#    mapshape = (mapd["map"].shape[0], mapd["map"].shape[1])
#    if mapshape != frame.shape:
#        raise ValueError("frame shape does not match map shape")
#    mlen = mapshape[0] * mapshape[1]
#    out = np.zeros((mapd["outshape"][0], mapd["outshape"][1]+1), dtype=np.float64)
#    nelems = np.zeros((mapd["outshape"][0], mapd["outshape"][1]+1), dtype=np.uint32)
#    mflat = mapd["map"].ravel()
#    cdef int i, p
#    if norm:
#        for i in xrange(mlen):
#            p = 2*i
#            out[mflat[p], mflat[p+1]] += frame.flat[i]
#            nelems[mflat[p], mflat[p+1]] += 1
#        for i in xrange(np.prod(out.shape)):
#            out.flat[i] / nelems.flat[i]
#    else:
#        for i in xrange(mlen):
#            p = 2*i
#            out[mflat[p], mflat[p+1]] += frame.flat[i]
#    if masked:
#        return out
#    else:
#        return out[:,1:]

@cython.boundscheck(False)
def maparr2indices(np.ndarray[np.uint16_t, ndim=3] marr not None):
    """Return index tables constructed from maparray.

    The return value is an array with shape [p,q] containing
    an index array in each element.
    """
    cdef np.ndarray[np.uint16_t, ndim=1] mflat = marr.ravel()
    cdef int mlen = marr.shape[0] * marr.shape[1]
    cdef int pmax = marr[...,0].max()
    cdef int qmax = marr[...,1].max()
    cdef np.ndarray nelems = np.zeros((pmax+1,qmax+1), dtype=np.int)
    cdef int i, p
    for i in xrange(mlen):
        p = 2*i
        nelems[mflat[p], mflat[p+1]] += 1
    cdef np.ndarray inds = np.zeros((pmax+1,qmax+1), dtype=np.object)
    cdef np.ndarray cnts = np.zeros((pmax+1,qmax+1), dtype=np.int)
    for i in xrange(len(nelems.flat)):
        inds.flat[i] = np.zeros((nelems.flat[i]), dtype=np.uint32)
    for i in xrange(mlen):
        p = 2*i
        inds[mflat[p], mflat[p+1]][cnts[mflat[p], mflat[p+1]]] = i
        cnts[mflat[p], mflat[p+1]] += 1
    return inds[:,1:]


def indbin(indices, frame):
    """Return a pixels of `frame` sorted into bins determined by `indices`.
    """
    if len(indices.shape) == 1:
        indices = np.reshape(indices, (1, len(indices)))
    elif len(indices.shape) != 2:
        raise ValueError
    Irad = np.zeros_like(indices).astype('float64')
    for j in range(indices.shape[1]):
        for i in range(indices.shape[0]):
            ind = indices[i,j].astype('int32')
            Irad[i,j] = np.sum(frame.flat[ind])
            Irad[i,j] /= len(ind)
    return Irad.squeeze()
