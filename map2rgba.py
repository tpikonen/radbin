import numpy as np
import matplotlib.pyplot as plt
import radbin as r

_cyclebits = 3


def marr2rgba(marr):
    """Return rgba array constructed from maparray.
    """
    mangle = np.vectorize(inds2rgba, otypes=[np.uint32])
    rgba = mangle(marr[:,:,0], marr[:,:,1])
    im = rgba.view(dtype=np.uint8, type=np.ndarray)
    im = im.reshape((marr.shape[0], marr.shape[1], 4))
    return im


def write_maparr_png(marr, fname):
    """Write a mapping array `marr` to an RGBA PNG file `fname`.

    The mapping array has to have three indices in the form (M,N,2).
    It defines a mapping from 2-dimensional arrays in domain (M,N)
    to 2-dimensional arrays with the range implicitly defined by the
    largest indices in the mapping array, i.e. arrays in the range have
    the shape (max(marr[:,:,0]), max(marr[:,:,1])).
    """
    import PIL.Image
    im = marr2rgba(marr)
    pim = PIL.Image.fromarray(im)
    ff = open(fname, "w")
    pim.save(ff, 'png')
    ff.close()


def read_maparr_png(fname):
    """Return a mapping array read from RGBA PNG file `fname`.
    """
    import matplotlib.image as mpli
    im = (0xff*mpli.imread(fname)).astype(np.uint8)
    rgba = im.view(dtype=np.uint32).squeeze()
    unmangle = np.vectorize(rgba2inds)
    (p, q) = unmangle(rgba)
    s = q.shape
    maparr = np.zeros((s[0], s[1], 2), dtype=np.uint16)
    maparr[...,0] = p
    maparr[...,1] = q
    return maparr


def plot_radmap(rmap):
    """Display a dict containing a mapping array.
    """
    im = marr2rgba(rmap['map'])
    plt.imshow(im)


def plot_radind(radind):
    """Display a dict containing radial indices.
    """
    marr = r.indices2maparr(radind['imshape'], radind['indices'])
    plt.imshow(marr2rgba(marr))


def inds2rgba(p, q):
    """Return a 32-bit RGBA value calculated from two indices `p` and `q`.

    `p` and `q` should be a positive integers smaller than 0xffff.

    LSB of `q` is stored in the blue channel.

    MSB of `q` is stored in the green channel.

    Three least significant bits of `p` are stored in bit 8 of the
    red channel. Rest of the bits LSB of `p` are shifted by one and
    stored in the lowest 5 bits of the red channel.

    MSB of `p` is subtracted from 0xff and stored in the alpha channel.

    Value 0 in the alpha channel is reserved for ignored values in the
    inverse mapping from RGBA to indices.

    (The motivation of this mangling is to easily visualize the
    mapping indices in an RGBA image.)
    """
    # FIXME: return a tuple or something sane
    assert(p < 0xffff)
    # Make junk values from q == 0 bins transparent in the PNG
    if q == 0:
        phi = 0
    else:
        phi = 0xff - (p >> 8)
    plsb = p & 0xff
    plo = (plsb << (8 - _cyclebits) | (plsb >> _cyclebits)) & 0xff
    qlo = q & 0xff
    qhi = (q >> 8) & 0xff
    return (phi << 24) | (qlo << 16) | (qhi << 8) | plo
#    return np.array([plo, qlo, qhi, phi])


def rgba2inds(rgba):
    """Return two indices (p, q) from a 32-bit RGBA value.

    See inds2rgba for further information.
    """
    a = (rgba >> 24) & 0xff
    b = (rgba >> 16) & 0xff
    g = (rgba >>  8) & 0xff
    r = rgba & 0xff
    if a == 0:
        q = 0
        p = 0
    else:
        q = (g << 8) | b
        plsb = ((r << _cyclebits) | (r >> (8 - _cyclebits))) & 0xff
        pmsb = 0xff - a
        p = (pmsb << 8) | plsb
    return (p, q)

