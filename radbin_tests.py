import radbin as r
import numpy as np
import matplotlib.pyplot as plt

def coordinate_test():
    m = np.zeros((9,9))
    m[0,4] = 1
    m[8,4]= 1
    m[4,0] = 1
    m[4,8]= 1
    (I,n) = r.radbin(m, (4.5, 4.5), radrange=np.arange(0,10), norm=0)
#    plt.plot(I)
#    plt.show()
#    plt.waitforbuttonpress()
#    print(I[4])
    assert(I[4] == 4.0)
    m = np.zeros((10,5))
    m[0,3] = 1
    (I,n) = r.radbin(m, (3.5, 5.5), radrange=np.arange(0,10), norm=0)
#    plt.plot(I)
#    plt.show()
    assert(I[5] == 1.0)
    I[5] = 0.0
    assert(not I.all())
    (I,n) = r.radbin(m, (3.5, 0.5), radrange=np.arange(0,10), norm=0)
    assert(I[0] == 1.0)
    I[0] = 0.0
    assert(not I.all())

def symmetric_matrix():
    xx = np.linspace(-1, 1, num=100)
    ons = np.ones_like(xx)
    rr = sqrt((np.outer(ons, xx**2)+np.outer(ons, xx**2).T))

