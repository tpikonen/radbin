import radbin as r
import numpy as np
#import matplotlib.pyplot as plt

def corners_test():
    m = np.zeros((9,9))
    m[0,4] = 1
    m[8,4]= 1
    m[4,0] = 1
    m[4,8]= 1
    (I,n) = r.radbin(m, c_x=4.5, c_y=4.5, radrange=np.arange(0,10), norm=0)
#    plt.plot(I)
#    plt.show()
#    plt.waitforbuttonpress()
    print(I[4])
    assert(I[4] == 4.0)

