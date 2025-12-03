# test

import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

def test(args):
    x = args[0]
    y = args[1]
    return(x*y)

if __name__ == "__main__":

    # make sparse meshgrid
    grdy, grdx = np.meshgrid(
        np.arange(0, 10, 1),
        np.arange(0, 5, 1),
        sparse=True,
        indexing="ij",
    )

    # print out grdx and grdy
    # print(grdx.shape) # 1 x 10
    # print(grdy.shape) # 10 x 1
    n = grdx.shape[-1]
    m = grdy.shape[0]

    # now make args list
    args = [(x,y) for x in grdx for y in grdy]
    print(args)

    # now run through multiprocessing and reshape
    with Pool() as p:
        out = p.map(test, args)

    out = np.array(out).reshape(m,n)
    print(out)
    print(grdx * np.ones((m,1)) * grdy * np.ones((1,n)))

    # plot things
    fig, ax = plt.subplots(1,3, figsize=(30,10))
    ax[0].imshow(grdx * np.ones((m, 1)))
    ax[1].imshow(grdy * np.ones((1, n)))
    ax[2].imshow(out)
    ax[0].set_title('X')
    ax[1].set_title('Y')
    ax[2].set_title('OUT')
    plt.show()
