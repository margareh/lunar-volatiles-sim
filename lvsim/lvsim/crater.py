
import numpy as np
from numpy.polynomial import Polynomial
from rasterio.transform import rowcol

# crater profile copied from FTmod_Crater class in synthterrain
# used here to create initial profiles when a crater is first added
def profile(dd, diameter, D=200):
    """Returns a numpy array of elevation values based in the input numpy
    array of radius fraction values, such that a radius fraction value
    of 1 is at the rim, less than that interior to the crater, etc.

    A ValueError will be thrown if any values in r are < 0.
    """

    x = np.linspace(-2, 2, D)  # spans a space 2x the diameter.
    xx, yy = np.meshgrid(x, x, sparse=True)  # square domain
    r = np.sqrt(xx**2 + yy**2)

    if not isinstance(r, np.ndarray):
        r = np.ndarray(r)

    out_arr = np.zeros_like(r)

    if np.any(r < 0):
        raise ValueError("The radius fraction value can't be less than zero.")

    inner_idx = np.logical_and(0 <= r, r <= 0.98)
    rim_idx = np.logical_and(0.98 < r, r <= 1.02)
    outer_idx = np.logical_and(1.02 < r, r <= 1.5)

    inner_poly = Polynomial([-0.228809953, 0.227533882, 0.083116795, -0.039499407])
    outer_poly = Polynomial([0.188253307, -0.187050452, 0.01844746, 0.01505647])

    rim_hoverd = 0.036822095

    out_arr[inner_idx] = inner_poly(r[inner_idx])
    out_arr[rim_idx] = rim_hoverd
    out_arr[outer_idx] = outer_poly(r[outer_idx])

    floor = rim_hoverd - (dd)
    out_arr[out_arr < floor] = floor

    return out_arr * diameter

# Stopar depth/diameter ratio for fresh craters
# Copied from synthterrain and modified to be usable with arrays of diameters
def stopar_fresh_dd(diameter):
    """
    Returns a depth/Diameter ratio based on the set of graduated d/D
    categories in Stopar et al. (2017), defined down to 40 m.  This
    function also adds two extrapolated categories.
    """
    # The last two elements are extrapolated
    d_lower_bounds = (0, 10, 40, 100, 200, 400)
    dds = (0.10, 0.11, 0.13, 0.15, 0.17, 0.21)

    dd_list = np.ones_like(diameter) * np.nan
    for d, dd in zip(d_lower_bounds, dds):
        dd_list[diameter >= d] = dd
    return dd_list

# Flag whether surface points are inside or outside of a crater
def in_crater(x, y, diam, dim, res, tf):

    # grid points for surface in meters
    xx, yy = np.meshgrid(np.arange(0, dim*res, res), np.arange(0, dim*res, res))

    # transform center of crater and grid points to row/column
    r_center, c_center = rowcol(tf, x, y)
    r_center /= res
    c_center /= res

    # compute whether or not each point is inside the crater
    r2 = np.power((yy - r_center), 2) + np.power((xx - c_center), 2)
    in_flag = (r2 <= np.power(diam / 2, 2))

    return in_flag
