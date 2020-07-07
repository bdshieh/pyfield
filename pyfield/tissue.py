'''
'''
import numpy as np
import scipy as sp
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter

from pyfield import util


def bsc_to_fir(freqs, bsc, area, ns, fs, ntap=100, window='hamming'):
    '''
    FIR filter design based on a desired backscattering coefficient spectrum.

    Parameters
    ----------
    freqs : array_like, 1-D
        The frequency sampling points in Hz.
    bsc : array_like, 1-D
        The backscattering coefficient at the frequency sampling
        points in 1 / (Sr * m).
    area : float
        The surface area of the receive aperture in m^2.
    ns : float
        The scatterer density in number of scatterers per m^3.
    fs : float
        The sampling frequency in Hz.
    ntap : int, optional
        The number of taps in the FIR filter. Defaults to 100.
    window : str, optional
        The window function used in the filter design. Defaults to 'hamming'.

    Returns
    -------
    taps: ndarray
        The taps of the FIR filter.
    '''
    freqs, bsc = np.atleast_1d(freqs, bsc)
    assert len(freqs) == len(bsc)

    # check first frequency bin is zero
    if not np.isclose(freqs[0], 0):
        freqs = np.insert(freqs, 0, 0)
        bsc = np.insert(bsc, 0, 0)

    # check last frequency bin is nyquist
    if not np.isclose(freqs[-1], fs / 2):
        freqs = np.append(freqs, fs / 2)
        bsc = np.append(bsc, 0)

    # normalize gains so that the filter results in the mean pressure over the
    # receive aperture
    gains = 2 * np.pi / (area * np.sqrt(ns)) * np.sqrt(np.abs(bsc))

    # create the filter
    fir = sp.signal.firwin2(ntap,
                            freqs,
                            gains,
                            fs=fs,
                            antisymmetric=False,
                            window=window)

    return fir


def calc_path_att(r0, r1, phantom, info=False):
    '''
    One-way attenuation along a given path in a simulated phantom.

    Parameters
    ----------
    r0 : array_like, 1-D
        The (x, y, z) coordinates of the path start point.
    r1 : array_like, 1-D
        The (x, y, z) coordinates of the path end point.
    phantom : dict
        A nested `dict` containing the geometry and material information of the
        phantom.
    info : bool, optional
        If True, also returns path-related meta information. Default is False

    Returns
    -------
    att_total : float
        The total attenuation. Always returned.
    points : ndarray, 2-D
        (x, y, z) coordinates of segment points along the path.
        Only returned if `info` is True.
    midpoints : ndarray, 2-D
        (x, y, z) coordinates of segment mid-points. 
        Only returned if `info` is True.
    path_lengths : ndarray
        Lengths of each segment. Only returned if `info` is True.
    att_coeffs : ndarray
        Attenuation coefficient of each segment. 
        Only returned if `info` is True.
    '''
    r0, r1 = np.atleast_1d(r0, r1)
    eps = np.finfo(np.float64).eps
    Dx, Dy, Dz = (r1 - r0).astype(np.float64)
    x0, y0, z0 = r0.astype(np.float64)
    x1, y1, z1 = r1.astype(np.float64)

    xflag = abs(Dx) < eps
    yflag = abs(Dy) < eps
    zflag = abs(Dz) < eps

    px = phantom['planes_x']
    py = phantom['planes_y']
    pz = phantom['planes_z']
    px, py, pz = np.atleast_1d(px, py, pz)

    if xflag and yflag:  # line parallel to z-axis

        def fz(z):
            x = x0
            y = y0
            return x, y, z

        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))

        zmin = min(z0, z1)
        zmax = max(z0, z1)

        for z in pz:
            if z < zmin:
                continue
            if z > zmax:
                continue
            points.append(fz(z))

        points = list(set(points))
        points.sort(key=itemgetter(2), reverse=(z1 < z0))
        points = np.array(points)
    elif yflag and zflag:  # line parallel to x-axis

        def fx(x):
            y = y0
            z = z0
            return x, y, z

        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))

        xmin = min(x0, x1)
        xmax = max(x0, x1)

        for x in px:
            if x < xmin:
                continue
            if x > xmax:
                continue
            points.append(fx(x))

        points = list(set(points))
        points.sort(key=itemgetter(0), reverse=(x1 < x0))
        points = np.array(points)
    elif xflag and zflag:  # line parallel to y-axis

        def fy(y):
            x = x0
            z = z0
            return x, y, z

        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))

        ymin = min(y0, y1)
        ymax = max(y0, y1)

        for y in py:
            if y < ymin:
                continue
            if y > ymax:
                continue
            points.append(fy(y))

        points = list(set(points))
        points.sort(key=itemgetter(1), reverse=(y1 < y0))
        points = np.array(points)
    elif xflag:  # line on y-z plane
        dzdy = Dz / Dy

        def fy(y):
            x = x0
            z = dzdy * (y - y0) + z0
            return x, y, z

        def fz(z):
            x = x0
            y = (z - z0) / dzdy + y0
            return x, y, z

        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))

        ymin = min(y0, y1)
        zmin = min(z0, z1)
        ymax = max(y0, y1)
        zmax = max(z0, z1)

        for y in py:
            if y < ymin:
                continue
            if y > ymax:
                continue
            points.append(fy(y))

        for z in pz:
            if z < zmin:
                continue
            if z > zmax:
                continue
            points.append(fz(z))

        points = list(set(points))
        points.sort(key=itemgetter(1), reverse=(y1 < y0))
        points = np.array(points)
    elif yflag:  # line on x-z plane
        dzdx = Dz / Dx

        def fx(x):
            y = y0
            z = dzdx * (x - x0) + z0
            return x, y, z

        def fz(z):
            x = (z - z0) / dzdx + x0
            y = y0
            return x, y, z

        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))

        xmin = min(x0, x1)
        zmin = min(z0, z1)
        xmax = max(x0, x1)
        zmax = max(z0, z1)

        for x in px:
            if x < xmin:
                continue
            if x > xmax:
                continue
            points.append(fx(x))

        for z in pz:
            if z < zmin:
                continue
            if z > zmax:
                continue
            points.append(fz(z))

        points = list(set(points))
        points.sort(key=itemgetter(0), reverse=(x1 < x0))
        points = np.array(points)
    elif zflag:  # line on x-y plane
        dydx = Dy / Dx

        def fx(x):
            y = dydx * (x - x0) + y0
            z = z0
            return x, y, z

        def fy(y):
            x = (y - y0) / dydx + x0
            z = z0
            return x, y, z

        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))

        xmin = min(x0, x1)
        ymin = min(y0, y1)
        xmax = max(x0, x1)
        ymax = max(y0, y1)

        for x in px:
            if x < xmin:
                continue
            if x > xmax:
                continue
            points.append(fx(x))

        for y in py:
            if y < ymin:
                continue
            if y > ymax:
                continue
            points.append(fy(y))

        points = list(set(points))
        points.sort(key=itemgetter(0), reverse=(x1 < x0))
        points = np.array(points)
    else:  # line in quadrant
        dzdx = Dz / Dx
        dzdy = Dz / Dy
        dydx = Dy / Dx

        def fx(x):
            y = dydx * (x - x0) + y0
            z = dzdx * (x - x0) + z0
            return x, y, z

        def fy(y):
            x = (y - y0) / dydx + x0
            z = dzdy * (y - y0) + z0
            return x, y, z

        def fz(z):
            x = (z - z0) / dzdx + x0
            y = (z - z0) / dzdy + y0
            return x, y, z

        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))

        xmin = min(x0, x1)
        ymin = min(y0, y1)
        zmin = min(z0, z1)
        xmax = max(x0, x1)
        ymax = max(y0, y1)
        zmax = max(z0, z1)

        for x in px:
            if x < xmin:
                continue
            if x > xmax:
                continue
            points.append(fx(x))

        for y in py:
            if y < ymin:
                continue
            if y > ymax:
                continue
            points.append(fy(y))

        for z in pz:
            if z < zmin:
                continue
            if z > zmax:
                continue
            points.append(fz(z))

        points = list(set(points))
        points.sort(key=itemgetter(0), reverse=(x1 < x0))
        points = np.array(points)

    midpoints = (points[0:-1, :] + points[1:, :]) / 2.
    path_lengths = np.zeros(midpoints.shape[0])
    att_coeffs = np.zeros(midpoints.shape[0])

    for idx, mp in enumerate(midpoints):
        x, y, z = mp
        path_lengths[idx] = (float(
            util.distance(points[idx, :], points[idx + 1, :])))

        xarg = np.where((x - px) >= 0.0)[0][-1]
        if xarg > (px.size - 2):
            xarg = px.size - 2
        yarg = np.where((y - py) >= 0.0)[0][-1]
        if yarg > (py.size - 2):
            yarg = py.size - 2
        zarg = np.where((z - pz) >= 0.0)[0][-1]
        if zarg > (pz.size - 2):
            zarg = pz.size - 2

        att_coeffs[idx] = phantom[(xarg, yarg, zarg)]['att']

    att_total = np.product(np.exp(-att_coeffs * path_lengths))

    if info:
        return att_total, points, midpoints, path_lengths, att_coeffs
    else:
        return att_total


def draw_phantom(phantom, colormap=None, ax=None, **kwargs):
    '''
    3-D plot of phantom geometry.

    Parameters
    ----------
    phantom : dict
        A nested `dict` containing the geometry and material information of the
        phantom.
    colormap : dict, optional
        A dict mapping material names (keys) to their display colors (values).
        Default is None.
    ax : `matplotlib.axes.Axes`, optional
        The axes to plot to. Default is None.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Only returned if `ax` is None.
    ax : `matplotlib.axes.Axes`
        Only returned if `ax` is None.
    '''
    if ax is None:
        makefig = True
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_axes_off()
    else:
        makefig = False

    px = phantom['planes_x']
    py = phantom['planes_y']
    pz = phantom['planes_z']

    # draw grid/boxes
    x, y, z = np.meshgrid(px, py, pz)
    filled = np.ones((len(px) - 1, len(py) - 1, len(pz) - 1))

    colors = np.empty_like(filled, dtype='object')
    if colormap is None:
        colors[:] = '#ffffff00'
    else:
        for i in range(colors.shape[0]):
            for j in range(colors.shape[1]):
                for k in range(colors.shape[2]):
                    key = phantom[i, j, k]['material']
                    colors[i, j, k] = colormap[key]

    ax.voxels(x, y, z, filled, facecolors=colors, edgecolors='gray', **kwargs)

    if makefig:
        util.set_axes_equal(ax)
        return fig, ax


def draw_path_att(r0, r1, phantom, ax=None, **kwargs):
    '''
    [summary]

    Parameters
    ----------
    r0 : array_like, 1-D
        (x, y, z) coordinates of the path start point.
    r1 : array_like, 1-D
        (x, y, z) coordinates of the path end point.
    phantom : dict
        A nested `dict` containing the geometry and material information of the
        phantom.
    ax : `matplotlib.axes.Axes`, optional
        The axes to plot to. Default is None.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Only returned if `ax` is None.
    ax : `matplotlib.axes.Axes`
        Only returned if `ax` is None.
    '''
    info = calc_path_att(r0, r1, phantom, info=True)
    points = info[1]
    # midpoints = info[2]

    px = phantom['planes_x']
    py = phantom['planes_y']
    pz = phantom['planes_z']

    if ax is None:
        makefig = True
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        makefig = False

    # draw grid/boxes
    x, y, z = np.meshgrid(px, py, pz)

    # plot path
    ax.plot([r0[0], r1[0]], [r0[1], r1[1]], [r0[2], r1[2]], 'b-', label='Path')

    # plot start and end points
    ax.plot([r0[0], r1[0]], [r0[1], r1[1]], [r0[2], r1[2]],
            'b.',
            label='Start/End points')

    # plot intersection points
    if points.shape[0] > 2:
        ax.plot(points[1:-1, 0],
                points[1:-1, 1],
                points[1:-1, 2],
                'rx',
                label='Intersection points')

    if makefig:
        util.set_axes_equal(ax)
        return fig, ax


'''Backscattering coefficient spectrum in units of 1/(Sr*m) of human blood at 8% hematocrit.
Source: K. K. Shung et. al. 

Returns:
    [type]: [description]
'''


def bsc_human_blood():
    '''
    Backscattering coefficient spectrum of human blood.

    Measured experimentally for blood with 8% hematocrit from 0 to 15 MHz [1].

    Returns
    -------
    freqs : ndarray, 1-D
        The frequency sample points in Hz.
    bsc : ndarray, 1-D
        The backscattering coefficient at the frequency sample
        points in 1 / (Sr * m).
    
    References
    -------
    .. [1] K. K. Shung et. al...
    '''
    freqs = [0., 5e6, 7e6, 8.5e6, 10e6, 15e6]
    bsc = [0., 0.001, 0.003004, 0.005464, 0.008702, 0.06394]

    return np.array(freqs), np.array(bsc)


def bsc_human_blood_powerfit(freqs=None, append=None):
    '''
    Power law fit of the human blood backscattering coefficient spectrum.

    Parameters
    ----------
    freqs : array_like, 1-D, optional
        The frequency sample points. If None, samples points will span
        from 0 to 20 MHz in 1 MHz steps. Default is None.
    append : sequence, optional
        Data points to append in terms of [[freqs,], [bsc,]]. Default is None

    Returns
    -------
    freqs : ndarray, 1-D
        The frequency sample points in Hz.
    bsc : ndarray, 1-D
        The backscattering coefficient at the frequency sample
        points in 1 / (Sr * m).
    '''
    if freqs is None:
        freqs = np.arange(0, 20e6 + 0.5e6, 1e6)

    a = 9.521356320585827e-29
    b = 3.724925809643933
    bsc = a * (freqs**b)

    if append is not None:
        freqs = np.append(freqs, append[0])
        bsc = np.append(bsc, append[1])

    return freqs, bsc


def bsc_canine_myocardium():
    '''
    Backscattering coefficient spectrum of canine myocardium.

    Measured experimentally from 0 to 10 MHz [1].

    Returns
    -------
    freqs : ndarray, 1-D
        The frequency sample points in Hz.
    bsc : ndarray, 1-D
        The backscattering coefficient at the frequency sample
        points in 1 / (Sr * m).
    
    References
    -------
    .. [1] O'Donnell et. al...
    '''
    freqs = [0., 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 10e6]
    bsc = [0., 2.4e-2, 1e-1, 3e-1, 5.4e-1, 9e-1, 1.4, 2.0, 3.0, 4.0]

    return np.array(freqs), np.array(bsc)


def bsc_canine_myocardium_powerfit(freqs=None, append=None):
    '''
    Power law fit of the canine myocardium backscattering coefficient spectrum.

    Parameters
    ----------
    freqs : array_like, 1-D, optional
        The frequency sample points. If None, samples points will span
        from 0 to 20 MHz in 1 MHz steps. Default is None.
    append : sequence, optional
        Data points to append in terms of [[freqs,], [bsc,]]. Default is None

    Returns
    -------
    freqs : ndarray, 1-D
        The frequency sample points in Hz.
    bsc : ndarray, 1-D
        The backscattering coefficient at the frequency sample
        points in 1 / (Sr * m).
    '''
    if freqs is None:
        freqs = np.arange(0, 20e6 + 0.5e6, 1e6)

    a = 5.572634270689345e-22
    b = 3.126758654375955
    bsc = a * (freqs**b)

    if append is not None:
        freqs = np.append(freqs, append[0])
        bsc = np.append(bsc, append[1])

    return freqs, bsc


def cardiac_penetration_phantom(dim=(0.01, 0.01, 0.1),
                                zthick=2e-3,
                                dz=0.01,
                                blood_att=14.0,
                                myo_att=58.0,
                                ns=5e6):
    '''
    Simulated phantom for assessing penetration in the heart.

    The phantom consists of alternating layers of blood and myocardium along
    the propagation path.

    Parameters
    ----------
    dim : tuple, optional
        The dimensions of the phantom in (x, y, z). Default is (0.01, 0.01, 0.1)
    zthick : float, optional
        The thickness of the myocardium layers in m. Default is 2e-3.
    dz : float, optional
        The spacing between myocardium layers in m. Default is 0.01
    blood_att : float, optional
        The attenuation coefficient of blood in Np/m. Default is 14.0
    myo_att : float, optional
        The attenuation coefficient of myocardium in Np/m. Default is 58.0
    ns : float, optional
        The scatterer density in number of scatterers per m^3. Default is 5e6

    Returns
    -------
    phantom : dict
        [description]
    '''
    # att_blood = 0.14 * 100  # 0.14 Np/cm which is about 1.25 dB/cm
    # att_heart = 0.58 * 100  # 0.58 Np/cm which is about 5 dB/cm
    # zthick = 0.002

    length_x, length_y, length_z = dim

    # define phantom geometry
    planes_x = [-length_x / 2, length_x / 2]
    planes_y = [-length_x / 2, length_x / 2]
    planes_z = [0.0, 0.001]
    # add reflecting layer (myocardium) every dz with thickness zthick
    for i in np.arange(dz, length_z + dz / 2, dz):
        planes_z.append(i)
        planes_z.append(i + zthick)

    phantom = {}
    for i in range(len(planes_x) - 1):
        for j in range(len(planes_y) - 1):
            for k in range(len(planes_z) - 1):

                xmin, xmax = planes_x[i], planes_x[i + 1]
                ymin, ymax = planes_y[j], planes_y[j + 1]
                zmin, zmax = planes_z[k], planes_z[k + 1]
                dim_x = xmax - xmin
                dim_y = ymax - ymin
                dim_z = zmax - zmin
                center = [xmin + dim_x / 2, ymin + dim_y / 2, zmin + dim_z / 2]
                bbox = ((xmin, xmax), (ymin, ymax), (zmin, zmax))

                if k == 0:
                    att = 0
                    scat = []
                    material = 'none'
                else:
                    if k % 2 == 0:
                        att = myo_att
                        material = 'myocardium'
                    else:
                        att = blood_att
                        material = 'blood'

                    nscat = int(round(ns * dim_x * dim_y * dim_z))
                    scat = np.c_[sp.rand(nscat) * dim_x,
                                 sp.rand(nscat) * dim_y,
                                 sp.rand(nscat) * dim_z] - np.array(
                                     [dim_x, dim_y, dim_z]) / 2 + center

                phantom[(i, j, k)] = dict(material=material,
                                          att=att,
                                          scat=scat,
                                          center=center,
                                          bbox=bbox)

    phantom['planes_x'] = planes_x
    phantom['planes_y'] = planes_y
    phantom['planes_z'] = planes_z

    return phantom


if __name__ == '__main__':
    pass
