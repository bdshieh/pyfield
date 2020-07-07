'''
Utility functions.
'''
import numpy as np
import scipy as sp
import scipy.signal
import scipy.fftpack
from scipy.spatial.distance import cdist


def meshview(v1, v2, v3, mode='cartesian', as_list=True):

    if mode.lower() in ('cart', 'cartesian'):
        x, y, z = np.meshgrid(v1, v2, v3, indexing='ij')
    elif mode.lower() in ('sph', 'spherical'):
        r, theta, phi = np.meshgrid(v1,
                                    np.deg2rad(v2),
                                    np.deg2rad(v3),
                                    indexing='ij')
        x, y, z = sph2cart(r, theta, phi)
    elif mode.lower() in ('sec', 'sector'):
        r, alpha, beta = np.meshgrid(v1,
                                     np.deg2rad(v2),
                                     np.deg2rad(v3),
                                     indexing='ij')
        x, y, z = sec2cart(r, alpha, beta)
    elif mode.lower() in ('dp', 'dpolar'):
        r, alpha, beta = np.meshgrid(v1,
                                     np.deg2rad(v2),
                                     np.deg2rad(v3),
                                     indexing='ij')
        x, y, z = dp2cart(r, alpha, beta)

    if as_list:
        return np.c_[x.ravel('F'), y.ravel('F'), z.ravel('F')]
    else:
        return x, y, z


def sec2cart(r, alpha, beta):
    ''''''
    z = r / np.sqrt(np.tan(alpha)**2 + np.tan(beta)**2 + 1)
    x = z * np.tan(alpha)
    y = z * np.tan(beta)

    return x, y, z


def cart2sec(x, y, z):
    ''''''
    r = np.sqrt(x**2 + y**2 + z**2)
    alpha = np.arccos(z / (np.sqrt(x**2 + z**2))) * np.sign(x)
    beta = np.arccos(z / (np.sqrt(y**2 + z**2))) * np.sign(y)

    return r, alpha, beta


def sph2cart(r, theta, phi):
    ''''''
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)

    return x, y, z


def cart2sph(x, y, z):
    ''''''
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan(y / x)
    phi = np.arccos(z / r)

    return r, theta, phi


def cart2dp(x, y, z):
    ''''''
    r = np.sqrt(x**2 + y**2 + z**2)
    alpha = np.arccos((np.sqrt(y**2 + z**2) / r))
    beta = np.arccos((np.sqrt(x**2 + z**2) / r))

    return r, alpha, beta


def dp2cart(r, alpha, beta):
    ''''''
    z = r * (1 - np.sin(alpha)**2 - np.sin(beta)**2)
    x = r * np.sin(alpha)
    y = r * np.sin(beta)

    return x, y, z


def distance(*args):
    return cdist(*np.atleast_2d(*args))


def gausspulse(fc, fbw, fs, sym=True):
    '''[summary]

    Args:
        fc ([type]): [description]
        fbw ([type]): [description]
        fs ([type]): [description]
        sym (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    '''
    cutoff = scipy.signal.gausspulse('cutoff', fc=fc, bw=fbw, tpr=-100, bwr=-3)
    adj_cutoff = np.ceil(cutoff * fs) / fs

    t = np.arange(-adj_cutoff, adj_cutoff + 1 / fs, 1 / fs)
    pulse, quad = sp.signal.gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)

    if sym:
        return pulse, t
    else:
        return quad, t


def nextpow2(n):
    return 2**int(np.ceil(np.log2(n)))


def envelope(rf_data, N=None, axis=-1):
    return np.abs(scipy.signal.hilbert(np.atleast_2d(rf_data), N, axis=axis))


def qbutter(x, fn, fs=1, btype='lowpass', n=4, plot=False, axis=-1):

    wn = fn / (fs / 2.)
    b, a = sp.signal.butter(n, wn, btype)
    fx = sp.signal.lfilter(b, a, x, axis=axis)

    return fx


def qfirwin(x,
            fn,
            fs=1,
            btype='lowpass',
            ntaps=80,
            plot=False,
            axis=-1,
            window='hamming'):
    if btype.lower() in ('lowpass', 'low'):
        pass_zero = 1
    elif btype.lower() in ('bandpass', 'band'):
        pass_zero = 0
    elif btype.lower() in ('highpass', 'high'):
        pass_zero = 0

    wn = fn / (fs / 2.)
    b = sp.signal.firwin(ntaps, wn, pass_zero=pass_zero, window=window)
    fx = np.apply_along_axis(lambda x: np.convolve(x, b), axis, x)

    return fx


def qfft(s, nfft=None, fs=1):
    '''
    Quick FFT plot. Returns frequency bins and FFT in dB.
    '''
    s = np.atleast_2d(s)

    nsig, nsample = s.shape

    if nfft is None:
        nfft = nsample

    if nfft > nsample:
        s = np.pad(s, ((0, 0), (0, nfft - nsample)), mode='constant')
    elif nfft < nsample:
        s = s[:, :nfft]

    ft = sp.fftpack.fft(s, axis=1)
    freqs = sp.fftpack.fftfreq(nfft, 1 / fs)

    cutoff = (nfft + 1) // 2

    return freqs[:cutoff], ft[:, :cutoff].squeeze()


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# def qfft(s, nfft=None, fs=1, dr=100, fig=None, **kwargs):
#     '''
#     Quick FFT plot. Returns frequency bins and FFT in dB.
#     '''
#     s = np.atleast_2d(s)

#     nsig, nsample = s.shape

#     if nfft is None:
#         nfft = nsample

#     if nfft > nsample:
#         s = np.pad(s, ((0, 0), (0, nfft - nsample)), mode='constant')
#     elif nfft < nsample:
#         s = s[:, :nfft]

#     ft = sp.fftpack.fft(s, axis=1)
#     freqs = sp.fftpack.fftfreq(nfft, 1 / fs)

#     ftdb = 20 * np.log10(np.abs(ft) / (np.max(np.abs(ft), axis=1)[..., None]))
#     ftdb[ftdb < -dr] = -dr

#     cutoff = (nfft + 1) // 2

#     return freqs[:cutoff], ftdb[:, :cutoff]


def concatenate_with_padding(rf_data, t0s, fs, axis=-1):

    if len(rf_data) <= 1:
        return np.atleast_2d(rf_data), t0s[0]

    rf_data = np.atleast_2d(*rf_data)

    mint0 = float(min(t0s))
    frontpads = [int(np.ceil((t - mint0) * fs)) for t in t0s]
    maxlen = max([fpad + rf.shape[1] for fpad, rf in zip(frontpads, rf_data)])
    backpads = [
        maxlen - (fpad + rf.shape[1]) for fpad, rf in zip(frontpads, rf_data)
    ]

    new_data = []

    for rf, fpad, bpad in zip(rf_data, frontpads, backpads):

        new_rf = np.pad(rf, ((0, 0), (fpad, bpad)), mode='constant')
        new_data.append(new_rf)

    if axis == 2:
        return np.stack(new_data, axis=axis), mint0
    else:
        return np.concatenate(new_data, axis=axis), mint0


def sum_with_padding(rfdata, t0s=None, fs=1, axis=-1):
    '''[summary]

    Args:
        rfdata ([type]): [description]
        t0s ([type], optional): [description]. Defaults to None.
        fs (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    '''
    if len(rfdata) == 1:
        return rfdata[0], t0s[0]

    nsig = len(rfdata)
    # rfdata = np.atleast_2d(*rfdata)
    shape = rfdata[0].shape
    ndim = len(shape)

    if t0s is None:
        t0s = [0] * nsig
    mint0 = min(t0s)

    frontpads = [int(np.ceil((t - mint0) * fs)) for t in t0s]
    maxlen = max([fpad + shape[axis] for fpad, rf in zip(frontpads, rfdata)])
    backpads = [
        maxlen - (fpad + shape[axis]) for fpad, rf in zip(frontpads, rfdata)
    ]

    # new_data = []
    # sumrf = np.zeros(maxlen)
    newshape = list(shape)
    newshape[axis] = maxlen
    sumrf = np.zeros(newshape)

    for rf, fpad, bpad in zip(rfdata, frontpads, backpads):
        padwidth = [
            (0, 0),
        ] * ndim
        padwidth[axis] = (fpad, bpad)
        sumrf += np.pad(rf, padwidth, mode='constant')
        # new_data.append(new_rf)

    # return np.sum(new_data, axis=0).squeeze(), mint0
    return sumrf, mint0


# def sum_with_padding(rf_data, t0s, fs):

#     if len(rf_data) <= 1:
#         return np.atleast_2d(rf_data[0]), t0s[0]

#     rf_data = np.atleast_2d(*rf_data)

#     mint0 = min(t0s)
#     frontpads = [int(np.ceil((t - mint0) * fs)) for t in t0s]
#     maxlen = max([fpad + rf.shape[1] for fpad, rf in zip(frontpads, rf_data)])
#     backpads = [
#         maxlen - (fpad + rf.shape[1]) for fpad, rf in zip(frontpads, rf_data)
#     ]

#     new_data = []

#     for rf, fpad, bpad in zip(rf_data, frontpads, backpads):

#         new_rf = np.pad(rf, ((0, 0), (fpad, bpad)), mode='constant')
#         new_data.append(new_rf)

#     return np.sum(new_data, axis=0), mint0


def memoize(func):
    '''
    Simple memoizer to cache repeated function calls.
    '''
    def ishashable(obj):
        try:
            hash(obj)
        except TypeError:
            return False
        return True

    def make_hashable(obj):
        if not ishashable(obj):
            return str(obj)
        return obj

    memo = {}

    def decorator(*args):
        key = tuple(make_hashable(a) for a in args)
        if key not in memo:
            memo[key] = func(*args)
        return memo[key]

    return decorator