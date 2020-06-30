'''
'''
import numpy as np
import scipy as sp
from scipy.signal import resample
from tqdm import tqdm

from pyfield.tissue import calc_path_att, bsc_to_fir, bsc_to_filt

##################################################################

# simulation parameters
center_frequency = 7e6
bandwidth = center_frequency * 0.8
sample_frequency = 200e6
sound_speed = 1500.
attenuation = 0
density = 1000.
frequency_attenuation = 0.
attenuation_center_frequency = center_frequency
use_attenuation = 1
rx_area = 35e-6 * 35e-6 * 4

# phantom paramters
ns = 5 * 1000**3  # scatterer density
# blood_att = 0.14*100 # 0.14 Np/cm which is about 1.25 dB/cm
# heart_att = 0.58*100 # 0.58 Np/cm which is about 5 dB/cm
blood_att = 0.14 * (5.6**1.21) / 8.6886 * 100  # in Np/m
heart_att = 0.52 * (5.6**1) / 8.6886 * 100  # in Np/m

# excitation parameters
path_to_excitation_file = 'd:/data/tissue simulation/simulation/data_2x2_nonlinear_sim.npz'

##################################################################


def resampler(signal, fs, new_fs):

    nsample = int(round(len(signal) / fs * new_fs))
    return resample(signal, nsample, window='hanning')


# SETUP PHANTOM
xs = np.array([-0.005, 0.005])
ys = np.array([-0.005, 0.005])
zs = np.array([
    0.0, 0.001, 0.03, 0.032, 0.04, 0.042, 0.05, 0.052, 0.06, 0.062, 0.07,
    0.072, 0.08, 0.082, 0.09, 0.092, 0.1, 0.102, 0.11, 0.112, 0.12, 0.122,
    0.13, 0.132, 0.14, 0.142, 0.15, 0.152
])

atts = {}
atts[(0, 0, 0)] = 0.0
atts[(0, 0, 1)] = blood_att
atts[(0, 0, 2)] = heart_att
atts[(0, 0, 3)] = blood_att
atts[(0, 0, 4)] = heart_att
atts[(0, 0, 5)] = blood_att
atts[(0, 0, 6)] = heart_att
atts[(0, 0, 7)] = blood_att
atts[(0, 0, 8)] = heart_att
atts[(0, 0, 9)] = blood_att
atts[(0, 0, 10)] = heart_att
atts[(0, 0, 11)] = blood_att
atts[(0, 0, 12)] = heart_att
atts[(0, 0, 13)] = blood_att
atts[(0, 0, 14)] = heart_att
atts[(0, 0, 15)] = blood_att
atts[(0, 0, 16)] = heart_att
atts[(0, 0, 17)] = blood_att
atts[(0, 0, 18)] = heart_att
atts[(0, 0, 19)] = blood_att
atts[(0, 0, 20)] = heart_att
atts[(0, 0, 21)] = blood_att
atts[(0, 0, 22)] = heart_att
atts[(0, 0, 23)] = blood_att
atts[(0, 0, 24)] = heart_att
atts[(0, 0, 25)] = blood_att
atts[(0, 0, 26)] = heart_att

layers = []
layers.append(None)  # layer 0 (buffer layer, empty)

for zid in xrange(1, zs.size - 1):

    lengthx = xs[1] - xs[0]
    lengthy = ys[1] - ys[0]
    lengthz = zs[zid + 1] - zs[zid]
    box_center = np.array([0., 0., lengthz / 2. + zs[zid]])
    nscat = int(round(ns * lengthx * lengthy * lengthz))
    layers.append(np.c_[sp.rand(nscat) * lengthx,
                        sp.rand(nscat) * lengthy,
                        sp.rand(nscat) * lengthz] -
                  np.array([lengthx, lengthy, lengthz]) / 2. + box_center)

# SETUP TRANSDUCERS
# import transducer module and get tx/rx membrane and channel definitions
transducer_module = import_module(transducer_module_str)
transducer = transducer_module.create()

# set transducer environmental parameters
transducer.set_fluid_properties(sound_speed=sound_speed, density=density)

# SETUP BSC FILTERS
with h5py.File(bsc_filepath) as root:

    blood_bsc = root[blood_bsc_key][:]
    heart_bsc = root[heart_bsc_key][:]

#blood_bsc[-1,1] = 0
#heart_bsc[-1,1] = 0
blood_bsc = np.append(blood_bsc, [[50e6, 0]], axis=0)
heart_bsc = np.append(heart_bsc, [[50e6, 0]], axis=0)

prms = {}
prms['c'] = 1  #sound_speed
prms['rho'] = 1  #density
prms['area'] = rx_area
prms['ns'] = ns
prms['fs'] = 100e6
blood_bsc_fir = bsc_to_fir(blood_bsc, deriv=False, ntap=100, **prms)
heart_bsc_fir = bsc_to_fir(heart_bsc, deriv=False, ntap=100, **prms)

blood_bsc_fir = resampler(blood_bsc_fir, 100e6, sample_frequency)
heart_bsc_fir = resampler(heart_bsc_fir, 100e6, sample_frequency)

# root = loadmat(disp_filepath)
with np.load(path_to_excitation_file) as root:

    # acc = root['a'].squeeze()*1e9*1e9*density
    pulse = root['a']
    pulse_fs = root['fs']
    pulse_t = root['t']

    # resample pulse if sampling frequency is different
    if np.abs(pulse_fs - sample_frequency) > np.finfo(float).eps:

        acc = resampler(pulse, pulse_fs, sample_frequency)


def run_simulation():

    field = MField()

    field.field_init()

    field.set_field('c', sound_speed)
    field.set_field('fs', sample_frequency)
    field.set_field('att', attenuation)
    field.set_field('freq_att', frequency_attenuation)
    field.set_field('att_f0', attenuation_center_frequency)
    field.set_field('use_att', use_attenuation)

    tx_rect, tx_cc, tx_delays, tx_apod = transducer.generate_fieldii('tx')

    tx = field.xdc_rectangles(tx_rect, tx_cc, np.array([[0, 0, 300]]))
    field.xdc_focus_times(tx, np.zeros((1, 1)), tx_delays)
    field.xdc_apodization(tx, np.zeros((1, 1)), tx_apod)

    rx_rect, rx_cc, rx_delays, rx_apod = transducer.generate_fieldii('rx')

    rx = field.xdc_rectangles(rx_rect, rx_cc, np.array([[0, 0, 300]]))
    field.xdc_focus_times(rx, np.zeros((1, 1)), rx_delays)
    field.xdc_apodization(rx, np.zeros((1, 1)), rx_apod)

    center = np.array([0, 0, 0])
    fs = sample_frequency

    p_total = []

    for lid, l in enumerate(layers):

        if l is None: continue

        p = np.zeros(200000, dtype=np.float64)

        for idx, pt in tqdm(enumerate(l), total=len(l)):

            tx_sir, tx_t0 = field.calc_h(tx, pt)
            rx_sir, rx_t0 = field.calc_h(rx, pt)

            tx_sir = np.pad(tx_sir * fs, (int(round(tx_t0 * fs)), 0),
                            'constant')
            rx_sir = np.pad(rx_sir * fs, (int(round(rx_t0 * fs)), 0),
                            'constant')

            att = calc_path_att(center, pt, xs, ys, zs, atts)

            p0 = np.convolve(density * np.convolve(acc, tx_sir) / fs,
                             rx_sir) / fs * (att**2)
            p[:p0.size] += p0

        if lid % 2 == 0:
            filt = heart_bsc_fir
        else:
            filt = blood_bsc_fir

        p = np.convolve(p, filt)
        p_total.append(p)

    field.field_end()
    field.close()

    return p_total


if __name__ == '__main__':

    results = run_simulation()

    p_layers = np.array(results)
    p = np.sum(p_layers, axis=0)
    t = np.arange(len(p)) / sample_frequency
    z = sound_speed * t / 2

    np.savez(
        'd:/data/tissue simulation/simulation/heart_blood_penetration_phantom_data.npz',
        p=p,
        t=t,
        z=z,
        p_layers=p_layers,
        *results)
