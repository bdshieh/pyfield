 
import numpy as np
import scipy as sp
from scipy.io import loadmat

from pyfield import abstract, tissue, util
from pyfield.solvers import TransmitReceiveBeamplot


# calculation
def main(cfg, args):

    # SETUP TRANSDUCERS
    solver_cfg = TransmitBeamplot.Config()
    solver_cfg._update(cfg._asdict())
    arrays = abstract.load(cfg.array_config)

    if not isinstance(arrays, list):
        arrays = [arrays,]
        
    solver = TransmitBeamplot.from_abstract(solver_cfg, *arrays)
    field = solver.field
    # tx_transducer, tx_channels, tx_subs = transmitter.main()
    # rx_transducer, rx_channels, rx_subs = receiver.main()

    # SETUP DISPLACEMENT EXCITATION
    root = loadmat(disp_filepath)
    acc = root['a'].squeeze() * 1e9 * 1e9 * density

    # field = Field()
    # field.field_init()
    # field.set_field('c', cfg.sound_speed)
    # field.set_field('fs', cfg.sample_frequency)
    # field.set_field('att', cfg.attenuation)
    # field.set_field('freq_att', cfg.frequency_attenuation)
    # field.set_field('att_f0', cfg.attenuation_center_frequency)
    # field.set_field('use_att', cfg.use_attenuation)
    
    # tx_rect, tx_cc, tx_delays, tx_apod = convert_to_field(tx_transducer, tx_channels)
    # tx = field.xdc_rectangles(tx_rect, tx_cc, np.array([[0, 0, 300]]))
    # field.xdc_focus_times(tx, np.zeros((1, 1)), tx_delays)
    # field.xdc_apodization(tx, np.zeros((1, 1)), tx_apod)
    
    # rx_rect, rx_cc, rx_delays, rx_apod = convert_to_field(rx_transducer, rx_channels)
    # rx = field.xdc_rectangles(rx_rect, rx_cc, np.array([[0, 0, 300]]))
    # field.xdc_focus_times(rx, np.zeros((1, 1)), rx_delays)
    # field.xdc_apodization(rx, np.zeros((1, 1)), rx_apod)
    
    center = np.array([0,0,0])
    fs = sample_frequency
    
    p_total = []
    for lid, l in enumerate(layers):
        if l is None: continue
        
        p = np.zeros(20000, dtype=np.float64)
        for idx, pt in enumerate(l):
            print str(idx + 1) + ' / ' + str(l.shape[0])
            tx_sir, tx_t0 = field.calc_h(tx, pt)
            rx_sir, rx_t0 = field.calc_h(rx, pt)
            
            tx_sir = np.pad(tx_sir * fs, (round(tx_t0 * fs), 0), 'constant')
            rx_sir = np.pad(rx_sir * fs, (round(rx_t0 * fs), 0), 'constant')
            
            att = calc_path_att(center, pt, xs, ys, zs, atts)
            
            p0 = np.convolve(np.convolve(acc, tx_sir) / fs, rx_sir) / fs * (att**2)
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


# define default configuration for this script
_Config = {}
_Config['array_config'] = ''
_Config['sampling_freq'] = 200e6
_Config['center_freq'] = 7e6
_Config['bandwidth'] = 5.6e6
_Config['use_attenuation'] = False
_Config['attenuation'] = 0
_Config['freq_attenuation'] = 0
_Config['attenuation_center_freq'] = 7e6
_Config['sound_speed'] = 1500.
_Config['density'] = 1000.
_Config['rx_area'] = 35e-6 * 35e-6 * 4
Config = abstract.register_type('Config', _Config)

if __name__ == '__main__':

    from pyfield import util

    # get script parser and parse arguments
    parser, run_parser = util.script_parser(main, Config)
    args = parser.parse_args()
    args.func(args)


    



