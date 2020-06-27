# pyfield / core / pyfield.py
'''
Implements Field II using the MATLAB Engine API for Python
'''
import numpy as np
import matlab
import matlab.engine
import time
import io
import os


class PyField(object):
    '''
    '''
    def __init__(self, path=None, quiet=False):

        # set default path to location of m-files (where this module is)
        if path is None:
            path = os.path.dirname(os.path.abspath(__file__))

        # try, for at most 30s, to start MATLAB engine
        success = False
        for i in range(6):
            try:
                self._mateng = matlab.engine.start_matlab()
                success = True
                break
            except (matlab.engine.EngineError, TypeError):
                time.sleep(5)

        if not success:
            raise matlab.engine.EngineError

        # set MATLAB engine path to location of m-files
        self._mateng.cd(str(os.path.normpath(path)), nargout=0)

    def __del__(self):

        # self.field_end() # end FIELD II
        self._mateng.quit() # shutdown MATLAB engine

    def _numpy_to_mat(self, array, orient='row'):

        if array.ndim == 1:
            if orient.lower() == 'row':
                sz = (1, array.size)
            elif orient.lower() in ('col', 'column'):
                sz = (array.size, 1)
        else:
            sz = None

        ret = matlab.double(initializer=array.tolist(), size=sz)
        return ret

    def _mat_to_numpy(self, array):
        return np.array(array).squeeze()


    ## FIELD FUNCTIONS ##

    def field_init(self, suppress=-1):
        self._mateng.field_init(suppress, nargout=0)

    def field_end(self):
        self._mateng.field_end(nargout=0)

    def set_field(self, option_name, value):
        self._mateng.set_field(option_name, value, nargout=0)

    def field_info(self):
        self._mateng.field_info(nargout=0)

    def field_debug(self):
        raise NotImplementedError

    def field_guide(self):
        raise NotImplementedError

    def field_logo(self):
        raise NotImplementedError

    def set_sampling(self):
        raise NotImplementedError


    ## CALC FUNCTIONS ##

    def calc_scat(self, Th1, Th2, points, amplitudes):

        points_mat = self._numpy_to_mat(points, orient='row')
        amplitudes_mat = self._numpy_to_mat(amplitudes, orient='col')

        ret = self._mateng.calc_scat(Th1, Th2, points_mat, amplitudes_mat,
            nargout=2)

        scat = self._mat_to_numpy(ret[0])
        t0 = ret[1]

        return scat, t0

    def calc_scat_all(self, Th1, Th2, points, amplitudes, dec_factor):

        points_mat = self._numpy_to_mat(points, orient='row')
        amplitudes_mat = self._numpy_to_mat(amplitudes, orient='col')

        ret = self._mateng.calc_scat_all(Th1, Th2, points_mat, amplitudes_mat,
            dec_factor, nargout=2)

        scat = self._mat_to_numpy(ret[0])
        t0 = ret[1]

        return scat, t0

    def calc_scat_multi(self, Th1, Th2, points, amplitudes):

        points_mat = self._numpy_to_mat(points, orient='row')
        amplitudes_mat = self._numpy_to_mat(amplitudes, orient='col')

        ret = self._mateng.calc_scat_multi(Th1, Th2, points_mat, amplitudes_mat,
            nargout=2)

        scat = self._mat_to_numpy(ret[0])
        t0 = ret[1]

        return scat, t0

    def calc_h(self, Th, points):

        points_mat = self._numpy_to_mat(points, orient='row')

        ret = self._mateng.calc_h(Th, points_mat, nargout=2)

        h = self._mat_to_numpy(ret[0])
        t0 = ret[1]

        return h, t0

    def calc_hp(self, Th, points):

        points_mat = self._numpy_to_mat(points, orient='row')

        ret = self._mateng.calc_hp(Th, points_mat, nargout=2)

        hp = self._mat_to_numpy(ret[0])
        t0 = ret[1]

        return hp, t0

    def calc_hhp(self, Th1, Th2, points):

        points_mat = self._numpy_to_mat(points, orient='row')

        ret = self._mateng.calc_hhp(Th1, Th2, points_mat, nargout=2)

        hhp = self._mat_to_numpy(ret[0])
        t0 = ret[1]

        return hhp, t0


    ## XDC FUNCTIONS ##

    def xdc_impulse(self, Th, pulse):

        pulse_mat = self._numpy_to_mat(pulse, orient='row')
        self._mateng.xdc_impulse(Th, pulse_mat, nargout=0)

    def xdc_excitation(self, Th, pulse):

        pulse_mat = self._numpy_to_mat(pulse, orient='row')
        self._mateng.xdc_excitation(Th, pulse_mat, nargout=0)

    def xdc_linear_array(self, no_elements, width, height, kerf, no_sub_x,
        no_sub_y, focus):

        focus_mat = self._numpy_to_mat(focus, orient='row')
        ret = self._mateng.xdc_linear_array(no_elements, width, height, kerf,
            no_sub_x, no_sub_y, focus_mat, nargout=1)

        return ret

    def xdc_show(self, Th, info_type='all'):
        self._mateng.xdc_show(Th, info_type, nargout=0)
    
    def xdc_focus(self, Th, times, points):

        times_mat = self._numpy_to_mat(times, orient='col')
        points_mat = self._numpy_to_mat(points, orient='row')

        self._mateng.xdc_focus(Th, times_mat, points_mat, nargout=0)   

    def xdc_focus_times(self, Th, times, delays):

        times_mat = self._numpy_to_mat(times, orient='col')
        delays_mat = self._numpy_to_mat(delays, orient='row')

        self._mateng.xdc_focus_times(Th, times_mat, delays_mat, nargout=0)

    def xdc_free(self, Th):
        self._mateng.xdc_free(Th, nargout=0)

    def xdc_get(self, Th, info_type='rect'):

        ret = self._mat_to_numpy(self._mateng.xdc_get(Th, info_type, nargout=1))
        return ret

    def xdc_rectangles(self, rect, center, focus):

        rect_mat = self._numpy_to_mat(rect, orient='row')
        center_mat = self._numpy_to_mat(center, orient='row')
        focus_mat = self._numpy_to_mat(focus, orient='row')

        ret = self._mateng.xdc_rectangles(rect_mat, center_mat, focus_mat,
            nargout=1)

        return ret

    def xdc_focused_array(self, no_elements, width, height, kerf, rfocus, no_sub_x, no_sub_y, focus):

        focus_mat = self._numpy_to_mat(focus, orient='row')

        ret = self._mateng.xdc_focused_array(no_elements, width, height, kerf,
            rfocus, no_sub_x, no_sub_y, focus_mat, nargout=1)

        return ret
    
    def xdc_piston(self, radius, ele_size):
        
        ret = self._mateng.xdc_piston(radius, ele_size)
        
        return ret

    def xdc_apodization(self, Th, times, values):

        times_mat = self._numpy_to_mat(times, orient='col')
        values_mat = self._numpy_to_mat(values, orient='row')

        self._mateng.xdc_apodization(Th, times_mat, values_mat, nargout=0)

    def xdc_quantization(self, Th, value):
        self._mateng.xdc_quantization(Th, value, nargout=0)

    def xdc_2d_array(self):
        raise NotImplementedError

    def xdc_concave(self):
        raise NotImplementedError

    def xdc_convex_array(self):
        raise NotImplementedError

    def xdc_convex_focused_array(self):
        raise NotImplementedError

    def xdc_baffle(self):
        raise NotImplementedError

    def xdc_center_focus(self):
        raise NotImplementedError

    def xdc_convex_focused_multirow(self):
        raise NotImplementedError

    def xdc_focused_multirow(self):
        raise NotImplementedError

    def xdc_dynamic_focus(self):
        raise NotImplementedError

    def xdc_lines(self):
        raise NotImplementedError

    def xdc_piston(self):
        raise NotImplementedError

    def xdc_times_focus(self):
        raise NotImplementedError

    def xdc_triangles(self):
        raise NotImplementedError

    ## ELE FUNCTIONS ##

    def ele_apodization(self, Th, element_no, apo):

        element_no_mat = self._numpy_to_mat(element_no, orient='col')
        apo_mat = self._numpy_to_mat(apo, orient='row')

        self._mateng.ele_apodization(Th, element_no_mat, apo_mat, nargout=0)

    def ele_delay(self, Th, element_no, delays):

        element_no_mat = self._numpy_to_mat(element_no, orient='col')
        delays_mat = self._numpy_to_mat(delays, orient='row')

        self._mateng.ele_delay(Th, element_no_mat, delays_mat, nargout=0)

    def ele_waveform(self):
        raise NotImplementedError

## TEST ##

if __name__ == '__main__':

    # from scipy.signal import gausspulse
    from pyfield import util

    field = PyField()

    field.field_init()
    field.set_field('c', 1500)
    field.set_field('fs', 100e6)
    field.set_field('att', 0)
    field.set_field('freq_att', 10e6)
    field.set_field('att_f0', 0)
    field.set_field('use_att', 1)

    fc = 10e6
    fbw = 1.0
    fs = 100e6

    pulse, t = util.gausspulse(fc, fbw, fs)

    tx = field.xdc_linear_array(64, 0.0002, 0.001, 300e-6, 1, 2, np.array([0, 0, 0.03]))
    field.xdc_impulse(tx, pulse)
    field.xdc_excitation(tx, np.array([1]))

    field.field_info()
    # field.xdc_show(tx)

    scat, t0 = field.calc_scat_multi(tx, tx, np.array([0, 0, 0.03]), np.array([1]))

    field.field_end()
