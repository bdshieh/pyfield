'''Implements Field II using the MATLAB Engine API for Python.'''
import numpy as np
import matlab
import matlab.engine
import time
import io
import os


class PyField(object):
    '''
    [summary]

    Parameters
    ----------
    object : [type]
        [description]
    '''
    def __init__(self, path=None, quiet=False):
        '''
        [summary]

        Parameters
        ----------
        path : [type], optional
            [description], by default None
        quiet : bool, optional
            [description], by default False

        Raises
        ------
        matlab.engine.EngineError
            [description]
        '''
        # set default path to location of m-files (where this module is)
        if path is None:
            path = os.path.dirname(os.path.abspath(__file__))

        if quiet:
            self.stdout = io.StringIO()
        else:
            self.stdout = None

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
        '''Shutdown MATLAB Engine'''
        # self.field_end() # end FIELD II
        self._mateng.quit()  # shutdown MATLAB engine

    def _numpy_to_mat(self, array, orient='row'):
        '''Convert ndarray to MATLAB array'''
        array = np.atleast_2d(array)
        assert array.ndim == 2

        if orient.lower() == 'row':
            if array.shape[1] == 1:
                array = array.T
        elif orient.lower() in ('col', 'column'):
            if array.shape[0] == 1:
                array = array.T

        return matlab.double(initializer=array.tolist(), size=array.shape)

    def _mat_to_numpy(self, array):
        '''Convert MATLAB array to ndarray'''
        return np.array(array).squeeze()

    '''Field functions '''

    def field_debug(self, state):
        self._mateng.field_debug(state, nargout=0)

    def field_end(self):
        self._mateng.field_end(nargout=0)

    def field_guide(self):
        self._mateng.field_guide(nargout=0)

    def field_info(self):
        self._mateng.field_info(nargout=0)

    def field_init(self, suppress=-1):
        self._mateng.field_init(suppress, nargout=0)

    def field_logo(self):
        self._mateng.field_logo(nargout=0)

    def set_field(self, option_name, value):
        self._mateng.set_field(option_name, value, nargout=0)

    def set_sampling(self, fs):
        self._mateng.set_sampling(fs, nargout=0)

    '''Calc functions'''

    def calc_scat(self, Th1, Th2, points, amplitudes):
        points_mat = self._numpy_to_mat(points, orient='row')
        amplitudes_mat = self._numpy_to_mat(amplitudes, orient='col')
        ret = self._mateng.calc_scat(Th1,
                                     Th2,
                                     points_mat,
                                     amplitudes_mat,
                                     nargout=2)

        scat = self._mat_to_numpy(ret[0])
        t0 = ret[1]

        return scat, t0

    def calc_scat_all(self, Th1, Th2, points, amplitudes, dec_factor):
        points_mat = self._numpy_to_mat(points, orient='row')
        amplitudes_mat = self._numpy_to_mat(amplitudes, orient='col')
        ret = self._mateng.calc_scat_all(Th1,
                                         Th2,
                                         points_mat,
                                         amplitudes_mat,
                                         dec_factor,
                                         nargout=2)

        scat = self._mat_to_numpy(ret[0])
        t0 = ret[1]
        return scat, t0

    def calc_scat_multi(self, Th1, Th2, points, amplitudes):
        points_mat = self._numpy_to_mat(points, orient='row')
        amplitudes_mat = self._numpy_to_mat(amplitudes, orient='col')
        ret = self._mateng.calc_scat_multi(Th1,
                                           Th2,
                                           points_mat,
                                           amplitudes_mat,
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

    '''Xdc functions'''

    def xdc_2d_array(self, no_elem_x, no_elem_y, width, heights, kerf_x,
                     kerf_y, enabled, no_sub_x, no_sub_y, focus):
        enabled_mat = self._numpy_to_mat(enabled, orient='row')
        focus_mat = self._numpy_to_mat(focus, orient='row')
        return self._mateng.xdc_2d_array(no_elem_x,
                                         no_elem_y,
                                         width,
                                         heights,
                                         kerf_x,
                                         kerf_y,
                                         enabled_mat,
                                         no_sub_x,
                                         no_sub_y,
                                         focus_mat,
                                         nargout=1)

    def xdc_apodization(self, Th, times, values):
        times_mat = self._numpy_to_mat(times, orient='col')
        values_mat = self._numpy_to_mat(values, orient='row')
        self._mateng.xdc_apodization(Th, times_mat, values_mat, nargout=0)

    def xdc_baffle(self, Th, soft_baffle):
        self._mateng.xdc_baffle(Th, soft_baffle, nargout=0)

    def xdc_center_focus(self, Th, point):
        point_mat = self._numpy_to_mat(point, orient='row')
        self._mateng.xdc_center_focus(Th, point_mat, nargout=0)

    def xdc_concave(self, radius, focal_radius, ele_size):
        return self._mateng.xdc_concave(radius,
                                        focal_radius,
                                        ele_size,
                                        nargout=1)

    def xdc_convert(self, Th):
        self._mateng.xdc_convert(Th, nargout=0)

    def xdc_convex_array(self, no_elements, width, height, kerf, Rconvex,
                         no_sub_x, no_sub_y, focus):
        focus_mat = self._numpy_to_mat(focus, orient='row')
        return self._mateng.xdc_convex_array(no_elements,
                                             width,
                                             height,
                                             kerf,
                                             Rconvex,
                                             no_sub_x,
                                             no_sub_y,
                                             focus_mat,
                                             nargout=1)

    def xdc_convex_focused_array(self, no_elements, width, height, kerf,
                                 Rconvex, Rfocus, no_sub_x, no_sub_y, focus):
        focus_mat = self._numpy_to_mat(focus, orient='row')
        return self._mateng.xdc_convex_focused_array(no_elements,
                                                     width,
                                                     height,
                                                     kerf,
                                                     Rconvex,
                                                     Rfocus,
                                                     no_sub_x,
                                                     no_sub_y,
                                                     focus_mat,
                                                     nargout=1)

    def xdc_convex_focused_multirow(self, no_elem_x, width, no_elem_y, heights,
                                    kerf_x, kerf_y, Rconvex, Rfocus, no_sub_x,
                                    no_sub_y, focus):
        heights_mat = self._numpy_to_mat(heights, orient='row')
        focus_mat = self._numpy_to_mat(focus, orient='row')
        return self._mateng.xdc_convex_focused_multirow(no_elem_x,
                                                        width,
                                                        no_elem_y,
                                                        heights_mat,
                                                        kerf_x,
                                                        kerf_y,
                                                        Rconvex,
                                                        Rfocus,
                                                        no_sub_x,
                                                        no_sub_y,
                                                        focus_mat,
                                                        nargout=1)

    def xdc_dynamic_focus(self, Th, time, dir_zx, dir_zy):
        self._mateng.xdc_dynamic_focus(Th, time, dir_zx, dir_zy, nargout=0)

    def xdc_excitation(self, Th, pulse):
        pulse_mat = self._numpy_to_mat(pulse, orient='row')
        self._mateng.xdc_excitation(Th, pulse_mat, nargout=0)

    def xdc_focus(self, Th, times, points):
        times_mat = self._numpy_to_mat(times, orient='col')
        points_mat = self._numpy_to_mat(points, orient='row')
        self._mateng.xdc_focus(Th, times_mat, points_mat, nargout=0)

    def xdc_focus_times(self, Th, times, delays):
        times_mat = self._numpy_to_mat(times, orient='col')
        delays_mat = self._numpy_to_mat(delays, orient='row')
        self._mateng.xdc_focus_times(Th, times_mat, delays_mat, nargout=0)

    def xdc_focused_array(self, no_elements, width, height, kerf, rfocus,
                          no_sub_x, no_sub_y, focus):
        focus_mat = self._numpy_to_mat(focus, orient='row')
        return self._mateng.xdc_focused_array(no_elements,
                                              width,
                                              height,
                                              kerf,
                                              rfocus,
                                              no_sub_x,
                                              no_sub_y,
                                              focus_mat,
                                              nargout=1)

    def xdc_focused_multirow(self, no_elem_x, width, no_elem_y, heights,
                             kerf_x, kerf_y, Rfocus, no_sub_x, no_sub_y,
                             focus):
        heights_mat = self._numpy_to_mat(heights, orient='row')
        focus_mat = self._numpy_to_mat(focus, orient='row')
        return self._mateng.xdc_focused_multirow(no_elem_x,
                                                 width,
                                                 no_elem_y,
                                                 heights_mat,
                                                 kerf_x,
                                                 kerf_y,
                                                 Rfocus,
                                                 no_sub_x,
                                                 no_sub_y,
                                                 focus_mat,
                                                 nargout=1)

    def xdc_free(self, Th):
        self._mateng.xdc_free(Th, nargout=0)

    def xdc_get(self, Th, info_type='rect'):
        return self._mat_to_numpy(
            self._mateng.xdc_get(Th, info_type, nargout=1))

    def xdc_impulse(self, Th, pulse):
        pulse_mat = self._numpy_to_mat(pulse, orient='row')
        self._mateng.xdc_impulse(Th, pulse_mat, nargout=0)

    def xdc_line_convert(self, Th):
        self._mateng.xdc_line_convert(Th, nargout=0)

    def xdc_linear_array(self, no_elements, width, height, kerf, no_sub_x,
                         no_sub_y, focus):

        focus_mat = self._numpy_to_mat(focus, orient='row')
        return self._mateng.xdc_linear_array(no_elements,
                                             width,
                                             height,
                                             kerf,
                                             no_sub_x,
                                             no_sub_y,
                                             focus_mat,
                                             nargout=1)

    def xdc_linear_multirow(self, no_elem_x, width, no_elem_y, heights, kerf_x,
                            kerf_y, no_sub_x, no_sub_y, focus):
        heights_mat = self._numpy_to_mat(heights, orient='row')
        focus_mat = self._numpy_to_mat(focus, orient='row')
        return self._mateng.xdc_linear_multirow(no_elem_x,
                                                width,
                                                no_elem_y,
                                                heights_mat,
                                                kerf_x,
                                                kerf_y,
                                                no_sub_x,
                                                no_sub_y,
                                                focus_mat,
                                                nargout=1)

    def xdc_lines(self, lines, center, focus):
        lines_mat = self._numpy_to_mat(lines, orient='row')
        center_mat = self._numpy_to_mat(center, orient='row')
        focus_mat = self._numpy_to_mat(focus, orient='row')
        return self._mateng.xdc_lines(lines_mat,
                                      center_mat,
                                      focus_mat,
                                      nargout=1)

    def xdc_piston(self, radius, ele_size):
        return self._mateng.xdc_piston(radius, ele_size, nargout=1)

    def xdc_quantization(self, Th, value):
        self._mateng.xdc_quantization(Th, value, nargout=0)

    def xdc_rectangles(self, rect, center, focus):
        rect_mat = self._numpy_to_mat(rect, orient='row')
        center_mat = self._numpy_to_mat(center, orient='row')
        focus_mat = self._numpy_to_mat(focus, orient='row')
        return self._mateng.xdc_rectangles(rect_mat,
                                           center_mat,
                                           focus_mat,
                                           nargout=1)

    def xdc_show(self, Th, info_type='all'):
        self._mateng.xdc_show(Th, info_type, nargout=0)

    def xdc_times_focus(self, Th, times, delays):
        times_mat = self._numpy_to_mat(times, orient='col')
        delays_mat = self._numpy_to_mat(delays, orient='row')
        self._mateng.xdc_times_focus(Th, times_mat, delays_mat, nargout=0)

    def xdc_triangles(self, data, center, focus):
        data_mat = self._numpy_to_mat(data, orient='row')
        center_mat = self._numpy_to_mat(center, orient='row')
        focus_mat = self._numpy_to_mat(focus, orient='row')
        return self._mateng.xdc_triangles(data_mat,
                                          center_mat,
                                          focus_mat,
                                          nargout=1)

    '''Ele functions'''

    def ele_apodization(self, Th, element_no, apo):
        element_no_mat = self._numpy_to_mat(element_no, orient='col')
        apo_mat = self._numpy_to_mat(apo, orient='row')
        self._mateng.ele_apodization(Th, element_no_mat, apo_mat, nargout=0)

    def ele_delay(self, Th, element_no, delays):
        element_no_mat = self._numpy_to_mat(element_no, orient='col')
        delays_mat = self._numpy_to_mat(delays, orient='row')
        self._mateng.ele_delay(Th, element_no_mat, delays_mat, nargout=0)

    def ele_waveform(self, Th, element_no, samples):
        element_no_mat = self._numpy_to_mat(element_no, orient='col')
        samples_mat = self._numpy_to_mat(samples, orient='row')
        self._mateng.ele_waveform(Th, element_no_mat, samples_mat, nargout=0)


if __name__ == '__main__':
    pass
