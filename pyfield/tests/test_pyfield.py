'''
'''
import pytest
import numpy as np
from pyfield import PyField
from pyfield import util


@pytest.fixture
def field():
    obj = PyField()
    obj.field_init()
    yield obj
    obj.field_end()


@pytest.fixture
def linear_array(field):
    no_elements = 10
    width = 0.001
    height = 0.01
    kerf = 0.001 / 2
    no_sub_x = 1
    no_sub_y = 1
    focus = [0, 0, 0.01]
    yield field.xdc_linear_array(no_elements, width, height, kerf, no_sub_x,
                                 no_sub_y, focus)


@pytest.fixture
def pulse():
    fc = 5e6
    fbw = 1.0
    fs = 100e6
    pulse, t = util.gausspulse(fc, fbw, fs)
    return pulse


@pytest.fixture
def points():
    return [[0, 0, 0.01], [0, 0, 0.02], [0, 0, 0.03]]


def test_matlab():
    import matlab.engine
    assert matlab.engine.start_matlab() is not None


''' Field methods '''


def test_field_debug(field):
    assert field.field_debug(1) is None


def test_field_info(field):
    assert field.field_info() is None


def test_set_field(field):
    assert field.set_field('fs', 100e6) is None


def test_set_sampling(field):
    assert field.set_sampling(100e6) is None


''' Xdc methods '''


def test_xdc_2d_array(field):
    no_elem_x = 2
    no_elem_y = 2
    width = 0.001
    heights = 0.001
    kerf_x = 0.001 / 2
    kerf_y = 0.001 / 2
    enabled = np.ones((no_elem_x, no_elem_y))
    no_sub_x = 1
    no_sub_y = 1
    focus = [0, 0, 0.01]
    assert field.xdc_2d_array(no_elem_x, no_elem_y, width, heights, kerf_x,
                              kerf_y, enabled, no_sub_x, no_sub_y, focus) > -1


def test_xdc_apodization(field, linear_array):
    times = [0, 1]
    values = np.ones((2, 10))
    assert field.xdc_apodization(linear_array, times, values) is None


def test_xdc_baffle(field, linear_array):
    soft_baffle = 1
    assert field.xdc_baffle(linear_array, soft_baffle) is None


def test_xdc_center_focus(field, linear_array):
    point = [0, 0, 0.01]
    assert field.xdc_center_focus(linear_array, point) is None


def test_xdc_concave(field):
    radius = 0.05
    focal_radius = 0.01
    ele_size = 0.001
    assert field.xdc_concave(radius, focal_radius, ele_size) > -1


def test_xdc_convert(field, linear_array):
    assert field.xdc_convert(linear_array) is None


def test_xdc_convex_array(field):
    no_elements = 5
    width = 0.001
    height = 0.001
    kerf = 0.001 / 2
    Rconvex = 0.01
    no_sub_x = 1
    no_sub_y = 1
    focus = [0, 0, 0.01]
    assert field.xdc_convex_array(no_elements, width, height, kerf, Rconvex,
                                  no_sub_x, no_sub_y, focus) > -1


def test_xdc_convex_focused_array(field):
    no_elements = 5
    width = 0.001
    height = 0.001
    kerf = 0.001 / 2
    Rconvex = 0.01
    Rfocus = 0.01
    no_sub_x = 1
    no_sub_y = 2
    focus = [0, 0, 0.01]
    assert field.xdc_convex_focused_array(no_elements, width, height, kerf,
                                          Rconvex, Rfocus, no_sub_x, no_sub_y,
                                          focus) > -1


def test_xdc_convex_focused_multirow(field):
    no_elem_x = 5
    width = 0.001
    no_elem_y = 5
    heights = np.ones(5) * 0.001
    kerf_x = 0.001 / 2
    kerf_y = 0.001 / 2
    Rconvex = 0.01
    Rfocus = 0.01
    no_sub_x = 1
    no_sub_y = 1
    focus = [0, 0, 0.01]
    assert field.xdc_convex_focused_multirow(
        no_elem_x, width, no_elem_y, heights, kerf_x, kerf_y, Rconvex, Rfocus,
        no_sub_x, no_sub_y, focus) > -1


def test_xdc_excitation(field, linear_array, pulse):
    assert field.xdc_excitation(linear_array, pulse) is None


def test_xdc_focus(field, linear_array):
    times = [0, 1e-6, 2e-6]
    points = [[0, 0, 0.01], [0, 0, 0.02], [0, 0, 0.03]]
    assert field.xdc_focus(linear_array, times, points) is None


def test_xdc_focus_times(field, linear_array):
    times = [0, 1e-6, 2e-6]
    delays = np.zeros((3, 10))
    assert field.xdc_focus_times(linear_array, times, delays) is None


def test_xdc_focused_array(field):
    no_elements = 5
    width = 0.001
    height = 0.001
    kerf = 0.001 / 2
    rfocus = 0.01
    no_sub_x = 1
    no_sub_y = 2
    focus = [0, 0, 0.01]
    assert field.xdc_focused_array(no_elements, width, height, kerf, rfocus,
                                   no_sub_x, no_sub_y, focus) > -1


def test_xdc_focused_multirow(field):
    no_elem_x = 5
    width = 0.001
    no_elem_y = 5
    heights = np.ones(5) * 0.001
    kerf_x = 0.001 / 2
    kerf_y = 0.001 / 2
    Rfocus = 0.01
    no_sub_x = 1
    no_sub_y = 1
    focus = [0, 0, 0.01]
    assert field.xdc_focused_multirow(no_elem_x, width, no_elem_y, heights,
                                      kerf_x, kerf_y, Rfocus, no_sub_x,
                                      no_sub_y, focus) > -1


def test_xdc_free(field, linear_array):
    assert field.xdc_free(linear_array) is None


def test_xdc_get(field, linear_array):
    info_type = 'rect'
    rect = field.xdc_get(linear_array, info_type)
    assert rect.shape == (26, 10)


def test_xdc_impulse(field, linear_array, pulse):
    assert field.xdc_impulse(linear_array, pulse) is None


def test_xdc_line_convert(field, linear_array):
    assert field.xdc_line_convert(linear_array) is None


def test_xdc_linear_array(linear_array):
    assert linear_array > -1


def test_xdc_linear_multirow(field):
    no_elem_x = 5
    width = 0.001
    no_elem_y = 5
    heights = np.ones(5) * 0.001
    kerf_x = 0.001 / 2
    kerf_y = 0.001 / 2
    no_sub_x = 1
    no_sub_y = 1
    focus = [0, 0, 0.01]
    assert field.xdc_linear_multirow(no_elem_x, width, no_elem_y, heights,
                                     kerf_x, kerf_y, no_sub_x, no_sub_y,
                                     focus) > -1


def test_xdc_lines(field):
    lines = [[1, 1, np.nan, 1, 2 / 1000, 1], [1, 1, 0, 0, 5 / 1000, 0],
             [1, 1, np.nan, 1, -2 / 1000, 0], [1, 1, 0, 0, -5 / 1000, 1],
             [2, 1, np.nan, 1, 6.5 / 1000, 1], [2, 1, 0, 0, 5 / 1000, 0],
             [2, 1, np.nan, 1, 2.5 / 1000, 0], [2, 1, 0, 0, -5 / 1000, 1]]
    center = [[0, 0, 0], [4.5 / 1000, 0, 0]]
    focus = [0, 0, 70 / 1000]
    assert field.xdc_lines(lines, center, focus) > -1


def test_xdc_piston(field):
    radius = 0.005
    ele_size = 0.001
    assert field.xdc_piston(radius, ele_size) > -1


def test_xdc_quantization(field, linear_array):
    value = 1e-9
    assert field.xdc_quantization(linear_array, value) is None


def test_xdc_rectangles(field, linear_array):

    data = field.xdc_get(linear_array, 'rect')
    phys_no = data[0, :] + 1
    assert phys_no[0] == 1
    width = data[2, :]
    height = data[3, :]
    apo = data[4, :]
    corners = data[10:22, :]
    centers = data[23:26, :]
    rect = np.concatenate(np.atleast_2d(phys_no, corners, apo, width, height,
                                        centers),
                          axis=0)
    assert rect.shape == (19, 10)
    focus = [0, 0, 0.01]
    assert field.xdc_rectangles(rect.T, centers.T, focus) > -1


def test_xdc_show(field, linear_array):
    field.xdc_show(linear_array, info_type='all')


def test_xdc_times_focus(field, linear_array):
    times = [0, 1e-6, 2e-6]
    delays = np.zeros((3, 10))
    assert field.xdc_times_focus(linear_array, times, delays) is None


def test_xdc_triangles(field, linear_array):

    rect = field.xdc_get(linear_array, 'rect')
    centers = rect[23:26, :]
    field.xdc_convert(linear_array)
    data = field.xdc_get(linear_array, 'tri')
    phys_no = data[0, :] + 1
    # math_no = data[1, :] + 1
    apo = data[2, :]
    # math_centers = data[3:6, :]
    corners = data[6:15, :]
    tri = np.concatenate(np.atleast_2d(phys_no, corners, apo), axis=0)
    assert tri.shape == (11, 20)
    focus = [0, 0, 0.01]
    assert field.xdc_triangles(tri.T, centers.T, focus) > -1


''' Calc methods '''


def test_calc_scat(field, linear_array, points):
    amplitudes = np.ones_like(points)
    scat, t0 = field.calc_scat(linear_array, linear_array, points, amplitudes)
    assert scat.ndim == 1
    assert t0 > 0


def calc_scat_all(linear_array, points):
    dec_factor = 1
    scat, t0 = field.calc_scat_all(linear_array, linear_array, points,
                                   dec_factor)
    assert scat.shape[1] == 100
    assert t0 > 0


def calc_scat_multi(linear_array, points):
    amplitudes = np.ones_like(points)
    scat, t0 = field.calc_scat_multi(linear_array, linear_array, points,
                                     amplitudes)
    assert scat.shape[1] == 10
    assert t0 > 0


def calc_h(linear_array, points):
    scat, t0 = field.calc_h(linear_array, points)
    assert scat.ndim == 1
    assert t0 > 0


def calc_hp(linear_array, points):
    scat, t0 = field.calc_hp(linear_array, points)
    assert scat.ndim == 1
    assert t0 > 0


def calc_hhp(linear_array, points):
    scat, t0 = field.calc_hhp(linear_array, linear_array, points)
    assert scat.ndim == 1
    assert t0 > 0
