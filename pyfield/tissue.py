'''
'''
import numpy as np
import scipy as sp
import scipy.signal
from matplotlib import pyplot as plt
from util import distance
from operator import itemgetter


def bsc_to_fir(freqs, bsc, c, rho, area, ns, fs, deriv=True, ntap=100):
    '''
    '''
    assert len(freqs) == len(bsc)

    # check first frequency bin is zero
    if not np.isclose(freqs[0], 0):
        freqs = np.insert(freqs, 0, 0)
        bsc = np.insert(bsc, 0, 0)

    # check last frequency bin is nyquist
    if not np.isclose(freqs[-1], fs / 2):
        freqs = np.append(freqs, fs / 2)
        bsc = np.append(bsc, 0)

    freq_resp = 2 * np.pi / (rho * c * area * np.sqrt(ns)) * np.sqrt(np.abs(bsc))
    imp_resp = sp.signal.firwin2(ntap, freqs, freq_resp, nyq=(fs / 2), 
        antisymmetric=False, window='hamming')
    
    if deriv:
        return np.gradient(imp_resp, 1 / fs)    
    return imp_resp
    

def xdc_get_area(file_path):
    '''
    '''
    with np.load(file_path) as varz:
        info = varz['info']
        widths = info[2,:]
        heights = info[3,:]

    area = np.sum(widths * heights)
    
    return area


def calc_path_att(r0, r1, xs, ys, zs, att, info=False):
    '''
    '''
    eps = np.finfo(np.float64).eps
    Dx, Dy, Dz = (r1 - r0).astype(np.float64)
    x0, y0, z0 = r0.astype(np.float64)
    x1, y1, z1 = r1.astype(np.float64)
    
    xflag, yflag, zflag = False, False, False
    
    if abs(Dx) < eps: xflag = True 
    if abs(Dy) < eps: yflag = True
    if abs(Dz) < eps: zflag = True
    
    if xflag and yflag: # line parallel to z-axis
        def fz(z):
            x = x0
            y = y0
            return x, y, z
        
        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))
        
        zmin = min(z0, z1)
        zmax = max(z0, z1)

        for z in zs:
            if z < zmin: continue
            if z > zmax: continue
            points.append(fz(z))
        
        points = list(set(points)) 
        points.sort(key=itemgetter(2), reverse=(z1 < z0))
        points = np.array(points)
    elif yflag and zflag: # line parallel to x-axis
        def fx(x):
            y = y0
            z = z0
            return x, y, z
        
        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))
        
        xmin = min(x0, x1)
        xmax = max(x0, x1)

        for x in xs:
            if x < xmin: continue
            if x > xmax: continue
            points.append(fx(x))
        
        points = list(set(points)) 
        points.sort(key=itemgetter(0), reverse=(x1 < x0))
        points = np.array(points)
    elif xflag and zflag: # line parallel to y-axis
        def fy(y):
            x = x0
            z = z0
            return x, y, z
        
        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))
        
        ymin = min(y0, y1)
        ymax = max(y0, y1)

        for y in ys:
            if y < ymin: continue
            if y > ymax: continue
            points.append(fy(y))
        
        points = list(set(points)) 
        points.sort(key=itemgetter(1), reverse=(y1 < y0))
        points = np.array(points)
    elif xflag: # line on y-z plane
        dzdy = Dz/Dy
        def fy(y):
            x = x0
            z = dzdy*(y - y0) + z0
            return x, y, z
        def fz(z):
            x = x0
            y = (z - z0)/dzdy + y0
            return x, y, z
        
        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))
        
        ymin = min(y0, y1)
        zmin = min(z0, z1)
        ymax = max(y0, y1)
        zmax = max(z0, z1)
        
        for y in ys:
            if y < ymin: continue
            if y > ymax: continue
            points.append(fy(y))
        
        for z in zs:
            if z < zmin: continue
            if z > zmax: continue
            points.append(fz(z))
        
        points = list(set(points)) 
        points.sort(key=itemgetter(1), reverse=(y1 < y0))
        points = np.array(points)
    elif yflag: # line on x-z plane
        dzdx = Dz/Dx
        def fx(x):
            y = y0
            z = dzdx*(x - x0) + z0
            return x, y, z
        def fz(z):
            x = (z - z0)/dzdx + x0
            y = y0
            return x, y, z
        
        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))
        
        xmin = min(x0, x1)
        zmin = min(z0, z1)
        xmax = max(x0, x1)
        zmax = max(z0, z1)
        
        for x in xs:
            if x < xmin: continue
            if x > xmax: continue
            points.append(fx(x))
                    
        for z in zs:
            if z < zmin: continue
            if z > zmax: continue
            points.append(fz(z))
        
        points = list(set(points)) 
        points.sort(key=itemgetter(0), reverse=(x1 < x0))
        points = np.array(points)
    elif zflag: # line on x-y plane
        dydx = Dy/Dx
        def fx(x):
            y = dydx*(x - x0) + y0
            z = z0
            return x, y, z
        
        def fy(y):
            x = (y - y0)/dydx + x0
            z = z0
            return x, y, z
        
        points = []
        points.append((x0, y0, z0))
        points.append((x1, y1, z1))
        
        xmin = min(x0, x1)
        ymin = min(y0, y1)
        xmax = max(x0, x1)
        ymax = max(y0, y1)
        
        for x in xs:
            if x < xmin: continue
            if x > xmax: continue
            points.append(fx(x))
        
        for y in ys:
            if y < ymin: continue
            if y > ymax: continue
            points.append(fy(y))
        
        points = list(set(points)) 
        points.sort(key=itemgetter(0), reverse=(x1 < x0))
        points = np.array(points)
    else: # line in quadrant
        dzdx = Dz/Dx
        dzdy = Dz/Dy
        dydx = Dy/Dx
        
        def fx(x):
            y = dydx*(x - x0) + y0
            z = dzdx*(x - x0) + z0
            return x, y, z
        def fy(y):
            x = (y - y0)/dydx + x0
            z = dzdy*(y - y0) + z0
            return x, y, z
        def fz(z):
            x = (z - z0)/dzdx + x0
            y = (z - z0)/dzdy + y0
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
        
        for x in xs:
            if x < xmin: continue
            if x > xmax: continue
            points.append(fx(x))
        
        for y in ys:
            if y < ymin: continue
            if y > ymax: continue
            points.append(fy(y))
        
        for z in zs:
            if z < zmin: continue
            if z > zmax: continue
            points.append(fz(z))
        
        points = list(set(points)) 
        points.sort(key=itemgetter(0), reverse=(x1 < x0))
        points = np.array(points)
    
    midpoints = (points[0:-1,:] + points[1:,:])/2.
    
    #box_ids = []
    path_lengths = np.zeros(midpoints.shape[0])
    att_coeffs = np.zeros(midpoints.shape[0])
    
    for idx, mp in enumerate(midpoints):
        x, y, z = mp
        path_lengths[idx] = (float(distance(points[idx,:], points[idx + 1,:])))
        
        xarg = np.where((x - xs) >= 0.0)[0][-1]
        if xarg > (xs.size - 2): xarg = xs.size - 2
        yarg = np.where((y - ys) >= 0.0)[0][-1]
        if yarg > (ys.size - 2): yarg = ys.size - 2
        zarg = np.where((z - zs) >= 0.0)[0][-1]
        if zarg > (zs.size - 2): zarg = zs.size - 2
        
        #box_ids.append((xarg, yarg, zarg))
        att_coeffs[idx] = att[(xarg, yarg, zarg)]
    
    att_total = np.product(np.exp(-att_coeffs*path_lengths))
    
    if info:
        return att_total, points, midpoints, path_lengths, att_coeffs
    else:
        return att_total


def draw_path_att(r0, r1, xs, ys, zs, att):
    '''
    '''
    info = calc_path_att(r0, r1, xs, ys, zs, att, info=True)
    
    points = info[1]
    midpoints = info[2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # draw grid/boxes
    x, y, z = np.meshgrid(xs, ys, zs)
    lw = 1
    ls = ':'
    for i in range(x.shape[0]):
        ax.plot_wireframe(x[i,:,:], y[i,:,:], z[i,:,:], linestyles=ls,
            linewidths=lw)
    for i in range(y.shape[1]):
        ax.plot_wireframe(x[:,i,:], y[:,i,:], z[:,i,:], linestyles=ls,
            linewidths=lw)
    for i in range(z.shape[2]):
        ax.plot_wireframe(x[:,:,i], y[:,:,i], z[:,:,i], linestyles=ls,
            linewidths=lw)
    
    # plot intersection points
    ax.plot(points[:,0], points[:,1], points[:,2], 'ro')
    
    # plot start and end points
    #ax.scatter(*r0, marker='x')
    #ax.scatter(*r1, marker='x')
    
    ax.plot([r0[0], r1[0]], [r0[1], r1[1]], [r0[2], r1[2]], 'g-')
        
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    fig.show()


def bsc_human_blood():
    '''
    Backscattering coefficient spectrum in units of 1/(Sr*m) of human blood at 8% hematocrit.
    Source: K. K. Shung et. al. ...
    '''
    freqs = [0., 5e6, 7e6, 8.5e6, 10e6, 15e6]
    bsc = [0., 0.001, 0.003004, 0.005464, 0.008702, 0.06394]

    return np.array(freqs), np.array(bsc)


def bsc_canine_heart():
    '''
    Backscattering coefficient spectrum in units of 1/(Sr*m) of canine myocardium.
    Source: O'Donnel et. al. ...
    '''
    freqs = [0., 2e6, 3e6, 4e6, 5e6, 6e6, 7e7, 8e6, 9e6, 10e6]
    bsc = [0., 2.4e-2, 1e-1, 3e-1, 5.4e-1, 9e-1, 1.4, 2.0, 3.0, 4.0]
    
    return np.array(freqs), np.array(bsc)


def blood_penetration_phantom(length_x=0.01, length_y=0.01, zmax=0.15, ns=5e6):
    '''
    '''
    att_blood = 0.14 * 100 # 0.14 Np/cm which is about 1.25 dB/cm
    att_heart = 0.58 * 100 # 0.58 Np/cm which is about 5 dB/cm
    layer_thickness = 0.002

    # define phantom geometry
    xs = np.array([-length_x / 2, length_x / 2])
    ys = np.array([-length_x / 2, length_x / 2])
    zs = np.array([0.0, 0.001])
    # add reflecting layer (heart tissue) every cm starting from 3 cm
    for i in np.arange(0.03, zmax + 0.01 / 2, 0.01):
        zs = zs.append([i, i + layer_thickness])

    # set attenuation properties of each layer
    atts = {}
    atts[(0, 0, 0)] = 0.0
    atts[(0, 0, 1)] = att_blood
    for i in range(2, len(zs) - 1):
        if i % 2 == 0:
            atts[(0, 0, i)] = att_heart
        else:
            atts[(0, 0, i)] = att_blood
        
    # construct phantom scatterers
    layers = []
    layers.append(None) # layer 0 is an empty buffer layer

    for zid in range(1, len(zs) - 1):
        length_x = xs[1] - xs[0]
        length_y = ys[1] - ys[0]
        length_z = zs[zid + 1] - zs[zid]
        box_center = np.array([0., 0., length_z / 2 + zs[zid]])
        nscat = round(ns * length_x * length_y * length_z)
        layers.append(np.c_[sp.rand(nscat) * length_x, sp.rand(nscat) * length_y, 
            sp.rand(nscat) * length_z] - np.array([length_x, length_y, length_z]) / 2 + box_center)


def heart_penetration_phantom(length_x=0.01, length_y=0.01, zmax=0.15, ns=5e6):
    '''
    '''
    att_blood = 0.14 * 100 # 0.14 Np/cm which is about 1.25 dB/cm
    att_heart = 0.58 * 100 # 0.58 Np/cm which is about 5 dB/cm
    layer_thickness = 0.002

    # define phantom geometry
    xs = np.array([-length_x / 2, length_x / 2])
    ys = np.array([-length_x / 2, length_x / 2])
    zs = np.array([0.0, 0.001, 0.03, 0.04])
    # add reflecting layer (heart tissue) every cm starting from 5 cm
    for i in np.arange(0.05, zmax + 0.01 / 2, 0.01):
        zs = zs.append([i, i + layer_thickness])

    # set attenuation properties of each layer
    atts = {}
    atts[(0, 0, 0)] = 0.0
    atts[(0, 0, 1)] = att_blood
    atts[(0, 0, 2)] = att_heart
    for i in range(3, len(zs) - 1):
        if i % 2 == 0:
            atts[(0, 0, i)] = att_heart
        else:
            atts[(0, 0, i)] = att_blood

    # construct phantom scatterers
    layers = []
    layers.append(None) # layer 0 is an empty buffer layer
    
    for zid in range(1, len(zs) - 1):
        length_x = xs[1] - xs[0]
        length_y = ys[1] - ys[0]
        length_z = zs[zid + 1] - zs[zid]
        box_center = np.array([0., 0., length_z / 2 + zs[zid]])
        nscat = round(ns * length_x * length_y * length_z)
        layers.append(np.c_[sp.rand(nscat) * length_x, sp.rand(nscat) * length_y, 
            sp.rand(nscat) * length_z] - np.array([length_x, length_y, length_z]) / 2 + box_center)


if __name__ == '__main__':
    
    pass
    #from itertools import product
    #
    #xs = np.arange(0, 5, dtype=np.float64)
    #ys = np.arange(0, 5, dtype=np.float64)
    #zs = np.arange(0, 5, dtype=np.float64)
    #
    #att = {}
    #for id in product(xrange(5), xrange(5), xrange(5)):
    #    att[id] = 2e-5
    #    
    #r0 = np.array([0,0,0])
    #rx = np.array([4,0,0])
    #ry = np.array([0,4,0])
    #rz = np.array([0,0,4])
    #rxy = np.array([4,4,0])
    #rxz = np.array([4,0,0])
    #ryz = np.array([0,4,4])
    #rdiag = np.array([4,4,4])
    #rxyz = np.array([3.5,2.5,3.5])
    #
    ##att1 = calc_path_att(r0, rx, xs, ys, zs, att)
    ##att2 = calc_path_att(r1, r0, xs, ys, zs, att)
    #
    #draw_path_att(r0, rx, xs, ys, zs, att)
    #draw_path_att(r0, ry, xs, ys, zs, att)
    #draw_path_att(r0, rz, xs, ys, zs, att)
    #draw_path_att(r0, rxy, xs, ys, zs, att)
    #draw_path_att(r0, rxz, xs, ys, zs, att)
    #draw_path_att(r0, ryz, xs, ys, zs, att)
    #draw_path_att(r0, rdiag, xs, ys, zs, att)
    #draw_path_att(r0, rxyz, xs, ys, zs, att)
        
        

