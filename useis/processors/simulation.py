#!/usr/bin/env python
# -*- coding: utf-8; -*-
#
# (c) 2016 microquake development team
#
# This file is part of the microquake library
#
# microquake is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# microquake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with microquake.  If not, see <http://www.gnu.org/licenses/>.

# Test of the performance simulation module

# NOTE: The default spark default configuration located in
# $SPARK_HOME/conf/spark-default.conf file must be edited to increase the
# default driver memory to avoid running out of heap space.

# TODO:
# - The script works with heterogeneous velocity model but does not
# currently supports heterogeneous Q.

from ..core.project_manager import ProjectManager


class Simulation(ProjectManager):
    
    def __init__(self, base_projects_path, project_name, network_code,
                 use_srces=True, **kwargs):

        super.__init__(base_projects_path, project_name, network_code,
                       use_srcs=use_srces, **kwargs)
    
    self.files.simulation_settings = self.paths.config / \
                                     'simulation_settings.py'

    sys.path.append(str(self.paths.config))

    if not self.files.simulation_settings.is_file():
        settings_template = Path(os.path.realpath(__file__)).parent / \
                            '../settings/simulation_settings_template.py'

        shutil.copyfile(settings_template, self.files.simulation_settings)

        self.simulation_settings = __import__('simulation_settings')

        self.settings = Settings(str(self.paths.config))
        
    def simulate_magnitude_completeness(self):
        min_num_site = self.settings.simulation.min_num_site
        gmpe = self.settings.simulation.gmpe


from microquake.core import ctl, logger
from microquake.waveform.mag import synthetic_seismogram
import numpy as np
import argparse
import os
import shutil
from microquake.spark import mq_map
from scipy.interpolate import interp2d
from microquake.simul.eik import eikonal_solver, ray_tracer
from IPython.core.debugger import Tracer

import sys  # put at top if not already there


def init_io_directories(params):
    vmodel = params.velgrids
    for scenario in params.sensors:
        dname = io_directory(params)
        if not os.path.exists(dname):
            os.makedirs(dname)

    return


def io_directory(params):
    output_directory = params.simulation.output_directory
    return '%s/scenario_%0d__models_%0d' % (output_directory,
                                            params.sensors.index,
                                            params.velgrids.index)


def detection_threshold(ev_tp, station, stalta_th, noise, vp, vs, ttp, tts, rho,
                        homogeneous, triaxial, orientation, magnitudes, sta,
                        ssd, quality, a_s_att_fun):
    """

    :param ev_tp:
    :param station:
    :param stalta_th:
    :param noise:
    :param vp:
    :param vs:
    :param rho:
    :param homogeneous:
    :param triaxial:
    :param orientation:
    :param magnitudes:
    :param sta:
    :param ssd:
    :param quality:
    :param acceleration:
    :return:
    """
    event_id = ev_tp[0]
    event = ev_tp[1]

    if homogeneous:
        dist = np.linalg.norm(station - event)
    else:
        ray_p = ray_tracer(ttp, event)
        ray_s = ray_tracer(tts, event)

        dist_p = 0
        for k in range(0, len(ray_p) - 1):
            dist_p += np.linalg.norm(ray_p[k, :] - ray_p[k+1, :])

        dist_s = 0
        for k in range(0, len(ray_p) - 1):
            dist_s += np.linalg.norm(ray_p[k, :] - ray_p[k+1, :])

        print(dist_p, dist_s)
        dist = np.max([dist_p, dist_s])
    rp_att = 0.55

    # theta = np.random.rand() * np.pi - np.pi / 2
    # phi = np.random.rand() * 2 * np.pi
    # rp_att = np.sin(2*theta) * np.cos(phi)

    # calculating anelastic and scattering attenuation
    if quality:
        a_s_att = a_s_att_fun(magnitudes, dist)
    else:
        a_s_att = 1

    if dist < 1:
        gs_att = 1
    else:
        gs_att = 1 / dist

    if triaxial:
        att = gs_att * rp_att * a_s_att
    else:
        sta_evn_vect = station - event
        arrival_angle = sta_evn_vect / np.linalg.norm(sta_evn_vect)
        orientation = np.array(orientation)
        vect_att = np.abs(np.dot(arrival_angle, orientation))
        att = gs_att * rp_att * vect_att * a_s_att

    fp, fs = frechet(event, ttp, tts)

    index = np.argmin(np.abs((np.array(sta) * att) / noise - stalta_th))

    mag_sensitivity = magnitudes[index]

    # magnitudes = np.arange(-3, 4, 0.1)
    # tmp = [] 
    # for magnitude in magnitudes:
    #     tmp.append(sta_lta(magnitude, noise, stalta_th, ssd, vp, vs, rho, attenuation, acceleration))

    # index = np.argmin(tmp)
    # mag_sensitivity = magnitudes[index]

    # mag_sensitivity = 1

    # mag_sensitivity = minimize(f, 0, method='cg')

    outdict = {}
    outdict['evloc'] = event
    outdict['stloc'] = station
    outdict['mag_sensitivity'] = mag_sensitivity
    # outdict['mag_sensitivity'] = mag_sensitivity.x[0]
    outdict['frechet_p'] = fp
    outdict['frechet_s'] = fs
    # return (event_id, (event, station, mag_sensitivity.x[0], fp, fs))
    return (event_id, outdict)


def frechet(evloc, ttp, tts, tpert=1, vpert=50):
    """
    sensitivity matrix (Frechet Derivatives) calculation
    :param evloc: numpy array containing the event location
    :param ttp: eikonal P-wave travel time grid
    :type ttp: microquake.core.data.gridData
    :param tts: eikonal S-wave trave time grid
    :type tts: microquake.core.data.gridData
    :param dist: event sensor distance
    :param tpert: distance perturbation to apply to the event location to
    measure the sensitivity of the travel time with respect to location.
    Default=1
    :param vpert: perturbation to apply to the velocity in m/s to measure the
    sensitivity of the location with respect to velocity. Default=50.
    :return: the Frechet derivative matrix
    """

    frechet_p = []
    frechet_s = []

    # Loop over all dimensions
    for k in range(0, len(evloc)):
        evloc1 = evloc.copy()
        evloc1[k] -= tpert
        evloc2 = evloc.copy()
        evloc2[k] += tpert
        tp1 = ttp.interpolate(evloc1, grid_coordinate=False, mode='reflect',
                              order=1)[0]
        ts1 = tts.interpolate(evloc1, grid_coordinate=False, mode='reflect',
                              order=1)[0]
        tp2 = ttp.interpolate(evloc2, grid_coordinate=False, mode='reflect',
                              order=1)[0]
        ts2 = tts.interpolate(evloc2, grid_coordinate=False, mode='reflect',
                              order=1)[0]


        frechet_p.append((tp2 - tp1) / (2 * tpert))
        frechet_s.append((ts2 - ts1) / (2 * tpert))

    return np.array(frechet_p), np.array(frechet_s)


def sta_interp_func(magnitudes, ssd, vp, vs, rho, acceleration):
    """ 
    return an interpolation function
    """
    sta_ = []
    for mag in magnitudes:
        tr = synthetic_seismogram(mag, sampling_rate=10000, duration=2.0, vp=vp,
                                 vs=vs, rho=rho, SSD=ssd, pwave=True)
        tr.differentiate()
        if acceleration:
            tr.differentiate()

        signal = tr.data[np.abs(tr.data) > 0.01 * np.max(np.max(tr.data))]
        sta_.append(np.std(signal))

    # have to make sure it is var and not std
    return sta_


def sta_lta(magnitude, noise, stalta_th, attenuation, sta_signal):
    sta = sta_signal * attenuation + noise
    lta = noise
    stalta = sta / lta

    return np.linalg.norm(stalta / stalta_th - 1)


def key_values_to_dict(key_values):
    skey = key_values[0]
    dicts = key_values[1]
    dct_out = []
    for k, dct in enumerate(dicts):
        tmp_dct = dict(dct)
        if k == 0:
            dct_out = tmp_dct
        else:
            for key in dct_out.keys():
                try:
                    dct_out[key].append(tmp_dct[key])
                except:
                    dct_out[key] = np.vstack(
                        (dct_out[key], tmp_dct[key].ravel()))

    return skey, dct_out


def uncertainty(frechetm, pick_uncertainty, vpert=50):
    """
    calculate the location uncertainty from the frechet derivative,
    the pick uncertainty, the distance and velocity perturbation
    :param frechetm: Frechet derivative matrix
    :param pick_uncertainty: pick uncertainty
    :param vpert: velocity perturbation in m/s
    :return: return the location uncertainty or error ellipsoid (eigen values
    and eigen vectors)
    """
    hess = np.linalg.inv(np.dot(frechetm.T, frechetm))
    eig = np.linalg.eig(hess)
    eig_vects = eig[1]
    eig_values = np.sqrt(eig[0] * pick_uncertainty ** 2)
    order = np.argsort(eig_values)[-1::-1]
    eig_vects = eig_vects[order, :]
    eig_values = eig_values[order]

    return eig_values, eig_vects


def uncertainty_magnitude(key_values, mag, pick_uncertainty, min_nsensor):
    """
    calculate the uncertainty for a list of magnitudes given
    a pick uncertainty
    :param key_values: key and value from previous steps
    :param mags: list of magnitude
    :pick_uncertainty: picking /or travel time uncertainty 
    """
    key, df = key_values_to_dict(key_values)
    indices = np.nonzero(df['mag_sensitivity'] <= mag)[0]
    if len(indices) >= min_nsensor:
        frechetm = np.vstack(
            (df['frechet_p'][indices], df['frechet_s'][indices]))
        try:
            eig_values, eig_vect = uncertainty(frechetm, pick_uncertainty)
            unc = np.max(eig_values)
        except:
            unc = np.nan
            eig_vect = np.ones((3, 3)) * np.nan
            eig_values = np.ones(3) * np.nan
    else:
        unc = np.nan
        eig_vect = np.ones((3, 3)) * np.nan
        eig_values = np.ones(3) * np.nan

    outdict = {}
    outdict['uncertainty'] = unc
    outdict['event_id'] = key
    outdict['evloc'] = df['evloc'][0]
    outdict['nstation'] = len(indices)
    outdict['major'] = eig_values[0]
    outdict['semi-major'] = eig_values[1]
    outdict['minor'] = eig_values[2]
    outdict['major_vect'] = eig_vect[0, :]
    outdict['semi-major_vect'] = eig_vect[1, :]
    outdict['minor_vect'] = eig_vect[2, :]

    return (mag, outdict)


def write_uncertainty(outdir, key_values):
    """
    writing the uncertainty for a given magnitude in a file
    """
    mag = key_values[0]
    ofile = '%s/uncertainty_magnitude_%0.1f' % (outdir, mag)
    with open(ofile + '.csv', 'w') as out:
        # writing the header
        header = "ex, ey, ez, <uncertainty (m)>, <number of stations>, <major axis magnitude>," + \
                 "<major x>, <major y>, <major z>, <semi-major axis magnitude>," + \
                 "<semi-major x>, <semi-major y>, <semi-major z>, <minor axis magnitude>," + \
                 "<minor x>, <minor y>, <minor z>\n"

        out.write(header)
        for value in key_values[1]:
            value = dict(value)
            evloc = value['evloc']
            unc = value['uncertainty']
            nsta = value['nstation']
            mam = value['major']
            mav = value['major_vect']
            smam = value['semi-major']
            smav = value['semi-major_vect']
            mnam = value['minor']
            mnav = value['minor_vect']

            tmpl = "%d,%d,%d,%0.1f,%d,%0.1f,%f,%f,%f,%0.1f,%f,%f,%f,%0.1f,%f,%f,%f\n"
            lineout = tmpl % (
            evloc[0], evloc[1], evloc[2], unc, nsta, mam, mav[0], mav[1],
            mav[2], smam, smav[0], smav[1], smav[2], mnam, mnav[0],
            mnav[1], mnav[2])
            out.write(lineout)

        return


def minimum_magnitude(key_values, pick_uncertainty, max_uncertainty,
                      min_nsensor):
    """
    calculates the minimum magnitude sensitivity
    :param key_values:
    """
    key, df = key_values_to_dict(key_values)

    indices = np.argsort(df['mag_sensitivity'].ravel())

    for k in range(min_nsensor, len(df['mag_sensitivity'])):
        frechetm = np.vstack((df['frechet_p'][indices[0:k]],
                              df['frechet_s'][indices[0:k]]))
        eig_values, eig_vects = uncertainty(frechetm, pick_uncertainty)
        if np.max(eig_values) < max_uncertainty:
            break
    minimum_magnitude = df['mag_sensitivity'][indices[k - 1]]
    outdict = {}
    outdict['minimum_magnitude'] = minimum_magnitude
    outdict['uncertainty'] = np.max(eig_values)
    outdict['nstation'] = k
    outdict['evloc'] = df['evloc'][0]

    return (0, outdict)


def write_min_mag(outdir, key_values):
    ofile = '%s/minimum_magnitude.csv' % outdir
    with open(ofile, 'w') as out:
        header = "ex, ey, ez, <minimum magnitude>, <uncertainty (m)>, <number of stations>\n"
        out.write(header)

        for value in key_values[1]:
            value = dict(value)
            evloc = value['evloc']
            unc = value['uncertainty']
            nstation = value['nstation']
            min_mag = value['minimum_magnitude']

            tmpl = "%d, %d, %d, %0.1f, %0.1f, %d\n"
            lineout = tmpl % (
            evloc[0], evloc[1], evloc[2], min_mag, unc, nstation)
            out.write(lineout)

        return


def anelastic_attenuation(magnitude, distances, quality, SSD=1,
                          acceleration=False, vp=5000, vs=5000/1.41,
                          rho=2400, **kwargs):
    """
    Calculate an interpolation function for anelastic and scattering
    attenuation
    :param magnitude: a magnitude
    :param distances: list of distances
    :param SSD: static stress drop
    :param acceleration: true if simulation domain is acceleration
    :return: a list of magnitudes, attenuations and distances
    """
    from scipy import interpolate
    outlist = []
    seis = synthetic_seismogram(magnitude, SSD=SSD, duration=0.2, vp=vp,
                                vs=vs, rho=rho, sampling_rate=10000)
    SEIS = np.fft.fft(seis.data)
    freq = np.fft.fftfreq(len(seis.data), 1 / seis.stats.sampling_rate)

    A0 = seis.copy()
    if acceleration:
        A0.differentiate().differentiate()
    else:
        A0.differentiate()

    for dist in distances:
        tmp = np.real(np.fft.ifft(SEIS * np.exp(
            -np.pi * np.abs(freq) * dist / (quality * vp))))
        A = seis.copy()
        A.data = tmp

        if acceleration:
            A.differentiate().differentiate()
        else:
            A.differentiate()

        S = np.std(A.data[np.abs(A.data) > 0.01 * np.max(np.max(A.data))])
        S0 =np.std(A0.data[np.abs(A0.data) > 0.01 * np.max(np.max(
            A0.data))])
        a_s_att = S / S0
        outlist.append([magnitude, dist, a_s_att])
    return outlist



def step_1(params, sc):
    events = params.events.data
    rho = params.densitygrid.r
    simulparams = params.simulation

    stalta_th = simulparams.sta_lta_threshold
    ssd = simulparams.static_stress_drop
    try:
        quality = simulparams.quality
    except:
        quality = None


    acceleration = simulparams.amplitude_unit_acceleration

    magnitudes = np.arange(-3, 3, 0.1)

    global seis
    seis = []
    if quality:
        vp = np.median(params.velgrids.grids.vp.data)
        vs = np.median(params.velgrids.grids.vs.data)
        shp = np.array(params.velgrids.grids.vp.shape)
        spc = params.velgrids.grids.vp.spacing
        max_dist = np.int(np.linalg.norm(spc * shp))
        distances = np.arange(0, max_dist, 10)

        out = np.array(mq_map(sc, anelastic_attenuation, magnitudes, distances,
                              quality, acceleration=acceleration,
                              flat_map=True, vp=vp, vs=vs))
        mags = out[:, 0]
        dists = out[:, 1]
        att = out[:, 2]
        a_s_att_fun = interp2d(mags, dists, att, kind='cubic')
    else:
        a_s_att_fun = None


    vmodel = params.velgrids
    if vmodel.homogeneous:
        vp = vmodel.vp
        vs = vmodel.vs
        rho = params.densitygrid.r
        # silly way to improve efficiency. calculating a lookup table
        sta_ = sta_interp_func(magnitudes, ssd, vp, vs, rho, acceleration)
    else:
        # print 'heteregenous models are not currently supported ... nothing
        # done'
        # return
        vp = vmodel.grids.vp
        vs = vmodel.grids.vs
        rho = params.densitygrid.data
        # this needs to be changed it is not adequate
        sta_ = sta_interp_func(magnitudes, ssd, np.mean(vp), np.mean(vs),
                               np.mean(rho), acceleration)

    tup = []
    ev_tup = []
    for ev_id, event in enumerate(events):
        ev_tup.append((ev_id, event))

    scenario = params.sensors
    rdds = []
    for station in scenario.site.stations():

        if params.noisegrid:
            noise = params.noisegrid.grids.noise.interpolate(station.loc,
                                                    grid_coordinate=False)
        else:
            noise = simulparams.noise_amplitude

        if len(station.channels) == 1:
            triaxial = False
            orientation = station.channels[0].orientation
        else:
            triaxial = True
            orientation = []

        # import time
        # start = time.clock()
        # for k_, evtp in enumerate(ev_tup[0:80]):
        #     print k_
        #     detection_threshold(evtp, station.loc, stalta_th, ssd, noise,
        #                                      acceleration, vp, vs, rho,
        #                                      vmodel.homogeneous,triaxial,
        #                                      orientation, magnitudes, sta_)
        # end = time.clock()
        # print end - start
        # tracer()()



        rdd = sc.parallelize(ev_tup).persist()

        ttp = eikonal_solver(vmodel.grids.vp, station.loc, station.code)
        tts = eikonal_solver(vmodel.grids.vs, station.loc, station.code)

        # import matplotlib.pyplot as plt
        # ev1 = event.copy()
        # ev1[k] += ttp.spacing / 2
        # ev2 = event.copy()
        # Tracer()()

        f = lambda evtp: detection_threshold(evtp, station.loc, stalta_th,
                                             noise, vp, vs, ttp, tts, rho,
                                             vmodel.homogeneous, triaxial,
                                             orientation, magnitudes, sta_,
                                             ssd, quality, a_s_att_fun)

        rdd2 = rdd.map(f)

        # rdd2 = rdd.map(lambda event: sensitivity(event[0], event[1]))

        rdds.append(rdd2.persist())

    rdd_scenario = sc.union(rdds).persist()

    odir = io_directory(params)

    # try:
    #     shutil.rmtree(odir)
    # except oserror:
    #     pass

    if os.path.exists('%s/step_1.pickle' % odir):
        shutil.rmtree('%s/step_1.pickle' % odir)
    if os.path.exists('%s/step_1.csv' % odir):
        shutil.rmtree('%s/step_1.csv' % odir)

    rdd_scenario.saveAsPickleFile('%s/step_1.pickle' % odir)
    rdd_scenario.saveAsTextFile('%s/step_1.csv' % odir)
    rdd_scenario.unpersist()
    rdd2.unpersist()
    rdd.unpersist()


def step_2(params, sc):
    vmodel = params.velgrids
    if vmodel.homogeneous:
        pass
    else:
        return

    pick_uncertainty = params.simulation.pick_uncertainty
    try:
        velocity_perturbation = params.simulation.velocity_perturbation
    except:
        velocity_perturbation = 0
    min_nsensor = params.simulation.minimum_number_sensor
    max_uncertainty = params.simulation.maximum_uncertainty

    iodir = io_directory(params)
    infile = '%s/step_1.pickle' % iodir

    rdd = sc.pickleFile(infile).groupByKey().persist()

    if "detection_range" in params.simulation.keys():
        params.simulation.magnitude_list = [1] # overriding magnitude list



    rdd_minmag = rdd.map(lambda key_val: minimum_magnitude(key_val,
                                                           pick_uncertainty,
                                                           max_uncertainty,
                                                           min_nsensor)).persist()

    foo = rdd_minmag.groupByKey().map(
        lambda key_val: write_min_mag(iodir, key_val)).collect()
    rdds = []

    for mag in params.simulation.magnitude_list:
        rdd_tmp = rdd.map(lambda key_val: uncertainty_magnitude(key_val, mag,
                                                                pick_uncertainty,
                                                                min_nsensor)).persist()

        rdds.append(rdd_tmp)

    rdd_tmp = sc.union(rdds).groupByKey()
    foo = rdd_tmp.map(
        lambda key_values: write_uncertainty(iodir, key_values)).collect()
    rdd_tmp.unpersist()
    rdd.unpersist()
    return


def step_3(params):
    pass


parser = argparse.ArgumentParser(description='system performance simulation')

help_steps = """
the steps to be performed\n
step 1: calculate the sensitivity and frechet derivatives\n
(dt/dx, dt/dy, and dt/dz) at each grid point
step 2: calculate the location uncertainty at each grid point\n
step 3: calculate the minimum magnitude sensitivity\n
note that steps can be combined (
e.g., --step 1 2 3 or -- step 2 3)\n
"""

help_local = "override the setting in the config file and run simulation locally"

parser.add_argument('config_file', type=str,
                    help="xml configuration file")

# defining the command line options

def main():
    args = parser.parse_args()

    config_file = args.config_file
    params = ctl.parse_control_file(config_file)

    init_io_directories(params)

    from microquake.spark import init_spark_from_params

    # Tracer()()
    sc = init_spark_from_params(params)

    step_1(params, sc)
    step_2(params, sc)


import argparse

if __name__ == '__main__':
   main()
