import json
import argparse
import concurrent.futures
import threading
from itertools import product
import ctypes

import numpy as np
import nibabel as nb
import popeye.og as og
import popeye.utilities as utils
from popeye.visual_stimulus import VisualStimulus


def estimate_prf(coords, bold, stimuli, parameters={}):

    bold = nb.load(bold).dataobj[coords]
    if bold.std() == 0:
        # Outside brain
        return (0, 0, 0, 0)

    stimuli = np.squeeze(nb.load(stimuli).get_fdata())

    parameters = {
        'viewing_distance': 38,
        'screen_width': 25,
        'scale_factor': 1.0,
        'tr': 1.0,
        'dtype': ctypes.c_int16,
        **parameters,
    }

    stimulus = VisualStimulus(
        stimuli,
        parameters['viewing_distance'],
        parameters['screen_width'],
        parameters['scale_factor'],
        parameters['tr'],
        parameters['dtype']
    )

    model = og.GaussianModel(stimulus, utils.spm_hrf)
    model.hrf_delay = 0
    model.mask_size = 6

    x_grid = utils.grid_slice(-10,10,5)
    y_grid = utils.grid_slice(-10,10,5)
    s_grid = utils.grid_slice (0.25,5.25,5)

    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (0.001,12.0)
    b_bound = (1e-8,1e2)
    m_bound = (-5.0,5.0)

    grids = (x_grid, y_grid, s_grid,)
    bounds = (x_bound, y_bound, s_bound, b_bound, m_bound)

    model = og.GaussianFit(model, bold, grids, bounds)

    return (
        model.rho,
        model.theta,
        model.sigma,
        model.rsquared
    )

def load_config(config):
    try:
        with open(config) as f:
            return json.load(f)
    except:
        raise argparse.ArgumentTypeError('Could not load config file')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("bold")
    parser.add_argument("mask")
    parser.add_argument("stimuli")
    parser.add_argument("config", type=load_config)
    args = parser.parse_args()

    bold = nb.load(args.bold)
    mask = nb.load(args.mask)
    mask_data = mask.get_fdata()
    stimuli = nb.load(args.stimuli)

    volume_shape = bold.shape[:-1]

    valid_voxels = np.where(mask_data > 0)
    total_voxels = len(valid_voxels[0])
    valid_voxels = zip(*valid_voxels)

    # total_voxels = np.prod(volume_shape)
    # valid_voxels = product(*[range(n) for n in volume_shape])

    # total_voxels = 5*5*5
    # valid_voxels = product(*[range(50, 55) for n in volume_shape])
    # valid_voxels = valid_voxels[0:20]  # TODO remove this line

    eccentricity = np.zeros(volume_shape)
    polar_angle = np.zeros(volume_shape)
    rf_size = np.zeros(volume_shape)
    rsquared = np.zeros(volume_shape)

    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        try:
            voxels = {}
            done_voxels = 0

            print(f'{done_voxels}/{total_voxels}')
            while done_voxels < total_voxels:
                try:
                    if len(voxels) < pool._max_workers:
                        vv = next(valid_voxels)
                        work = pool.submit(estimate_prf, vv, args.bold, args.stimuli)
                        voxels[work] = vv
                except StopIteration as e:
                    pass

                try:
                    for future in concurrent.futures.as_completed(voxels, timeout=1):
                        rho, theta, sigma, r2 = future.result(timeout=1)
                        vv = voxels.pop(future)

                        eccentricity[vv] = rho
                        polar_angle[vv] = theta
                        rf_size[vv] = sigma
                        rsquared[vv] = r2

                        done_voxels += 1

                        print(f'{done_voxels}/{total_voxels}')
                except concurrent.futures._base.TimeoutError:
                    pass

            nb.save(nb.Nifti1Image(eccentricity, bold.affine), './output/eccentricity.nii.gz')
            nb.save(nb.Nifti1Image(polar_angle, bold.affine), './output/polarAngle.nii.gz')
            nb.save(nb.Nifti1Image(rf_size, bold.affine), './output/rfWidth.nii.gz')
            nb.save(nb.Nifti1Image(rsquared, bold.affine), './output/r2.nii.gz')

        except KeyboardInterrupt:
            pass
        finally:
            pool.shutdown()