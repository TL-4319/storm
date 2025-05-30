from carbs.swarm_estimator.tracker import ProbabilityHypothesisDensity


import sys
import pytest
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from copy import deepcopy

import scipy.stats as stats

import gncpy.filters as gfilts
import gncpy.dynamics.basic as gdyn
import gncpy.distributions as gdistrib
import carbs.swarm_estimator.tracker as tracker
#from ....src.carbs.swarm_estimator import tracker as tracker
import serums.models as smodels
from serums.enums import GSMTypes, SingleObjectDistance


global_seed = 69
debug_plots = False

_meas_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)


def _state_mat_fun(t, dt, useless):
    # print('got useless arg: {}'.format(useless))
    return np.array(
        [[1.0, 0, dt, 0], [0.0, 1.0, 0, dt], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    )


def _meas_mat_fun(t, useless):
    # print('got useless arg: {}'.format(useless))
    return _meas_mat


def _meas_mat_fun_nonlin(t, x, *args):
    return _meas_mat @ x


def _setup_double_int_kf(dt):
    m_noise = 0.02
    p_noise = 0.2

    filt = gfilts.KalmanFilter()
    filt.set_state_model(state_mat_fun=_state_mat_fun)
    filt.set_measurement_model(meas_fun=_meas_mat_fun)
    filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(
        dt, np.array([[p_noise**2]])
    )
    filt.meas_noise = m_noise**2 * np.eye(2)

    return filt


def _setup_double_int_stf(dt):
    m_noise = 0.02
    p_noise = 0.2

    filt = gfilts.StudentsTFilter()
    filt.set_state_model(state_mat_fun=_state_mat_fun)
    filt.set_measurement_model(meas_fun=_meas_mat_fun)
    filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(
        dt, np.array([[p_noise**2]])
    )
    filt.meas_noise = m_noise**2 * np.eye(2)

    filt.meas_noise_dof = 3
    filt.proc_noise_dof = 3
    # Note filt.dof is determined by the birth terms

    return filt


def _setup_double_int_pf(dt, rng):
    m_noise = 0.02
    p_noise = 0.2

    doubleInt = gdyn.DoubleIntegrator()
    proc_noise = doubleInt.get_dis_process_noise_mat(dt, np.array([[p_noise**2]]))

    def meas_likelihood(meas, est, *args):
        return stats.multivariate_normal.pdf(
            meas.flatten(), mean=est.flatten(), cov=m_noise**2 * np.eye(2)
        )

    def proposal_sampling_fnc(x, rng):  # noqa
        val = rng.multivariate_normal(x.flatten(), proc_noise).reshape(x.shape)
        return val

    def proposal_fnc(x_hat, mean, y, *args):
        return 1

    filt = gfilts.ParticleFilter(rng=rng)
    filt.set_state_model(dyn_obj=doubleInt)
    filt.set_measurement_model(meas_fun=_meas_mat_fun_nonlin)

    filt.proc_noise = proc_noise.copy()
    filt.meas_noise = m_noise**2 * np.eye(2)

    filt.meas_likelihood_fnc = meas_likelihood
    filt.proposal_sampling_fnc = proposal_sampling_fnc
    filt.proposal_fnc = proposal_fnc

    return filt


def _setup_double_int_upf(dt, rng, use_MCMC):
    m_noise = 0.02
    p_noise = 0.2

    doubleInt = gdyn.DoubleIntegrator()

    filt = gfilts.UnscentedParticleFilter(use_MCMC=use_MCMC, rng=rng)
    filt.use_cholesky_inverse = False

    filt.set_state_model(dyn_obj=doubleInt)
    filt.set_measurement_model(meas_mat=_meas_mat.copy())

    proc_noise = doubleInt.get_dis_process_noise_mat(dt, np.array([[p_noise**2]]))
    meas_noise = m_noise**2 * np.eye(2)
    filt.proc_noise = proc_noise.copy()
    filt.meas_noise = meas_noise.copy()

    return filt


def _setup_double_int_gci_kf(dt):
    m_noise = 0.002
    p_noise = 0.2
    m_model1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    m_model2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    m_model_list = [m_model1, m_model2]

    meas_noise_list = [m_noise**2 * np.eye(2), 0.01 * m_noise**2 * np.eye(2)]

    doubleInt = gdyn.DoubleIntegrator()

    in_filt = gfilts.KalmanFilter()
    in_filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator())
    in_filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(
        dt, np.array([[p_noise**2]])
    )
    in_filt.meas_noise = meas_noise_list[0]
    filt = gfilts.GCIFilter(
        base_filter=in_filt,
        meas_model_list=m_model_list,
        meas_noise_list=meas_noise_list,
    )
    filt.cov = 0.25 * np.eye(4)
    return filt


def _setup_double_int_gci_ekf(dt):
    def range_func(t, x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2)
        # return np.array([x[0], x[1]])

    def bear_func(t, x):
        # return np.arctan2(x[0], x[1])
        return np.arctan2(x[1], x[0])

    def range_func2(t, x):
        # return np.array([x[0] + 5, x[1]])
        return np.sqrt((x[0] - 5) ** 2 + x[1] ** 2)

    def bear_func2(t, x):
        # return np.arctan2(x[0] - 5, x[1])
        return np.arctan2(x[1], x[0] - 5)

    def xfunc(t, x):
        return x[0]

    def yfunc(t, x):
        return x[1]

    def xfunc2(t, x):
        return x[0] + 5

    m_noise = 0.02
    p_noise = 0.2
    # m_noise_list = [
    #     np.diag([m_noise**2, (np.pi / 180 * m_noise) ** 2]),
    #     np.diag([m_noise**2, (np.pi / 180 * m_noise) ** 2]),
    # ]
    # m_mdl_lst = [[range_func, bear_func], [range_func2, bear_func2]]
    m_noise_list = [
        np.diag([m_noise**2]),
        np.diag([m_noise**2]),
    ]
    m_mdl_lst = [[xfunc, yfunc], [xfunc2, yfunc]]

    in_filt = gfilts.ExtendedKalmanFilter()
    in_filt.dt = dt
    in_filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator())
    in_filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(
        dt, np.array([[p_noise**2]])
    )
    # in_filt.meas_noise = m_noise**2 * np.eye(2)
    in_filt.meas_noise = np.diag([m_noise**2, (np.pi / 180 * m_noise) ** 2])

    filt = gfilts.GCIFilter(
        base_filter=in_filt,
        meas_model_list=m_mdl_lst,
        meas_noise_list=m_noise_list,
    )
    filt.cov = 0.25 * np.eye(4)
    # filt.cov = np.diag([2.0, 2.0, 2.0, 2.0])

    return filt


def _setup_ct_ktr_gci_imm_kf(dt):
    m_noise = 0.02
    p_noise = 0.2
    m_model1 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=float)
    m_model2 = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]], dtype=float)
    m_model_list = [m_model1, m_model2]

    meas_noise_list = [m_noise**2 * np.eye(2), 0.01 * m_noise**2 * np.eye(2)]

    dyn_obj1 = gdyn.CoordinatedTurnKnown(turn_rate=0)
    dyn_obj2 = gdyn.CoordinatedTurnKnown(turn_rate=60 * np.pi / 180)

    in_filt1 = gfilts.KalmanFilter()
    in_filt1.set_state_model(dyn_obj=dyn_obj1)
    in_filt1.set_measurement_model(meas_mat=m_model1)
    in_filt1.proc_noise = np.diag([1, 1, 1, 1, 3 * np.pi / 180]) * np.array(
        [[p_noise**2]]
    )
    in_filt1.meas_noise = meas_noise_list[0]

    in_filt2 = gfilts.KalmanFilter()
    in_filt2.set_state_model(dyn_obj=dyn_obj2)
    in_filt2.set_measurement_model(meas_mat=m_model1)
    in_filt2.proc_noise = np.diag([1, 1, 1, 1, 3 * np.pi / 180]) * np.array(
        [[p_noise**2]]
    )
    in_filt2.meas_noise = meas_noise_list[0]

    filt_list = [in_filt1, in_filt2]

    model_trans = np.array([[1 - 1 / 200, 1 / 200], [1 / 200, 1 - 1 / 200]])

    filt = gfilts.IMMGCIFilter(
        meas_model_list=m_model_list, meas_noise_list=meas_noise_list
    )
    filt.initialize_filters(filt_list, model_trans)

    return filt


def __gsm_import_dist_factory():
    def import_dist_fnc(parts, rng):
        new_parts = np.nan * np.ones(parts.particles.shape)

        disc = 0.99
        a = (3 * disc - 1) / (2 * disc)
        h = np.sqrt(1 - a**2)
        last_means = np.mean(parts.particles, axis=0)
        means = a * parts.particles[:, 0:2] + (1 - a) * last_means[0:2]

        # df, sig
        for ind in range(means.shape[1]):
            std = np.sqrt(h**2 * np.cov(parts.particles[:, ind]))

            for ii, m in enumerate(means):
                new_parts[ii, ind] = stats.norm.rvs(
                    loc=m[ind], scale=std, random_state=rng
                )
        for ii in range(new_parts.shape[0]):
            new_parts[ii, 2] = stats.invgamma.rvs(
                np.abs(new_parts[ii, 0]) / 2,
                scale=1 / (2 / np.abs(new_parts[ii, 0])),
                random_state=rng,
            )
        return new_parts

    return import_dist_fnc


def _setup_qkf(dt, use_sqkf, m_vars):
    state_mat = np.vstack(
        (
            np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
            np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
            np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2))),
        )
    )
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun(t, x, *args):
        return np.array(
            [[np.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2)], [np.arctan2(x[1, 0], x[0, 0])]]
        )

    # define base GSM parameters
    if use_sqkf:
        filt = gfilts.SquareRootQKF()
    else:
        filt = gfilts.QuadratureKalmanFilter()
    filt.set_state_model(state_mat=state_mat)
    filt.set_measurement_model(meas_fun=meas_fun)
    filt.proc_noise = proc_cov
    filt.meas_noise = np.diag(m_vars)

    filt.points_per_axis = 3

    return filt, state_mat, meas_fun


def _setup_ukf(dt, m_vars):
    state_mat = np.vstack(
        (
            np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
            np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
            np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2))),
        )
    )
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun(t, x, *args):
        return np.vstack((meas_fun_range(t, x), meas_fun_bearing(t, x)))

    def meas_fun_range(t, x, *args):
        return np.array([np.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2)])

    def meas_fun_bearing(t, x, *args):
        return np.array([np.arctan2(x[1, 0], x[0, 0])])

    # define filter parameters
    filt = gfilts.UnscentedKalmanFilter()

    filt.set_state_model(state_mat=state_mat)
    filt.set_measurement_model(meas_fun_lst=[meas_fun_range, meas_fun_bearing])
    filt.proc_noise = proc_cov
    filt.meas_noise = np.diag(m_vars)

    # define UKF specific parameters
    filt.alpha = 0.5
    filt.kappa = 1
    filt.beta = 1.5

    return filt, state_mat, meas_fun


def __gsm_import_w_factory(inov_cov):
    def import_w_fnc(meas, parts):
        stds = np.sqrt(parts[:, 2] * parts[:, 1] ** 2 + inov_cov)
        return np.array([stats.norm.pdf(meas.item(), scale=scale) for scale in stds])

    return import_w_fnc


def _setup_qkf_gsm(dt, rng, use_sqkf, m_dfs, m_vars):
    state_mat = np.vstack(
        (
            np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
            np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
            np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2))),
        )
    )
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun(t, x, *args):
        return np.array(
            [[np.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2)], [np.arctan2(x[1, 0], x[0, 0])]]
        )

    # define base GSM parameters
    if use_sqkf:
        filt = gfilts.SQKFGaussianScaleMixtureFilter()
    else:
        filt = gfilts.QKFGaussianScaleMixtureFilter()
    filt.set_state_model(state_mat=state_mat)
    filt.proc_noise = proc_cov
    filt.set_measurement_model(meas_fun=meas_fun)

    # define measurement noise filters
    num_parts = 500
    bootstrap_lst = [None] * 2

    # manually setup each bootstrap filter (stripped down PF)
    for ind in range(len(bootstrap_lst)):
        mf = gfilts.BootstrapFilter()
        mf.importance_dist_fnc = __gsm_import_dist_factory()
        mf.particleDistribution = gdistrib.SimpleParticleDistribution()
        df_particles = stats.uniform.rvs(
            loc=1, scale=4, size=num_parts, random_state=rng
        )
        sig_particles = stats.uniform.rvs(
            loc=0, scale=5 * np.sqrt(m_vars[ind]), size=num_parts, random_state=rng
        )
        z_particles = np.nan * np.ones(num_parts)
        for ii, v in enumerate(df_particles):
            z_particles[ii] = stats.invgamma.rvs(
                v / 2, scale=1 / (2 / v), random_state=rng
            )
        mf.particleDistribution.particles = np.stack(
            (df_particles, sig_particles, z_particles), axis=1
        )

        mf.particleDistribution.num_parts_per_ind = np.ones(num_parts)
        mf.particleDistribution.weights = 1 / num_parts * np.ones(num_parts)
        mf.rng = rng
        bootstrap_lst[ind] = mf
    importance_weight_factory_lst = [__gsm_import_w_factory] * len(bootstrap_lst)
    filt.set_meas_noise_model(
        bootstrap_lst=bootstrap_lst,
        importance_weight_factory_lst=importance_weight_factory_lst,
    )

    # define QKF specific parameters for core filter
    filt.points_per_axis = 3

    return filt, state_mat, meas_fun


def _setup_ukf_gsm(dt, rng, m_dfs, m_vars):
    state_mat = np.vstack(
        (
            np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
            np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
            np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2))),
        )
    )
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun(t, x, *args):
        return np.vstack((meas_fun_range(t, x), meas_fun_bearing(t, x)))

    def meas_fun_range(t, x, *args):
        return np.array([np.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2)])

    def meas_fun_bearing(t, x, *args):
        return np.array([np.arctan2(x[1, 0], x[0, 0])])

    # define base GSM parameters
    filt = gfilts.UKFGaussianScaleMixtureFilter()

    filt.set_state_model(state_mat=state_mat)
    filt.proc_noise = proc_cov
    filt.set_measurement_model(meas_fun_lst=[meas_fun_range, meas_fun_bearing])

    # define measurement noise filters
    num_parts = 500
    bootstrap_lst = [None] * 2

    # manually setup each bootstrap filter (stripped down PF)
    for ind in range(len(bootstrap_lst)):
        mf = gfilts.BootstrapFilter()
        mf.importance_dist_fnc = __gsm_import_dist_factory()
        mf.particleDistribution = gdistrib.SimpleParticleDistribution()
        df_particles = stats.uniform.rvs(
            loc=1, scale=4, size=num_parts, random_state=rng
        )
        sig_particles = stats.uniform.rvs(
            loc=0, scale=5 * np.sqrt(m_vars[ind]), size=num_parts, random_state=rng
        )
        z_particles = np.nan * np.ones(num_parts)
        for ii, v in enumerate(df_particles):
            z_particles[ii] = stats.invgamma.rvs(
                v / 2, scale=1 / (2 / v), random_state=rng
            )
        mf.particleDistribution.particles = np.stack(
            (df_particles, sig_particles, z_particles), axis=1
        )

        mf.particleDistribution.num_parts_per_ind = np.ones(num_parts)
        mf.particleDistribution.weights = 1 / num_parts * np.ones(num_parts)
        mf.rng = rng
        bootstrap_lst[ind] = mf
    importance_weight_factory_lst = [__gsm_import_w_factory] * len(bootstrap_lst)
    filt.set_meas_noise_model(
        bootstrap_lst=bootstrap_lst,
        importance_weight_factory_lst=importance_weight_factory_lst,
    )

    # define UKF specific parameters for core filter
    filt.alpha = 0.5
    filt.kappa = 1
    filt.beta = 1.5

    return filt, state_mat, meas_fun


def _setup_ekf_gsm(dt, rng, m_dfs, m_vars):
    state_mat = np.vstack(
        (
            np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
            np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
            np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2))),
        )
    )
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun(t, x, *args):
        return np.vstack((meas_fun_range(t, x), meas_fun_bearing(t, x)))

    def meas_fun_range(t, x, *args):
        return np.array([np.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2)])

    def meas_fun_bearing(t, x, *args):
        return np.array([np.arctan2(x[1, 0], x[0, 0])])

    # define base GSM parameters
    filt = gfilts.EKFGaussianScaleMixtureFilter()

    def _x_dot(t, x, *args):
        return x[2]

    def _y_dot(t, x, *args):
        return x[3]

    def _x_ddot(t, x, *args):
        return x[4]

    def _y_ddot(t, x, *args):
        return x[5]

    def _x_dddot(t, x, *args):
        return 0

    def _y_dddot(t, x, *args):
        return 0

    ode_lst = [_x_dot, _y_dot, _x_ddot, _y_ddot, _x_dddot, _y_dddot]

    filt.set_state_model(ode_lst=ode_lst)
    filt.proc_noise = proc_cov
    filt.set_measurement_model(meas_fun_lst=[meas_fun_range, meas_fun_bearing])
    filt.cov = np.diag((5 * 10**4, 5 * 10**4, 8, 8, 0.02, 0.02))

    # define measurement noise filters
    num_parts = 500

    range_gsm = smodels.GaussianScaleMixture(
        gsm_type=GSMTypes.STUDENTS_T,
        degrees_of_freedom=m_dfs[0],
        df_range=(1, 5),
        scale=np.sqrt(m_vars[0]).reshape((1, 1)),
        scale_range=(0, 5 * np.sqrt(m_vars[0])),
    )

    bearing_gsm = smodels.GaussianScaleMixture(
        gsm_type=GSMTypes.STUDENTS_T,
        degrees_of_freedom=m_dfs[1],
        df_range=(1, 5),
        scale=np.sqrt(m_vars[1]).reshape((1, 1)),
        scale_range=(0, 5 * np.sqrt(m_vars[1])),
    )

    filt.set_meas_noise_model(
        gsm_lst=[range_gsm, bearing_gsm], num_parts=num_parts, rng=rng
    )

    # set EKF specific settings
    filt.dt = dt

    return filt, state_mat, meas_fun


def _setup_ctktr_kf(dt):
    m_noise = 0.02
    p_noise = 0.2

    filt = gfilts.KalmanFilter()
    filt.set_state_model(dyn_obj=gdyn.CoordinatedTurnKnown(turn_rate=0))
    filt.set_measurement_model(
        meas_mat=np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    filt.proc_noise = np.eye(5) * np.array([[p_noise**2]])
    filt.meas_noise = m_noise**2 * np.diag([1.0, 1.0, 1.0 * np.pi / 180])

    return filt


def _setup_imm_ctktr_kf(dt):
    m_noise = 0.02
    p_noise = 0.2

    dyn_obj1 = gdyn.CoordinatedTurnKnown(turn_rate=0)
    dyn_obj2 = gdyn.CoordinatedTurnKnown(turn_rate=60 * np.pi / 180)

    in_filt1 = gfilts.KalmanFilter()
    in_filt1.set_state_model(dyn_obj=dyn_obj1)
    m_mat = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])
    # m_mat = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    in_filt1.set_measurement_model(meas_mat=m_mat)
    in_filt1.proc_noise = np.diag([1, 1, 1, 1, 3 * np.pi / 180]) * np.array(
        [[p_noise**2]]
    )
    # in_filt1.meas_noise = m_noise ** 2 * np.diag([1.0, 1.0])
    in_filt1.meas_noise = m_noise**2 * np.diag([1.0, 1.0, 3 * np.pi / 180])

    in_filt2 = gfilts.KalmanFilter()
    in_filt2.set_state_model(dyn_obj=dyn_obj2)
    in_filt2.set_measurement_model(meas_mat=m_mat)
    in_filt2.proc_noise = np.diag([1, 1, 1, 1, 3 * np.pi / 180]) * np.array(
        [[p_noise**2]]
    )
    # in_filt2.meas_noise = m_noise ** 2 * np.diag([1.0, 1.0])
    in_filt2.meas_noise = m_noise**2 * np.diag([1.0, 1.0, 3 * np.pi / 180])

    v = np.sqrt(2**2 + 1**2)
    angle = 60 * np.pi / 180
    vx0 = v * np.cos(angle)
    vy0 = v * np.sin(angle)

    filt_list = [in_filt1, in_filt2]

    model_trans = np.array([[1 - 1 / 200, 1 / 200], [1 / 200, 1 - 1 / 200]])

    init_means = [
        np.array([10.0, 0.0, 0.0, 1.0, 0.0]).reshape(-1, 1),
        np.array([10.0, 0.0, 0.0, 1.0, 0.0]).reshape(-1, 1),
    ]
    init_covs = [
        np.diag(np.array([1.0, 1.0, 1.0, 1.0, 3.0 * np.pi / 180])) ** 2,
        np.diag(np.array([1.0, 1.0, 1.0, 1.0, 3.0 * np.pi / 180])) ** 2,
    ]

    filt = gfilts.InteractingMultipleModel()
    filt.initialize_filters(filt_list, model_trans)
    # filt.set_models(
    #     filt_list, model_trans, init_means, init_covs, init_weights=[0.5, 0.5]
    # )
    return filt


def _setup_phd_double_int_birth():
    mu = [np.array([10.0, 0.0, 0.0, 0.0]).reshape((4, 1))]
    cov = [np.diag(np.array([1, 1, 1, 1])) ** 2]
    gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[1])

    return [
        gm0,
    ]


def _setup_gm_glmb_double_int_birth():
    mu = [np.array([10.0, 0.0, 0.0, 1.0]).reshape((4, 1))]
    cov = [np.diag(np.array([1, 1, 1, 1])) ** 2]
    gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[1])

    return [
        (gm0, 0.003),
    ]


def _setup_stm_glmb_double_int_birth():
    mu = [np.array([10.0, 0.0, 0.0, 1.0]).reshape((4, 1))]
    scale = [np.diag(np.array([1, 1, 1, 1])) ** 2]
    stm0 = smodels.StudentsTMixture(means=mu, scalings=scale, weights=[1], dof=3)

    return [
        (stm0, 0.003),
    ]


def _setup_smc_glmb_double_int_birth(num_parts, rng):
    means = [np.array([10.0, 0.0, 0.0, 2.0]).reshape((4, 1))]
    cov = np.diag(np.array([1, 1, 1, 1])) ** 2
    b_probs = [
        0.003,
    ]

    birth_terms = []
    for m, p in zip(means, b_probs):
        distrib = gdistrib.ParticleDistribution()
        spread = 2 * np.sqrt(np.diag(cov)).reshape(m.shape)
        l_bnd = m - spread / 2
        for ii in range(0, num_parts):
            part = gdistrib.Particle()
            part.point = l_bnd + spread * rng.random(m.shape)
            w = 1 / num_parts
            distrib.add_particle(part, w)
        birth_terms.append((distrib, p))
    return birth_terms


def _setup_usmc_glmb_double_int_birth(num_parts, rng):
    # means = [np.array([10., 0., 0., 2.]).reshape((4, 1))]
    means = [np.array([20, 80, 3, -3]).reshape((4, 1))]
    # cov = np.diag(np.array([1, 1, 1, 1]))**2
    cov = np.diag([3**2, 5**2, 2**2, 1])
    b_probs = [
        0.005,
    ]
    alpha = 10**-3
    kappa = 0

    birth_terms = []
    for m, p in zip(means, b_probs):
        distrib = gdistrib.ParticleDistribution()
        spread = 2 * np.sqrt(np.diag(cov)).reshape(m.shape)
        l_bnd = m - spread / 2
        for ii in range(0, num_parts):
            part = gdistrib.Particle()
            part.point = l_bnd + spread * rng.random(m.shape)
            part.uncertainty = cov.copy()
            part.sigmaPoints = gdistrib.SigmaPoints(alpha=alpha, kappa=kappa)
            part.sigmaPoints.update_points(part.point, part.uncertainty)
            part.sigmaPoints.init_weights()
            distrib.add_particle(part, 1 / num_parts)
        birth_terms.append((distrib, p))
    return birth_terms


def _setup_gsm_birth():
    # note: GSM filter assumes noise is conditionally Gaussian so use GM with 1 term for birth
    means = [np.array([2000, 2000, 20, 20, 0, 0]).reshape((6, 1))]
    cov = [np.diag((5 * 10**4, 5 * 10**4, 8, 8, 0.02, 0.02))]
    gm0 = smodels.GaussianMixture(means=means, covariances=cov, weights=[1])

    return [
        (gm0, 0.05),
    ]


def _setup_imm_phd_ct_ktr_birth():
    mu = [np.array([10.0, 0.0, 1.0, 2.0, 0.0]).reshape((5, 1))]
    cov = [np.diag(np.array([1, 1, 0.1, 0.1, 3 * np.pi / 180])) ** 2]
    # cov = [np.diag(np.array([1, 1, 1, 1, np.pi])) ** 2]
    gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[1])

    return [
        gm0,
    ]


def _setup_imm_gm_glmb_ct_ktr_birth():
    mu = [np.array([10.0, 0.0, 1, 2, 0.0]).reshape((5, 1))]
    cov = [np.diag(np.array([1, 1, 0.1, 0.1, 3 * np.pi / 180])) ** 2]
    # cov = [np.diag(np.array([1, 1, 1, 1, np.pi])) ** 2]
    gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[1])

    return [
        (gm0, 0.003),
    ]


def _setup_pmbm_double_int_birth():
    mu = [np.array([1.0, 1.0, 1, 2]).reshape((4, 1))]
    cov = [np.diag(np.array([0.1, 0.1, 0.01, 0.01])) ** 2]
    # cov = [np.diag(np.array([1, 1, 0.1, 0.1])) ** 2]
    gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[1])
    return [gm0]


def _setup_stm_pmbm_double_int_birth():
    mu = [np.array([0.0, 0.0, 1, 2]).reshape((4, 1))]
    scale = [np.diag(np.array([1, 1, 1, 1])) ** 2]
    stm0 = smodels.StudentsTMixture(means=mu, scalings=scale, weights=[1], dof=3)
    return [stm0]


def _setup_smc_pmbm_double_int_birth(num_parts, rng):
    means = [np.array([0.0, 0.0, 1.0, 2.0]).reshape((4, 1))]
    cov = np.diag(np.array([1, 1, 1, 1])) ** 2

    birth_terms = []
    for m in means:
        distrib = gdistrib.ParticleDistribution()
        spread = 2 * np.sqrt(np.diag(cov)).reshape(m.shape)
        l_bnd = m - spread / 2
        for ii in range(0, num_parts):
            part = gdistrib.Particle()
            part.point = l_bnd + spread * rng.random(m.shape)
            w = 1 / num_parts
            distrib.add_particle(part, w)
        birth_terms.append(distrib)
    return birth_terms


def _setup_imm_pmbm_ct_ktr_birth():
    mu = [np.array([10.0, 0.0, 1, 2, 0.0]).reshape((5, 1))]
    cov = [np.diag(np.array([1, 1, 0.1, 0.1, 3 * np.pi / 180])) ** 2]
    # cov = [np.diag(np.array([1, 1, 1, 1, np.pi])) ** 2]
    gm0 = smodels.GaussianMixture(means=mu, covariances=cov, weights=[1])
    return [gm0]


def _gen_meas(tt, true_agents, proc_noise, meas_noise, rng):
    meas_in = []
    for x in true_agents:
        xp = rng.multivariate_normal(x.flatten(), proc_noise).reshape(x.shape)
        meas = _meas_mat @ xp
        m = rng.multivariate_normal(meas.flatten(), meas_noise).reshape(meas.shape)
        meas_in.append(m.copy())
    return meas_in


def _gen_meas_stf(tt, true_agents, proc_noise, p_df, meas_noise, m_df, rng):
    meas_in = []
    for x in true_agents:
        xp = stats.multivariate_t.rvs(
            df=p_df, loc=x.flatten(), shape=proc_noise, random_state=rng
        ).reshape(x.shape)
        meas = _meas_mat @ xp
        m = stats.multivariate_t.rvs(
            df=m_df, loc=meas.flatten(), shape=meas_noise, random_state=rng
        ).reshape(meas.shape)
        meas_in.append(m.copy())
    return meas_in


def _gen_meas_qkf(tt, true_agents, proc_noise, meas_fun, m_vars, rng):
    meas_out = []
    for x in true_agents:
        xp = rng.multivariate_normal(x.flatten(), proc_noise).reshape(x.shape)
        meas = meas_fun(tt, xp)
        for ii, var in enumerate(m_vars):
            meas[ii, 0] += stats.norm.rvs(scale=np.sqrt(var), random_state=rng)
        meas_out.append(meas.copy())
    return meas_out


def _gen_meas_gsm(tt, true_agents, proc_noise, meas_fun, m_dfs, m_vars, rng):
    meas_out = []
    for x in true_agents:
        xp = rng.multivariate_normal(x.flatten(), proc_noise).reshape(x.shape)
        meas = meas_fun(tt, xp)
        for ii, (df, var) in enumerate(zip(m_dfs, m_vars)):
            meas[ii, 0] += stats.t.rvs(df, scale=np.sqrt(var), random_state=rng)
        meas_out.append(meas.copy())
    return meas_out


def _gen_meas_imm(tt, true_agents, proc_noise, meas_noise, rng):
    meas_in = []
    m_mat = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]])
    # m_mat = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    for x in true_agents:
        # xp = rng.multivariate_normal(x.flatten(), proc_noise).reshape(x.shape)
        # meas = m_mat @ xp
        meas = m_mat @ x
        # m = rng.multivariate_normal(meas.flatten(), meas_noise).reshape(meas.shape)
        # meas_in.append(m.copy())
        meas_in.append(meas.copy())
    return meas_in


def _gen_meas_ms(tt, true_agents, proc_noise, meas_noise, rng, meas_model_list):
    meas_in = []
    for model in meas_model_list:
        sens_list = []
        for x in true_agents:
            sens_list.append(model @ x)
        meas_in.append(sens_list)

    return meas_in


def _gen_meas_ms2(tt, true_agents, proc_noise, meas_noise, rng, meas_model_list):
    meas_in = []
    for ii, model in enumerate(meas_model_list):
        sens_list = []
        for x in true_agents:
            if ii == 0:
                if x[0] > -5 and x[0] < 25:
                    if isinstance(model, list):
                        cur_temp_meas = []
                        for func in model:
                            cur_temp_meas.append(func(tt, x))
                        sens_list.append(np.array(cur_temp_meas).reshape(-1, 1))
                    else:
                        sens_list.append(model @ x)
                # else:
                #     flag = False
                #     for arr in sens_list:
                #         if arr.size == 0:
                #             flag = True
                #     if flag == True:
                # sens_list.append(np.array([]))
            else:
                if x[0] > 2 and x[0] < 32:
                    if isinstance(model, list):
                        cur_temp_meas = []
                        for func in model:
                            cur_temp_meas.append(func(tt, x))
                        sens_list.append(np.array(cur_temp_meas).reshape(-1, 1))
                    else:
                        sens_list.append(model @ x)
                # else:
                # flag = False
                # for arr in sens_list:
                #     if arr.size == 0:
                #         flag = True
                # if flag == True:
                # sens_list.append(np.array([]))
        # sens_list.append(np.array([]))
        if len(sens_list) == 0:
            np.array([])
        np.random.shuffle(sens_list)
        meas_in.append(sens_list)

    return meas_in


def _gen_meas_ms_ekf(tt, true_agents, proc_noise, meas_noise, rng, meas_model_list):
    def range_func(t, x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2)

    def bear_func(t, x):
        return np.arctan2(x[1], x[0])

    def range_func2(t, x):
        return np.sqrt(x[0] ** 2 + x[1] + 5**2)

    def bear_func2(t, x):
        return np.arctan2(x[1], x[0] + 5)

    sens_bounds = [[0, 15], [5, 20]]
    meas_in = []
    for ii, model in enumerate(meas_model_list):
        cur_sens_lst = []
        for x in true_agents:
            if x[0] > sens_bounds[ii][0] and x[0] < sens_bounds[ii][1]:
                meas1 = meas_model_list[ii][0](0, x)
                meas2 = meas_model_list[ii][1](0, x)
                cur_sens_lst.append(np.array([meas1, meas2]))
            else:
                cur_sens_lst.append(np.array([]))
        meas_in.append(cur_sens_lst)

    return meas_in


def _prop_true(true_agents, tt, dt):
    out = []
    for ii, x in enumerate(true_agents):
        out.append(_state_mat_fun(tt, dt, "useless") @ x)
    return out


def _update_true_agents(true_agents, tt, dt, b_model, rng):
    out = _prop_true(true_agents, tt, dt)

    if any(np.abs(tt - np.array([0, 1, 1.5])) < 1e-8):
        x = b_model[0].means[0] + (rng.standard_normal(4) * np.ones(4)).reshape((4, 1))
        out.append(x.copy())
    return out


def _update_true_agents_phd_spawn(true_agents, tt, dt, b_model, rng):
    out = _prop_true(true_agents, tt, dt)

    if any(np.abs(tt - np.array([0, 1, 1.5])) < 1e-8):
        x = b_model[0].means[0] + (rng.standard_normal(4) * np.ones(4)).reshape((4, 1))
        out.append(x.copy())

    if any(np.abs(tt - np.array([2.5, 3.5])) < 1e-8):
        y = true_agents[0] + (rng.standard_normal(4) * np.ones(4)).reshape((4, 1))
        out.append(y.copy())
    return out


def _update_true_agents_prob(true_agents, tt, dt, b_model, rng):
    out = _prop_true(true_agents, tt, dt)

    p = rng.uniform()
    for gm, w in b_model:
        if p <= w:
            print("birth at {:.2f}".format(tt))
            x = gm.means[0] + (1 * rng.standard_normal(4)).reshape((4, 1))
            out.append(x.copy())
    return out


def _update_true_agents_prob2(true_agents, tt, dt, b_model, rng):
    out = _prop_true(true_agents, tt, dt)

    p = rng.uniform()
    for gm, w in b_model:
        if any(np.abs(tt - np.array([0, 1, 1.5])) < 1e-8):
            print("birth at {:.2f}".format(tt))
            x = gm.means[0] + (1 * rng.standard_normal(4)).reshape((4, 1))
            out.append(x.copy())
    return out


def _update_true_agents_prob_smc(true_agents, tt, dt, b_model, rng):
    out = []
    doubleInt = gdyn.DoubleIntegrator()
    for ii, x in enumerate(true_agents):
        out.append(doubleInt.get_state_mat(tt, dt) @ x)
    if any(np.abs(tt - np.array([0.5])) < 1e-8):
        for distrib, w in b_model:
            print("birth at {:.2f}".format(tt))
            inds = np.arange(0, len(distrib.particles))
            ii = rnd.choice(inds, p=distrib.weights)
            out.append(distrib.particles[ii].copy())
    return out


def _update_true_agents_prob_usmc(true_agents, tt, dt, b_model, rng):
    out = []
    doubleInt = gdyn.DoubleIntegrator()
    for ii, x in enumerate(true_agents):
        out.append(doubleInt.get_state_mat(tt, dt) @ x)
    if any(np.abs(tt - np.array([0.5])) < 1e-8):
        for distrib, w in b_model:
            print("birth at {:.2f}".format(tt))
            out.append(distrib.mean.copy())
    return out


def _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat):
    out = []
    for existing in true_agents:
        out.append(state_mat @ existing)
    if any(np.abs(tt - np.array([5])) < 1e-8):
        print("birth at {:.2f}".format(tt))
        gm = b_model[0][0]
        out.append(gm.means[0].copy().reshape(gm.means[0].shape))
    return out


def _update_true_agents_imm(true_agents, tt, dt, b_model, rng, state_mat):
    # out = _prop_true(true_agents, tt, dt)
    out = []
    for x in true_agents:
        out.append(state_mat @ x)

    if any(np.abs(tt - np.array([0, 1, 1.5])) < 1e-8):
        # if any(np.abs(tt - np.array([0])) < 1e-8):
        x = b_model[0].means[0] + (
            rng.standard_normal(5) * np.array([1, 1, 1, 1, 3 * np.pi / 180])
        ).reshape((5, 1))
        out.append(x.copy())
    return out


def _update_true_agents_prob_imm(true_agents, tt, dt, b_model, rng, state_mat):
    # out = _prop_true(true_agents, tt, dt)
    out = []
    for x in true_agents:
        out.append(state_mat @ x)

    p = rng.uniform()
    for gm, w in b_model:
        if p <= w:
            print("birth at {:.2f}".format(tt))
            x = gm.means[0] + np.array([1, 1, 1, 1, 3 * np.pi / 180]).reshape(
                (5, 1)
            ) * (rng.standard_normal(5)).reshape((5, 1))
            out.append(x.copy())
    return out


def _update_true_agents_pmbm_lmb_var(true_agents, tt, dt, b_model, rng):
    out = _prop_true(true_agents, tt, dt)
    # if any(np.abs(tt - np.array([0, 1, 1.5, 2.5, 3])) < 1e-8):
    if any(np.abs(tt - np.array([0, 1])) < 1e-8):
        for gm, w in b_model:
            x = gm.means[0] + (rng.standard_normal(4) * np.ones(4)).reshape((4, 1))
            out.append(x.copy())

    # if any(np.abs(tt - np.array([2])) < 1e-8):
    #     out.pop(1)

    # if any(np.abs(tt - np.array([3])) < 1e-8):
    #     out.pop(1)

    return out


def _update_true_agents_pmbm(true_agents, tt, dt, b_model, rng):
    out = _prop_true(true_agents, tt, dt)
    # if any(np.abs(tt - np.array([0, 0.05]))<1e-8):
    if any(np.abs(tt - np.array([0, 1])) < 1e-8):
        # if any(np.abs(tt - np.array([0, 1, 1.5, 2.5, 3])) < 1e-8):
        for gm in b_model:
            noise = (rng.standard_normal(4) * np.ones(4)).reshape((4, 1))
            # noise[0] = noise[0] / 100
            # noise[1] = noise[1] / 100

            x = gm.means[0] + noise
            out.append(x.copy())
    # if any(np.abs(tt - np.array([2])) < 1e-8):
    #     out.pop(1)

    # if any(np.abs(tt - np.array([3])) < 1e-8):
    #     out.pop(1)

    return out


def _update_true_agents_imm_pmbm(true_agents, tt, dt, b_model, rng, state_mat):
    out = []
    for x in true_agents:
        out.append(state_mat @ x)
    if any(np.abs(tt - np.array([0, 1.0])) < 1e-8):
        # if any(np.abs(tt - np.array([0])) < 1e-8):
        for gm in b_model:
            x = gm.means[0] + np.array([2, 2, 0.01, 0.01, 3 * np.pi / 180]).reshape(
                (5, 1)
            ) * (rng.standard_normal(5)).reshape((5, 1))
            out.append(x.copy())
    return out


def test_PHD():  # noqa
    print("Test PHD")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 10 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_phd_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    phd = tracker.ProbabilityHypothesisDensity(**RFS_base_args)
    phd.gating_on = False
    phd.save_covs = True

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    for kk, tt in enumerate(time):
        true_agents = _update_true_agents(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        filt_args = {"state_mat_args": state_mat_args}
        phd.predict(tt, filt_args=filt_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        print(meas_in)

        filt_args = {"meas_fun_args": meas_fun_args}
        phd.correct(
            tt, meas_in, meas_mat_args={}, est_meas_args={}, filt_args=filt_args
        )

        phd.cleanup(enable_merge=True)
    true_covs = []
    for ii, lst in enumerate(global_true):
        true_covs.append([])
        for jj in lst:
            true_covs[ii].append(np.diag([7e-5, 7e-5, 0.1, 0.1]))
    phd.calculate_ospa(global_true, 5, 1)
    if debug_plots:
        phd.plot_ospa_history(time=time, time_units="s")
    phd.calculate_ospa(global_true, 5, 1, core_method=SingleObjectDistance.MANHATTAN)
    if debug_plots:
        phd.plot_ospa_history(time=time, time_units="s")
    phd.calculate_ospa(
        global_true,
        1,
        1,
        core_method=SingleObjectDistance.HELLINGER,
        true_covs=true_covs,
    )
    if debug_plots:
        phd.plot_ospa_history(time=time, time_units="s")
    phd.calculate_ospa(global_true, 5, 1, core_method=SingleObjectDistance.MAHALANOBIS)
    if debug_plots:
        phd.plot_ospa_history(time=time, time_units="s")
    if debug_plots:
        phd.plot_states([0, 1])
    assert len(true_agents) == phd.cardinality, "Wrong cardinality"


def test_PHD_spawning():
    print("Test PHD with Target Spawning")
    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 10 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_phd_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    phd = tracker.ProbabilityHypothesisDensity(**RFS_base_args)
    phd.gating_on = False
    phd.save_covs = True
    phd.enable_spawning = True
    phd.spawn_cov = np.diag([1.0, 1.0, 5.0, 5.0])
    phd.spawn_weight = 1

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    for kk, tt in enumerate(time):
        true_agents = _update_true_agents_phd_spawn(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        filt_args = {"state_mat_args": state_mat_args}
        phd.predict(tt, filt_args=filt_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        filt_args = {"meas_fun_args": meas_fun_args}
        phd.correct(
            tt, meas_in, meas_mat_args={}, est_meas_args={}, filt_args=filt_args
        )

        phd.cleanup()
    true_covs = []
    for ii, lst in enumerate(global_true):
        true_covs.append([])
        for jj in lst:
            true_covs[ii].append(np.diag([7e-5, 7e-5, 0.1, 0.1]))
    phd.calculate_ospa(global_true, 5, 1)
    if debug_plots:
        phd.plot_ospa_history(time=time, time_units="s")
    phd.calculate_ospa(global_true, 5, 1, core_method=SingleObjectDistance.MANHATTAN)
    if debug_plots:
        phd.plot_ospa_history(time=time, time_units="s")
    # phd.calculate_ospa(
    #     global_true,
    #     1,
    #     1,
    #     core_method=SingleObjectDistance.HELLINGER,
    #     true_covs=true_covs,
    # )
    # if debug_plots:
    #     phd.plot_ospa_history(time=time, time_units="s")
    # phd.calculate_ospa(global_true, 5, 1, core_method=SingleObjectDistance.MAHALANOBIS)
    # if debug_plots:
    #     phd.plot_ospa_history(time=time, time_units="s")
    if debug_plots:
        phd.plot_states([0, 1])
    assert len(true_agents) == phd.cardinality, "Wrong cardinality"


def test_CPHD():  # noqa
    print("Test CPHD")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 10 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_phd_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    phd = tracker.CardinalizedPHD(**RFS_base_args)
    phd.gating_on = False

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    for kk, tt in enumerate(time):
        true_agents = _update_true_agents(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        filt_args = {"state_mat_args": state_mat_args}
        phd.predict(tt, filt_args=filt_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        filt_args = {"meas_fun_args": meas_fun_args}
        phd.correct(
            tt, meas_in, meas_mat_args={}, est_meas_args={}, filt_args=filt_args
        )

        phd.cleanup()
    phd.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        phd.plot_card_history(time_vec=time)
        phd.plot_states([0, 1])
        phd.plot_ospa_history(time=time, time_units="s")
    assert len(true_agents) == phd.cardinality, "Wrong cardinality"


def test_CPHD_spawning():  # noqa
    print("Test CPHD with Target Spawning")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 10 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_phd_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    phd = tracker.CardinalizedPHD(**RFS_base_args)
    phd.gating_on = False
    phd.enable_spawning = True
    phd.spawn_cov = np.diag([1.0, 1.0, 5.0, 5.0])
    phd.spawn_weight = 1

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    for kk, tt in enumerate(time):
        true_agents = _update_true_agents_phd_spawn(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        filt_args = {"state_mat_args": state_mat_args}
        phd.predict(tt, filt_args=filt_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        filt_args = {"meas_fun_args": meas_fun_args}
        phd.correct(
            tt, meas_in, meas_mat_args={}, est_meas_args={}, filt_args=filt_args
        )

        phd.cleanup()
    phd.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        phd.plot_card_history(time_vec=time)
        phd.plot_states([0, 1])
        phd.plot_ospa_history(time=time, time_units="s")
    assert len(true_agents) == phd.cardinality, "Wrong cardinality"


def test_IMM_PHD():
    print("Test IMM-PHD")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 10 + dt

    filt = _setup_imm_ctktr_kf(dt)
    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    # change for imm???
    b_model = _setup_imm_phd_ct_ktr_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    phd = tracker.IMMProbabilityHypothesisDensity(**RFS_base_args)
    phd.gating_on = False
    # phd.prune_threshold = 1e-2
    # phd.extract_threshold = phd.prune_threshold

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    state_mat = gdyn.CoordinatedTurnKnown(turn_rate=0 * np.pi / 180).get_state_mat(
        0, dt
    )
    saved_times = []
    # fig = plt.figure()
    # fig.add_subplot(111)
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if tt > 5:
            state_mat = gdyn.CoordinatedTurnKnown(
                turn_rate=60 * np.pi / 180
            ).get_state_mat(tt, dt)

        true_agents = _update_true_agents_imm(
            true_agents, tt, dt, b_model, rng, state_mat
        )
        global_true.append(deepcopy(true_agents))

        filt_args = {"state_mat_args": state_mat_args}
        phd.predict(tt, filt_args=filt_args)

        meas_in = _gen_meas_imm(
            tt,
            true_agents,
            filt.in_filt_list[0].proc_noise * 1e-3,
            filt.in_filt_list[0].meas_noise * 1e-3,
            rng,
        )

        filt_args = {"meas_fun_args": meas_fun_args}
        phd.correct(
            tt, meas_in, meas_mat_args={}, est_meas_args={}, filt_args=filt_args
        )

        phd.cleanup(enable_merge=True)
        # fig.axes[0].scatter(true_agents[0][0, 0], true_agents[0][1, 0], color='k', marker='*')
        # phd.plot_states([0, 1], f_hndl=fig, meas_inds=[0, 1])
        # plt.pause(0.003)
        # print(len(phd._gaussMix.means))
        if phd.cardinality > 0:
            saved_times.append(tt)

    true_covs = []
    for ii, lst in enumerate(global_true):
        true_covs.append([])
        for jj in lst:
            true_covs[ii].append(np.diag([7e-5, 7e-5, 0.1, 0.1, 1.0 * np.pi / 180.0]))
    # phd.calculate_ospa(global_true, 5, 1)
    # if debug_plots:
    #     phd.plot_ospa_history(time=time, time_units="s")
    # phd.calculate_ospa(global_true, 5, 1, core_method=SingleObjectDistance.MANHATTAN)
    # if debug_plots:
    #     phd.plot_ospa_history(time=time, time_units="s")
    # phd.calculate_ospa(
    #     global_true,
    #     1,
    #     1,
    #     core_method=SingleObjectDistance.HELLINGER,
    #     true_covs=true_covs,
    # )
    # if debug_plots:
    #     phd.plot_ospa_history(time=time, time_units="s")
    # phd.calculate_ospa(global_true, 5, 1, core_method=SingleObjectDistance.MAHALANOBIS)
    # if debug_plots:
    #     phd.plot_ospa_history(time=time, time_units="s")
    if debug_plots:
        phd.plot_states([0, 1])
    # assert len(true_agents) == phd.cardinality, "Wrong cardinality"


def test_IMM_CPHD():
    print("Test IMM-CPHD")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 10 + dt

    filt = _setup_imm_ctktr_kf(dt)
    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    # change for imm???
    b_model = _setup_imm_phd_ct_ktr_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    phd = tracker.IMMCardinalizedPHD(**RFS_base_args)
    phd.gating_on = False

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    state_mat = gdyn.CoordinatedTurnKnown(turn_rate=0 * np.pi / 180).get_state_mat(
        0, dt
    )

    # fig = plt.figure()
    # fig.add_subplot(111)

    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if tt > 5:
            state_mat = gdyn.CoordinatedTurnKnown(
                turn_rate=5 * np.pi / 180
            ).get_state_mat(tt, dt)
        if np.mod(kk, 588) == 0:
            asdfhpoa = 123
        true_agents = _update_true_agents_imm(
            true_agents, tt, dt, b_model, rng, state_mat
        )
        global_true.append(deepcopy(true_agents))

        filt_args = {"state_mat_args": state_mat_args}
        phd.predict(tt, filt_args=filt_args)

        meas_in = _gen_meas_imm(
            tt,
            true_agents,
            filt.in_filt_list[0].proc_noise * 1e-3,
            filt.in_filt_list[0].meas_noise * 1e-3,
            rng,
        )

        filt_args = {"meas_fun_args": meas_fun_args}
        phd.correct(
            tt, meas_in, meas_mat_args={}, est_meas_args={}, filt_args=filt_args
        )

        phd.cleanup()
        # if np.mod(kk, 50) == 0:
        #     fig.axes[0].scatter(true_agents[0][0, 0], true_agents[0][1, 0], color='k', marker='*')
        #     phd.plot_states([0, 1], f_hndl=fig, meas_inds=[0, 1])
        #     plt.pause(0.003)

    phd.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        phd.plot_card_history(time_vec=time)
        phd.plot_states([0, 1])
        phd.plot_ospa_history(time=time, time_units="s")
    # assert len(true_agents) == phd.cardinality, "Wrong cardinality"


def test_GLMB():  # noqa
    print("Test GM-GLMB")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 6 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_gm_glmb_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    glmb = tracker.GeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)
    glmb.save_covs = True

    # test save/load filter
    filt_state = glmb.save_filter_state()
    glmb = tracker.GeneralizedLabeledMultiBernoulli()
    glmb.load_filter_state(filt_state)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
        # true_agents = _update_true_agents_prob2(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        glmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {"meas_fun_args": meas_fun_args}
        glmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    glmb.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.EUCLIDEAN
    )
    if debug_plots:
        glmb.plot_ospa2_history(time=time, time_units="s")
    glmb.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.MAHALANOBIS
    )
    if debug_plots:
        glmb.plot_ospa2_history(time=time, time_units="s")
    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_STM_GLMB():  # noqa
    print("Test STM-GLMB")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 6 + dt

    filt = _setup_double_int_stf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_stm_glmb_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.99,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    glmb = tracker.STMGeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        glmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_stf(
            tt,
            true_agents,
            filt.proc_noise,
            filt.proc_noise_dof,
            filt.meas_noise,
            filt.meas_noise_dof,
            rng,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        glmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        glmb.plot_card_dist()
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


@pytest.mark.slow
def test_SMC_GLMB():  # noqa
    print("Test SMC-GLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 1000
    prob_detection = 0.99
    prob_survive = 0.98

    filt = _setup_double_int_pf(dt, filt_rng)
    meas_fun_args = ()
    dyn_fun_params = (dt,)

    b_model = _setup_smc_glmb_double_int_birth(num_parts, rng)
    other_bm = _setup_gm_glmb_double_int_birth()

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    SMC_args = {
        "compute_prob_detection": compute_prob_detection,
        "compute_prob_survive": compute_prob_survive,
    }
    glmb = tracker.SMCGeneralizedLabeledMultiBernoulli(
        **SMC_args, **GLMB_args, **RFS_base_args
    )

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        # true_agents = _update_true_agents(true_agents, tt, dt, other_bm[0], rng)
        true_agents = _update_true_agents_prob_smc(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"dyn_fun_params": dyn_fun_params}
        prob_surv_args = (prob_survive,)
        glmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {"meas_fun_args": meas_fun_args}
        prob_det_args = (prob_detection,)
        glmb.correct(tt, meas_in, prob_det_args=prob_det_args, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_USMC_GLMB():  # noqa
    print("Test USMC-GLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 75
    prob_detection = 0.99
    prob_survive = 0.98
    use_MCMC = False

    filt = _setup_double_int_upf(dt, filt_rng, use_MCMC)
    meas_fun_args = ()
    dyn_fun_params = (dt,)

    b_model = _setup_usmc_glmb_double_int_birth(num_parts, rng)

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    SMC_args = {
        "compute_prob_detection": compute_prob_detection,
        "compute_prob_survive": compute_prob_survive,
    }
    glmb = tracker.SMCGeneralizedLabeledMultiBernoulli(
        **SMC_args, **GLMB_args, **RFS_base_args
    )

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_prob_usmc(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        prob_surv_args = (prob_survive,)
        ukf_kwargs_pred = {"state_mat_args": dyn_fun_params}
        filt_args_pred = {"ukf_kwargs": ukf_kwargs_pred}
        glmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=filt_args_pred)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        prob_det_args = (prob_detection,)
        ukf_kwargs_cor = {"meas_fun_args": meas_fun_args}
        filt_args_cor = {"ukf_kwargs": ukf_kwargs_cor}
        glmb.correct(tt, meas_in, prob_det_args=prob_det_args, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_MCMC_USMC_GLMB():  # noqa
    print("Test MCMC USMC-GLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 30
    prob_detection = 0.99
    prob_survive = 0.98
    use_MCMC = True

    filt = _setup_double_int_upf(dt, filt_rng, use_MCMC)
    meas_fun_args = ()
    dyn_fun_params = (dt,)

    b_model = _setup_usmc_glmb_double_int_birth(num_parts, rng)

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    SMC_args = {
        "compute_prob_detection": compute_prob_detection,
        "compute_prob_survive": compute_prob_survive,
    }
    glmb = tracker.SMCGeneralizedLabeledMultiBernoulli(
        **SMC_args, **GLMB_args, **RFS_base_args
    )

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_prob_usmc(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        prob_surv_args = (prob_survive,)
        ukf_kwargs_pred = {"state_mat_args": dyn_fun_params}
        filt_args_pred = {"ukf_kwargs": ukf_kwargs_pred}
        glmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=filt_args_pred)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        prob_det_args = (prob_detection,)
        ukf_kwargs_cor = {"meas_fun_args": meas_fun_args}
        filt_args_cor = {"ukf_kwargs": ukf_kwargs_cor}
        glmb.correct(tt, meas_in, prob_det_args=prob_det_args, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("max cardinality {}".format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_JGLMB():  # noqa
    print("Test GM-JGLMB")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 6 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_gm_glmb_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1**-3,
        "clutter_rate": 1**-3,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.JointGeneralizedLabeledMultiBernoulli(**JGLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_pmbm_lmb_var(
            true_agents, tt, dt, b_model, rng
        )
        # true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        jglmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)
        np.random.shuffle(meas_in)

        cor_args = {"meas_fun_args": meas_fun_args}
        jglmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    jglmb.calculate_ospa(global_true, 2, 1)
    jglmb.calculate_ospa2(global_true, 5, 1, 100)
    jglmb.calculate_gospa(global_true, 2, 1, 2)

    true_agents = _update_true_agents_pmbm_lmb_var(true_agents, tt, dt, b_model, rng)
    # true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
    global_true.append(deepcopy(true_agents))

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
        jglmb.plot_ospa_history()
        jglmb.plot_ospa2_history()
        jglmb.plot_gospa_history()
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


def test_JGLMB_high_birth():  # noqa
    print("Test GM-JGLMB high birth")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 6 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_gm_glmb_double_int_birth()
    b_model[0] = (b_model[0][0], 0.007)  # increase birth prob for this test

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1**-3,
        "clutter_rate": 1**-3,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.JointGeneralizedLabeledMultiBernoulli(**JGLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        # true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
        true_agents = _update_true_agents_prob2(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        jglmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {"meas_fun_args": meas_fun_args}
        jglmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    jglmb.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
        jglmb.plot_ospa_history()
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


def test_STM_JGLMB():  # noqa
    print("Test STM-JGLMB")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 4 + dt

    filt = _setup_double_int_stf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_stm_glmb_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1**-3,
        "clutter_rate": 1**-3,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.STMJointGeneralizedLabeledMultiBernoulli(
        **JGLMB_args, **RFS_base_args
    )

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        jglmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_stf(
            tt,
            true_agents,
            filt.proc_noise,
            filt.proc_noise_dof,
            filt.meas_noise,
            filt.meas_noise_dof,
            rng,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        jglmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    jglmb.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        jglmb.plot_ospa_history()
        jglmb.plot_card_dist()
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


@pytest.mark.slow
def test_SMC_JGLMB():  # noqa
    print("Test SMC-JGLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 1000
    prob_detection = 0.99
    prob_survive = 0.98

    filt = _setup_double_int_pf(dt, filt_rng)
    meas_fun_args = ()
    dyn_fun_params = (dt,)

    b_model = _setup_smc_glmb_double_int_birth(num_parts, rng)

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    SMC_args = {
        "compute_prob_detection": compute_prob_detection,
        "compute_prob_survive": compute_prob_survive,
    }
    jglmb = tracker.SMCJointGeneralizedLabeledMultiBernoulli(
        **SMC_args, **JGLMB_args, **RFS_base_args
    )

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_prob_smc(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"dyn_fun_params": dyn_fun_params}
        prob_surv_args = (prob_survive,)
        jglmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {"meas_fun_args": meas_fun_args}
        prob_det_args = (prob_detection,)
        jglmb.correct(tt, meas_in, prob_det_args=prob_det_args, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


def test_USMC_JGLMB():  # noqa
    print("Test USMC-JGLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 75
    prob_detection = 0.99
    prob_survive = 0.98
    use_MCMC = False

    filt = _setup_double_int_upf(dt, filt_rng, use_MCMC)
    meas_fun_args = ()
    dyn_fun_params = (dt,)

    b_model = _setup_usmc_glmb_double_int_birth(num_parts, rng)

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    SMC_args = {
        "compute_prob_detection": compute_prob_detection,
        "compute_prob_survive": compute_prob_survive,
    }
    jglmb = tracker.SMCJointGeneralizedLabeledMultiBernoulli(
        **SMC_args, **JGLMB_args, **RFS_base_args
    )

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_prob_usmc(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        prob_surv_args = (prob_survive,)
        ukf_kwargs_pred = {"state_mat_args": dyn_fun_params}
        filt_args_pred = {"ukf_kwargs": ukf_kwargs_pred}
        jglmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=filt_args_pred)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        prob_det_args = (prob_detection,)
        ukf_kwargs_cor = {"meas_fun_args": meas_fun_args}
        filt_args_cor = {"ukf_kwargs": ukf_kwargs_cor}
        jglmb.correct(tt, meas_in, prob_det_args=prob_det_args, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


def test_MCMC_USMC_JGLMB():  # noqa
    print("Test MCMC USMC-JGLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 30
    prob_detection = 0.99
    prob_survive = 0.98
    use_MCMC = True

    filt = _setup_double_int_upf(dt, filt_rng, use_MCMC)
    meas_fun_args = ()
    dyn_fun_params = (dt,)

    b_model = _setup_usmc_glmb_double_int_birth(num_parts, rng)

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    SMC_args = {
        "compute_prob_detection": compute_prob_detection,
        "compute_prob_survive": compute_prob_survive,
    }
    jglmb = tracker.SMCJointGeneralizedLabeledMultiBernoulli(
        **SMC_args, **JGLMB_args, **RFS_base_args
    )

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_prob_usmc(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        prob_surv_args = (prob_survive,)
        ukf_kwargs_pred = {"state_mat_args": dyn_fun_params}
        filt_args_pred = {"ukf_kwargs": ukf_kwargs_pred}
        jglmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=filt_args_pred)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        prob_det_args = (prob_detection,)
        ukf_kwargs_cor = {"meas_fun_args": meas_fun_args}
        filt_args_cor = {"ukf_kwargs": ukf_kwargs_cor}
        jglmb.correct(tt, meas_in, prob_det_args=prob_det_args, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("max cardinality {}".format(np.max([len(s_set) for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


def test_QKF_JGLMB():  # noqa
    print("Test QKF-JGLMB")

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = False
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_qkf(dt, use_sqkf, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.JointGeneralizedLabeledMultiBernoulli(**JGLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun, m_vars, rng)

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


def test_SQKF_JGLMB():  # noqa
    print("Test SQKF-JGLMB")

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = True
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_qkf(dt, use_sqkf, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.GeneralizedLabeledMultiBernoulli(**JGLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun, m_vars, rng)

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("max cardinality {}".format(np.max([len(s_set) for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


def test_UKF_JGLMB():  # noqa
    print("Test UKF-JGLMB")

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_ukf(dt, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.JointGeneralizedLabeledMultiBernoulli(**JGLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun, m_vars, rng)

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


def test_QKF_GSM_JGLMB():  # noqa
    print("Test QKF GSM-JGLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = False
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_qkf_gsm(dt, filt_rng, use_sqkf, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.GSMJointGeneralizedLabeledMultiBernoulli(
        **JGLMB_args, **RFS_base_args
    )

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(
            tt, true_agents, filt.proc_noise, meas_fun, m_dfs, m_vars, rng
        )

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


def test_SQKF_GSM_JGLMB():  # noqa
    print("Test SQKF GSM-JGLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = True
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_qkf_gsm(dt, filt_rng, use_sqkf, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.GSMJointGeneralizedLabeledMultiBernoulli(
        **JGLMB_args, **RFS_base_args
    )

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(
            tt, true_agents, filt.proc_noise, meas_fun, m_dfs, m_vars, rng
        )

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


def test_UKF_GSM_JGLMB():  # noqa
    print("Test UKF GSM-JGLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_ukf_gsm(dt, filt_rng, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.GSMJointGeneralizedLabeledMultiBernoulli(
        **JGLMB_args, **RFS_base_args
    )

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(
            tt, true_agents, filt.proc_noise, meas_fun, m_dfs, m_vars, rng
        )

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


def test_QKF_GLMB():  # noqa
    print("Test QKF-GLMB")

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = False
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_qkf(dt, use_sqkf, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    glmb = tracker.GeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun, m_vars, rng)

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_SQKF_GLMB():  # noqa
    print("Test SQKF-GLMB")

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = True
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_qkf(dt, use_sqkf, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    glmb = tracker.GeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun, m_vars, rng)

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("max cardinality {}".format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_UKF_GLMB():  # noqa
    print("Test UKF-GLMB")

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_ukf(dt, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    glmb = tracker.GeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun, m_vars, rng)

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_QKF_GSM_GLMB():  # noqa
    print("Test QKF GSM-GLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = False
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_qkf_gsm(dt, filt_rng, use_sqkf, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    glmb = tracker.GSMGeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)

    # test save/load filter
    filt_state = glmb.save_filter_state()
    glmb = tracker.GSMGeneralizedLabeledMultiBernoulli()
    glmb.load_filter_state(filt_state)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(
            tt, true_agents, filt.proc_noise, meas_fun, m_dfs, m_vars, rng
        )

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_SQKF_GSM_GLMB():  # noqa
    print("Test SQKF GSM-GLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = True
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_qkf_gsm(dt, filt_rng, use_sqkf, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    glmb = tracker.GSMGeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(
            tt, true_agents, filt.proc_noise, meas_fun, m_dfs, m_vars, rng
        )

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_UKF_GSM_GLMB():  # noqa
    print("Test UKF GSM-GLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_ukf_gsm(dt, filt_rng, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    glmb = tracker.GSMGeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(
            tt, true_agents, filt.proc_noise, meas_fun, m_dfs, m_vars, rng
        )

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_EKF_GSM_GLMB():  # noqa
    print("Test EKF GSM-GLMB")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180) ** 2)

    filt, state_mat, meas_fun = _setup_ekf_gsm(dt, filt_rng, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    glmb = tracker.GSMGeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(
            tt, true_agents, filt.proc_noise, meas_fun, m_dfs, m_vars, rng
        )

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    print("\tmax cardinality {}".format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_GLMB_ct_ktr():  # noqa
    print("Test GLMB with CT-KTR dynamics")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 3.5 + dt

    filt = _setup_ctktr_kf(dt)
    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    b_model = _setup_imm_gm_glmb_ct_ktr_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    glmb = tracker.GeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)
    glmb.save_covs = True

    # test save/load filter
    filt_state = glmb.save_filter_state()
    glmb = tracker.GeneralizedLabeledMultiBernoulli()
    glmb.load_filter_state(filt_state)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    state_mat = gdyn.CoordinatedTurnKnown(turn_rate=0 * np.pi / 180).get_state_mat(
        0, dt
    )
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if tt > 5:
            state_mat = gdyn.CoordinatedTurnKnown(
                turn_rate=5 * np.pi / 180
            ).get_state_mat(tt, dt)

        true_agents = _update_true_agents_prob_imm(
            true_agents, tt, dt, b_model, rng, state_mat
        )
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        glmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_imm(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {"meas_fun_args": meas_fun_args}
        glmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    glmb.calculate_ospa(global_true, 2, 1)
    if debug_plots:
        glmb.plot_ospa_history(time=time, time_units="s")
    glmb.calculate_ospa2(global_true, 5, 1, 10)
    if debug_plots:
        glmb.plot_ospa2_history(time=time, time_units="s")
    glmb.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.EUCLIDEAN
    )
    if debug_plots:
        glmb.plot_ospa2_history(time=time, time_units="s")
    glmb.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.MAHALANOBIS
    )
    if debug_plots:
        glmb.plot_ospa2_history(time=time, time_units="s")
    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_IMM_GLMB():  # noqa
    print("Test IMM-GLMB")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 5.5 + dt

    filt = _setup_imm_ctktr_kf(dt)
    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    b_model = _setup_imm_gm_glmb_ct_ktr_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    glmb = tracker.IMMGeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)
    glmb.save_covs = True

    # test save/load filter
    filt_state = glmb.save_filter_state()
    glmb = tracker.IMMGeneralizedLabeledMultiBernoulli()
    glmb.load_filter_state(filt_state)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []

    state_mat = gdyn.CoordinatedTurnKnown(turn_rate=0 * np.pi / 180).get_state_mat(
        0, dt
    )

    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if tt > 5:
            state_mat = gdyn.CoordinatedTurnKnown(
                turn_rate=5 * np.pi / 180
            ).get_state_mat(tt, dt)

        true_agents = _update_true_agents_prob_imm(
            true_agents, tt, dt, b_model, rng, state_mat
        )
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        glmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_imm(
            tt,
            true_agents,
            filt.in_filt_list[0].proc_noise,
            filt.in_filt_list[0].meas_noise,
            rng,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        glmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        glmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    glmb.extract_states(**extract_kwargs)

    glmb.calculate_ospa(global_true, 2, 1)
    if debug_plots:
        glmb.plot_ospa_history(time=time, time_units="s")
    glmb.calculate_ospa2(global_true, 5, 1, 10)
    if debug_plots:
        glmb.plot_ospa2_history(time=time, time_units="s")
    glmb.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.EUCLIDEAN
    )
    if debug_plots:
        glmb.plot_ospa2_history(time=time, time_units="s")
    glmb.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.MAHALANOBIS
    )
    if debug_plots:
        glmb.plot_ospa2_history(time=time, time_units="s")
    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == glmb.cardinality, "Wrong cardinality"


def test_IMM_JGLMB():  # noqa
    print("Test IMM-JGLMB")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 5.5 + dt

    filt = _setup_imm_ctktr_kf(dt)
    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    b_model = _setup_imm_gm_glmb_ct_ktr_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1**-3,
        "clutter_rate": 1**-3,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.IMMJointGeneralizedLabeledMultiBernoulli(
        **JGLMB_args, **RFS_base_args
    )

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    state_mat = gdyn.CoordinatedTurnKnown(turn_rate=0 * np.pi / 180).get_state_mat(
        0, dt
    )
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if tt > 5:
            state_mat = gdyn.CoordinatedTurnKnown(
                turn_rate=5 * np.pi / 180
            ).get_state_mat(tt, dt)

        true_agents = _update_true_agents_prob_imm(
            true_agents, tt, dt, b_model, rng, state_mat
        )
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        jglmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_imm(
            tt,
            true_agents,
            filt.in_filt_list[0].proc_noise,
            filt.in_filt_list[0].meas_noise,
            rng,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        jglmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    jglmb.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units="s", time=time)
        jglmb.plot_ospa_history()
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


# @pytest.mark.slow
@pytest.mark.skip(reason="Unresolved Errors in Data Fusion")
def test_MS_JGLMB():  # noqa
    print("Test MS-GM-JGLMB")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 5.0 + dt
    # t0, t1 = 0, 2.0 + dt

    filt = _setup_double_int_gci_ekf(dt)
    # filt = _setup_double_int_gci_kf(dt)

    state_mat_args = (dt,)
    meas_fun_args = ()  # ("useless arg",)

    b_model = _setup_gm_glmb_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.97,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 10**-2,
        "clutter_rate": 10**-2,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.MSJointGeneralizedLabeledMultiBernoulli(
        **JGLMB_args, **RFS_base_args
    )
    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        # true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
        true_agents = _update_true_agents_pmbm_lmb_var(
            true_agents, tt, dt, b_model, rng
        )
        global_true.append(deepcopy(true_agents))

        pred_args = {"dyn_fun_params": state_mat_args}
        # pred_args = {"state_mat_args": state_mat_args}
        jglmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_ms2(
            # meas_in = _gen_meas_ms(
            tt,
            true_agents,
            filt.proc_noise,
            filt.meas_noise_list,
            rng,
            filt.meas_model_list,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        jglmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    jglmb.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        # plt.savefig("save_in_dumb_folder/msjglmb_state_lbl.png")
        jglmb.plot_card_dist()
        # plt.savefig("save_in_dumb_folder/msjglmb_card_dist.png")
        jglmb.plot_card_history(time_units="s", time=time)
        # plt.savefig("save_in_dumb_folder/msjglmb_card_hist.png")
        jglmb.plot_ospa_history()
        # plt.savefig("save_in_dumb_folder/msjglmb_ospa_hist.png")
    print("\tExpecting {} agents".format(len(true_agents)))

    # assert len(true_agents) == jglmb.cardinality, "Wrong cardinality"


@pytest.mark.slow
def test_MS_IMM_JGLMB():  # noqa
    print("Test MS-IMM-GM-JGLMB")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    # t0, t1 = 0, 5.5 + dt
    # t0, t1 = 0, 4 + dt
    t0, t1 = 0, 4 + dt

    # TODO GCI IMM FILT SETUP
    filt = _setup_ct_ktr_gci_imm_kf(dt)

    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    b_model = _setup_imm_gm_glmb_ct_ktr_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-3,
        "clutter_rate": 1e-3,
    }
    JGLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }
    jglmb = tracker.MSIMMJointGeneralizedLabeledMultiBernoulli(
        **JGLMB_args, **RFS_base_args
    )
    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    state_mat = gdyn.CoordinatedTurnKnown(turn_rate=0 * np.pi / 180).get_state_mat(
        0, dt
    )
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if tt > 3.75:
            state_mat = gdyn.CoordinatedTurnKnown(
                turn_rate=60 * np.pi / 180
            ).get_state_mat(tt, dt)

        true_agents = _update_true_agents_prob_imm(
            true_agents, tt, dt, b_model, rng, state_mat
        )
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        jglmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_ms(
            tt,
            true_agents,
            filt.in_filt_list[0].proc_noise,
            filt.in_filt_list[0].meas_noise,
            rng,
            filt.meas_model_list,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        jglmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    jglmb.extract_states(**extract_kwargs)

    jglmb.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        plt.savefig("save_in_dumb_folder/msimmjglmb_state_lbl.png")
        jglmb.plot_card_dist()
        plt.savefig("save_in_dumb_folder/msimmjglmb_card_dist.png")
        jglmb.plot_card_history(time_units="s", time=time)
        plt.savefig("save_in_dumb_folder/msimmjglmb_card_hist.png")
        jglmb.plot_ospa_history()
        plt.savefig("save_in_dumb_folder/msimmjglmb_ospa_hist.png")
    print("\tExpecting {} agents".format(len(true_agents)))


def test_PMBM():  # noqa
    print("Test PMBM")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 6 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_pmbm_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-5,
        "max_hyps": 1000,
    }
    pmbm = tracker.PoissonMultiBernoulliMixture(**PMBM_args, **RFS_base_args)
    pmbm.save_covs = True

    # test save/load filter
    filt_state = pmbm.save_filter_state()
    pmbm = tracker.PoissonMultiBernoulliMixture()
    pmbm.load_filter_state(filt_state)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_pmbm(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        pmbm.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {"meas_fun_args": meas_fun_args}
        pmbm.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        pmbm.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    pmbm.calculate_ospa(global_true, 2, 1)
    if debug_plots:
        pmbm.plot_ospa_history(time=time, time_units="s")
    pmbm.calculate_ospa2(global_true, 5, 1, 10)
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    pmbm.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.EUCLIDEAN
    )
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    pmbm.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.MAHALANOBIS
    )
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    if debug_plots:
        pmbm.plot_states([0, 1], true_states=global_true, meas_inds=[0, 1])
        pmbm.plot_card_dist()
        pmbm.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))

    # assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"


def test_LPMBM():  # noqa
    print("Test Labeled PMBM")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    # t0, t1 = 0, 1.5 + dt
    t0, t1 = 0, 6 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_pmbm_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-1,
        "clutter_rate": 1e-1,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-15,
        "max_hyps": 1000,
    }
    pmbm = tracker.LabeledPoissonMultiBernoulliMixture(**PMBM_args, **RFS_base_args)
    pmbm.save_covs = True

    # test save/load filter
    filt_state = pmbm.save_filter_state()
    pmbm = tracker.LabeledPoissonMultiBernoulliMixture()
    pmbm.load_filter_state(filt_state)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}, kk: {:.2f}".format(tt, kk))
            sys.stdout.flush()

        true_agents = _update_true_agents_pmbm(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        pmbm.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)
        np.random.shuffle(meas_in)

        cor_args = {"meas_fun_args": meas_fun_args}
        pmbm.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        pmbm.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    pmbm.calculate_ospa(global_true, 2, 1)
    if debug_plots:
        pmbm.plot_ospa_history(time=time, time_units="s")
    pmbm.calculate_ospa2(global_true, 5, 1, 10)
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    pmbm.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.EUCLIDEAN
    )
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    pmbm.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.MAHALANOBIS
    )
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    if debug_plots:
        # pmbm.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        pmbm.plot_states_labels([0, 1], true_states=global_true)
        pmbm.plot_card_dist()
        pmbm.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))
    plt.figure()
    all_xy = []
    for ii, time in enumerate(global_true):
        for agent in global_true[ii]:
            all_xy.append([agent[0], agent[1]])

    all_xy = np.array(all_xy)
    plt.plot(all_xy[:, 0], all_xy[:, 1], color="k", marker="o", linestyle="None")

    # assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"


def test_STM_PMBM():  # noqa
    print("Test STM-PMBM")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 2 + dt

    filt = _setup_double_int_stf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_stm_pmbm_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.99,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-5,
        "max_hyps": 1000,
    }
    pmbm = tracker.STMPoissonMultiBernoulliMixture(**PMBM_args, **RFS_base_args)
    pmbm.save_covs = True

    # test save/load filter
    filt_state = pmbm.save_filter_state()
    pmbm = tracker.STMPoissonMultiBernoulliMixture()
    pmbm.load_filter_state(filt_state)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if np.mod(kk, 1) == 0:
            print("\t\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if np.mod(kk, 253) == 0:
            test = 1

        true_agents = _update_true_agents_pmbm(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        pmbm.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_stf(
            tt,
            true_agents,
            filt.proc_noise,
            filt.proc_noise_dof,
            filt.meas_noise,
            filt.meas_noise_dof,
            rng,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        pmbm.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        pmbm.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    if debug_plots:
        pmbm.plot_states([0, 1], true_states=global_true, meas_inds=[0, 1])
        pmbm.plot_card_dist()
        pmbm.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"


def test_STM_LPMBM():  # noqa
    print("Test STM-LPMBM")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 2 + dt

    filt = _setup_double_int_stf(dt)
    state_mat_args = (dt, "test arg")
    meas_fun_args = ("useless arg",)

    b_model = _setup_stm_pmbm_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.99,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-15,
        "max_hyps": 1000,
    }
    pmbm = tracker.STMLabeledPoissonMultiBernoulliMixture(**PMBM_args, **RFS_base_args)
    pmbm.save_covs = True

    # test save/load filter

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_pmbm(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        pmbm.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_stf(
            tt,
            true_agents,
            filt.proc_noise,
            filt.proc_noise_dof,
            filt.meas_noise,
            filt.meas_noise_dof,
            rng,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        pmbm.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        pmbm.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    if debug_plots:
        pmbm.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        pmbm.plot_card_dist()
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"


@pytest.mark.slow
def test_SMC_PMBM():  # noqa
    print("Test SMC-PMBM")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1.2 + dt
    num_parts = 1000
    prob_detection = 0.99
    prob_survive = 0.98

    filt = _setup_double_int_pf(dt, filt_rng)
    meas_fun_args = ()
    dyn_fun_params = (dt,)

    b_model = _setup_smc_pmbm_double_int_birth(num_parts, rng)
    other_bm = _setup_pmbm_double_int_birth()

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-15,
        "max_hyps": 1000,
    }
    SMC_args = {
        "compute_prob_detection": compute_prob_detection,
        "compute_prob_survive": compute_prob_survive,
    }
    pmbm = tracker.SMCPoissonMultiBernoulliMixture(
        **SMC_args, **PMBM_args, **RFS_base_args
    )
    pmbm.save_covs = True

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_pmbm(true_agents, tt, dt, other_bm, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"dyn_fun_params": dyn_fun_params}
        prob_surv_args = (prob_survive,)
        pmbm.predict(tt, prob_surv_args, filt_args=pred_args)

        meas_in = _gen_meas(
            tt,
            true_agents,
            filt.proc_noise,
            filt.meas_noise,
            rng,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        prob_det_args = (prob_detection,)
        pmbm.correct(tt, meas_in, prob_det_args=prob_det_args, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        pmbm.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    if debug_plots:
        pmbm.plot_states([0, 1], true_states=global_true, meas_inds=[0, 1])
        pmbm.plot_card_dist()
        pmbm.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"


@pytest.mark.slow
def test_SMC_LPMBM():  # noqa
    print("Test SMC-LPMBM")

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1.2 + dt
    num_parts = 1000
    prob_detection = 0.99
    prob_survive = 0.98

    filt = _setup_double_int_pf(dt, filt_rng)
    meas_fun_args = ()
    dyn_fun_params = (dt,)

    b_model = _setup_smc_pmbm_double_int_birth(num_parts, rng)
    other_bm = _setup_pmbm_double_int_birth()

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {
        "prob_detection": prob_detection,
        "prob_survive": prob_survive,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-15,
        "max_hyps": 1000,
    }
    SMC_args = {
        "compute_prob_detection": compute_prob_detection,
        "compute_prob_survive": compute_prob_survive,
    }
    pmbm = tracker.SMCLabeledPoissonMultiBernoulliMixture(
        **SMC_args, **PMBM_args, **RFS_base_args
    )
    pmbm.save_covs = True

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_pmbm(true_agents, tt, dt, other_bm, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"dyn_fun_params": dyn_fun_params}
        prob_surv_args = (prob_survive,)
        pmbm.predict(tt, prob_surv_args, filt_args=pred_args)

        meas_in = _gen_meas(
            tt,
            true_agents,
            filt.proc_noise,
            filt.meas_noise,
            rng,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        prob_det_args = (prob_detection,)
        pmbm.correct(tt, meas_in, prob_det_args=prob_det_args, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        pmbm.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    if debug_plots:
        pmbm.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        pmbm.plot_card_dist()
        pmbm.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"


def test_IMM_PMBM():  # noqa
    print("Test IMM-PMBM")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 6 + dt  # 6 + dt

    filt = _setup_imm_ctktr_kf(dt)
    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    b_model = _setup_imm_pmbm_ct_ktr_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-5,
        "max_hyps": 1000,
    }
    pmbm = tracker.IMMPoissonMultiBernoulliMixture(**PMBM_args, **RFS_base_args)
    pmbm.save_covs = True

    # test save/load filter
    filt_state = pmbm.save_filter_state()
    pmbm = tracker.IMMPoissonMultiBernoulliMixture()
    pmbm.load_filter_state(filt_state)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    state_mat = gdyn.CoordinatedTurnKnown(turn_rate=0 * np.pi / 180).get_state_mat(
        0, dt
    )
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if tt > 3:
            state_mat = gdyn.CoordinatedTurnKnown(
                turn_rate=60 * np.pi / 180
            ).get_state_mat(tt, dt)

        true_agents = _update_true_agents_imm_pmbm(
            true_agents, tt, dt, b_model, rng, state_mat
        )
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        pmbm.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_imm(
            tt,
            true_agents,
            filt.in_filt_list[0].proc_noise,
            filt.in_filt_list[0].meas_noise,
            rng,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        pmbm.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        pmbm.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    pmbm.calculate_ospa(global_true, 2, 1)
    if debug_plots:
        pmbm.plot_ospa_history(time=time, time_units="s")
    pmbm.calculate_ospa2(global_true, 5, 1, 10)
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    pmbm.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.EUCLIDEAN
    )
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    pmbm.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.MAHALANOBIS
    )
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    if debug_plots:
        pmbm.plot_states([0, 1], true_states=global_true, meas_inds=[0, 1])
        pmbm.plot_card_dist()
        pmbm.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"


def test_IMM_LPMBM():  # noqa
    print("Test IMM-LPMBM")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 6 + dt  # 6 + dt

    filt = _setup_imm_ctktr_kf(dt)
    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    b_model = _setup_imm_pmbm_ct_ktr_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-5,
        "max_hyps": 1000,
    }
    pmbm = tracker.IMMLabeledPoissonMultiBernoulliMixture(**PMBM_args, **RFS_base_args)
    pmbm.save_covs = True

    # test save/load filter
    filt_state = pmbm.save_filter_state()
    pmbm = tracker.IMMLabeledPoissonMultiBernoulliMixture()
    pmbm.load_filter_state(filt_state)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    state_mat = gdyn.CoordinatedTurnKnown(turn_rate=0 * np.pi / 180).get_state_mat(
        0, dt
    )
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        # print("\t\t{:.2f}".format(tt))
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if tt > 3:
            state_mat = gdyn.CoordinatedTurnKnown(
                turn_rate=60 * np.pi / 180
            ).get_state_mat(tt, dt)

        true_agents = _update_true_agents_imm_pmbm(
            true_agents, tt, dt, b_model, rng, state_mat
        )
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        pmbm.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_imm(
            tt,
            true_agents,
            filt.in_filt_list[0].proc_noise,
            filt.in_filt_list[0].meas_noise,
            rng,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        pmbm.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        pmbm.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    pmbm.calculate_ospa(global_true, 2, 1)
    if debug_plots:
        pmbm.plot_ospa_history(time=time, time_units="s")
    pmbm.calculate_ospa2(global_true, 5, 1, 10)
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    pmbm.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.EUCLIDEAN
    )
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    pmbm.calculate_ospa2(
        global_true, 5, 1, 10, core_method=SingleObjectDistance.MAHALANOBIS
    )
    if debug_plots:
        pmbm.plot_ospa2_history(time=time, time_units="s")
    if debug_plots:
        pmbm.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        pmbm.plot_card_dist()
        pmbm.plot_card_history(time_units="s", time=time)
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"


@pytest.mark.slow
def test_MS_PMBM():  # noqa
    print("Test MS-GM-PMBM")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1.0 + dt  # 5.5 + dt

    filt = _setup_double_int_gci_kf(dt)

    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    b_model = _setup_pmbm_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1**-3,
        "clutter_rate": 1**-3,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-3,
        "exist_threshold": 10**-5,
        "max_hyps": 400,
    }
    pmbm = tracker.MSPoissonMultiBernoulliMixture(**PMBM_args, **RFS_base_args)
    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            # if np.mod(kk, 50) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_pmbm(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        pmbm.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_ms(
            tt,
            true_agents,
            filt.proc_noise,
            filt.meas_noise_list,
            rng,
            filt.meas_model_list,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        pmbm.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        pmbm.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    pmbm.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        pmbm.plot_states([0, 1], true_states=global_true, meas_inds=[0, 1])
        # pmbm.plot_states([0, 1], meas_inds=[0, 1])
        pmbm.plot_card_dist()
        pmbm.plot_card_history(time_units="s", time=time)
        pmbm.plot_ospa_history()
    print("\tExpecting {} agents".format(len(true_agents)))

    assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"


# @pytest.mark.slow
@pytest.mark.skip(reason="Unresolved Errors in Data Fusion")
def test_MS_LPMBM():  # noqa
    print("Test MS-GM-LPMBM")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 0.1 + dt
    # t0, t1 = 0, 2 + dt  # 5.5 + dt
    # t0, t1 = 0, 5.0 + dt

    # filt = _setup_double_int_gci_kf(dt)
    filt = _setup_double_int_gci_ekf(dt)

    state_mat_args = (dt,)
    # meas_fun_args = ("useless arg",)
    meas_fun_args = ()

    b_model = _setup_pmbm_double_int_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1**-3,
        "clutter_rate": 1**-3,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-5,
        "max_hyps": 1000,
    }
    pmbm = tracker.MSLabeledPoissonMultiBernoulliMixture(**PMBM_args, **RFS_base_args)
    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        print(kk)
        if kk == 100:
            print("error here")
        if np.mod(kk, 10) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        true_agents = _update_true_agents_pmbm(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        # pred_args = {"state_mat_args": state_mat_args}
        pred_args = {"dyn_fun_params": state_mat_args}
        pmbm.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_ms2(
            tt,
            true_agents,
            filt.proc_noise,
            filt.meas_noise_list,
            rng,
            filt.meas_model_list,
        )

        # meas_in = _gen_meas_ms_ekf(
        #     tt,
        #     true_agents,
        #     filt.proc_noise,
        #     filt.meas_noise_list,
        #     rng,
        #     filt.meas_model_list,
        # )

        cor_args = {"meas_fun_args": meas_fun_args}
        pmbm.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        pmbm.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    pmbm.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        pmbm.plot_states_labels([0, 1], true_states=global_true, meas_inds=[0, 1])
        plt.savefig("save_in_dumb_folder/mslpmbm_state_lbl.png")
        pmbm.plot_card_dist()
        plt.savefig("save_in_dumb_folder/mslpmbm_card_dist.png")
        pmbm.plot_card_history(time_units="s", time=time)
        plt.savefig("save_in_dumb_folder/mslpmbm_card_hist.png")
        pmbm.plot_ospa_history()
        plt.savefig("save_in_dumb_folder/mslpmbm_ospa_hist.png")
    print("\tExpecting {} agents".format(len(true_agents)))

    # assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"


@pytest.mark.slow
def test_MS_IMM_PMBM():  # noqa
    print("Test MS-IMM-GM-PMBM")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    # t0, t1 = 0, 5.5 + dt
    t0, t1 = 0, 1.2 + dt

    # TODO GCI IMM FILT SETUP
    filt = _setup_ct_ktr_gci_imm_kf(dt)
    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    b_model = _setup_imm_pmbm_ct_ktr_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1**-3,
        "clutter_rate": 1**-3,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-5,
        "max_hyps": 1000,
    }
    pmbm = tracker.MSIMMPoissonMultiBernoulliMixture(**PMBM_args, **RFS_base_args)
    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    state_mat = gdyn.CoordinatedTurnKnown(turn_rate=0 * np.pi / 180).get_state_mat(
        0, dt
    )
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if (
            tt > 0.5
        ):  # Error caused here, after it changes, tracker loses all confidence
            state_mat = gdyn.CoordinatedTurnKnown(
                turn_rate=60 * np.pi / 180
            ).get_state_mat(tt, dt)

        true_agents = _update_true_agents_imm_pmbm(
            true_agents, tt, dt, b_model, rng, state_mat
        )
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        pmbm.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_ms(
            tt,
            true_agents,
            filt.in_filt_list[0].proc_noise,
            filt.in_filt_list[0].meas_noise,
            rng,
            filt.meas_model_list,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        pmbm.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": False}
        pmbm.cleanup(extract_kwargs=extract_kwargs)
    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    pmbm.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        pmbm.plot_states([0, 1], true_states=global_true, meas_inds=[0, 1])
        pmbm.plot_card_dist()
        pmbm.plot_card_history(time_units="s", time=time)
        pmbm.plot_ospa_history()
    print("\tExpecting {} agents".format(len(true_agents)))
    assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"


@pytest.mark.slow
def test_MS_IMM_LPMBM():  # noqa
    print("Test MS-IMM-GM-LPMBM")

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    # t0, t1 = 0, 5.5 + dt
    t0, t1 = 0, 1.2 + dt
    # t0, t1 = 0, 2 + dt

    filt = _setup_ct_ktr_gci_imm_kf(dt)

    state_mat_args = (dt,)
    meas_fun_args = ("useless arg",)

    b_model = _setup_imm_pmbm_ct_ktr_birth()

    RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 1**-3,
        "clutter_rate": 1**-3,
    }
    PMBM_args = {
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "exist_threshold": 10**-5,
        "max_hyps": 1000,
    }
    pmbm = tracker.MSIMMLabeledPoissonMultiBernoulliMixture(
        **PMBM_args, **RFS_base_args
    )
    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    state_mat = gdyn.CoordinatedTurnKnown(turn_rate=0 * np.pi / 180).get_state_mat(
        0, dt
    )
    print("\tStarting sim")
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print("\t\t{:.2f}".format(tt))
            sys.stdout.flush()
        if tt > 0.5:
            state_mat = gdyn.CoordinatedTurnKnown(
                turn_rate=60 * np.pi / 180
            ).get_state_mat(tt, dt)

        true_agents = _update_true_agents_imm_pmbm(
            true_agents, tt, dt, b_model, rng, state_mat
        )
        global_true.append(deepcopy(true_agents))

        pred_args = {"state_mat_args": state_mat_args}
        pmbm.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_ms(
            tt,
            true_agents,
            filt.in_filt_list[0].proc_noise,
            filt.in_filt_list[0].meas_noise,
            rng,
            filt.meas_model_list,
        )

        cor_args = {"meas_fun_args": meas_fun_args}
        pmbm.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {"update": True, "calc_states": True}
        pmbm.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {"update": False, "calc_states": True}
    pmbm.extract_states(**extract_kwargs)

    pmbm.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        pmbm.plot_states_labels([0, 1], true_states=global_true)  # , meas_inds=[0, 1])
        pmbm.plot_card_dist()
        pmbm.plot_card_history(time_units="s", time=time)
        pmbm.plot_ospa_history()
    print("\tExpecting {} agents".format(len(true_agents)))
    assert len(true_agents) == pmbm.cardinality, "Wrong cardinality"

def main():
    from timeit import default_timer as timer
    import matplotlib

    matplotlib.use("WebAgg")

    plt.close("all")

    debug_plots = True

    start = timer()

    test_PHD()
    # test_PHD_spawning()
    # test_CPHD()
    # test_CPHD_spawning()
    # test_IMM_PHD()
    # test_IMM_CPHD()

    #test_GLMB()
    # test_STM_GLMB()
    # test_SMC_GLMB()
    # test_USMC_GLMB()
    # test_MCMC_USMC_GLMB()
    # test_QKF_GLMB()
    # test_SQKF_GLMB()
    # test_UKF_GLMB()
    # test_QKF_GSM_GLMB()
    # test_SQKF_GSM_GLMB()
    # test_UKF_GSM_GLMB()
    # test_EKF_GSM_GLMB()

    # test_JGLMB()
    # test_JGLMB_high_birth()
    # test_STM_JGLMB()
    # test_SMC_JGLMB()
    # test_USMC_JGLMB()
    # test_MCMC_USMC_JGLMB()
    # test_QKF_JGLMB()
    # test_SQKF_JGLMB()
    # test_UKF_JGLMB()
    # test_QKF_GSM_JGLMB()
    # test_SQKF_GSM_JGLMB()
    # test_UKF_GSM_JGLMB()
    # test_GLMB_ct_ktr()
    # test_IMM_GLMB()
    # test_IMM_JGLMB()
    # test_MS_JGLMB()
    # test_MS_IMM_JGLMB()

    # test_PMBM()
    # test_LPMBM()
    # test_STM_PMBM()
    # test_STM_LPMBM()
    # test_SMC_PMBM()
    # test_SMC_LPMBM()
    # test_IMM_PMBM()
    # test_IMM_LPMBM()
    # test_MS_PMBM()
    # test_MS_LPMBM()
    # test_MS_IMM_PMBM()
    # test_MS_IMM_LPMBM()

    end = timer()
    print("{:.2f} s".format(end - start))
    print("Close all plots to exit")
    plt.show()

# %% main
if __name__ == "__main__":
    main()



