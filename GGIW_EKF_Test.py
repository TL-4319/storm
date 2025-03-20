import numpy as np
import matplotlib.pyplot as plt
import gncpy.dynamics.basic as gdyn
import gncpy.filters as gfilts
import numpy.random as rnd
from carbs.extended_targets.GGIW_EKF import GGIW_ExtendedKalmanFilter
from carbs.extended_targets.GGIW_Serums_Models import GGIW

global_seed = 69
d2r = np.pi / 180
r2d = 1 / d2r

m_noise = 0.02
p_noise = 0.2

dt = 0.01
t0, t1 = 0, 2 + dt

rng = rnd.default_rng(global_seed)

filt = GGIW_ExtendedKalmanFilter()
filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator())
m_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
filt.set_measurement_model(meas_mat=m_mat)

filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(
    dt, np.array([[p_noise**2]])
)
filt.meas_noise = m_noise**2 * np.eye(2)

filt.dt = dt

vx0 = 5
vy0 = 1

ggiw_est = GGIW(alpha=10.0, 
            beta=3.0,
            mean=np.array([0,0,vx0,vy0]),
            covariance=0.25 * np.eye(4),
            IWdof=3.0,
            IWshape=np.array([[1, 0],[0, 1]]))

print(ggiw_est)

ggiw = GGIW(alpha=10.0, 
            beta=1/20.0,
            mean=np.array([0.5,2.0,1.5 * vx0, 0.75 * vy0]),
            covariance=0.2 * np.eye(4),
            IWdof=6.0,
            IWshape=np.array([[1.0, 0.2],
                                [0.2, 0.5]]))


truth_kinematics = gdyn.DoubleIntegrator()

# measurements = ggiw.sample_measurements()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(measurements[:, 0], measurements[:, 1], marker='.', label='sampled points')

# ggiw.plot_distribution(ax=ax, num_std=2.0, edgecolor='r')
# plt.legend()
# plt.show()

time = np.arange(t0, t1, dt)

# print(filt._dyn_obj)

for kk, t in enumerate(time[:-1]):
    ggiw_est = filt.predict(t, ggiw_est, dyn_fun_params=(dt,))

    ggiw.location = truth_kinematics.propagate_state(t,ggiw.mean,state_args=(dt,)).flatten()

    measurements = ggiw.sample_measurements()

    (ggiw_est, q) = filt.correct(t, measurements, ggiw_est)

    print(ggiw)
    print(ggiw_est)

    # fig = plt.figure()
    # ax = fig.add_subplot(1)
    # # ax.scatter(measurements[:, 0], measurements[:, 1], marker='.', label='sampled points')

    # ggiw.plot_distribution(ax=ax, num_std=2.0, edgecolor='r')
    # plt.legend()
    # plt.show()