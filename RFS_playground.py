import sys
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy

import scipy.stats as stats
import gncpy.dynamics.basic as gdyn

from carbs.extended_targets.GGIW_Serums_Models import GGIW, GGIWMixture

from carbs.extended_targets.GGIW_EKF import GGIW_ExtendedKalmanFilter
import carbs.extended_targets.GGIW_RFS as GGIW_RFS



dt = 1
t0, t1 = 0, 90 + dt  

birth_model = GGIWMixture(alphas=[10.0, 10.0], 
            betas=[1/2.0, 1/5.0],
            means=[np.array([65, 0, 65, 0]).reshape((4, 1)),np.array([20, 0, 20, 0]).reshape((4, 1))],
            covariances=[np.diag([65**2,10,65**2,10]),np.diag([65,10,65,10])],
            IWdofs=[8.0, 20.0],
            IWshapes=[np.array([[25, 5],[5, 25]]),np.array([[50, 15],[15, 50]])])

# birth_model.weights[1] = 0.7                      # Just showing that the weights could be changed, but initially they'll be equal to 1/(number of mixtures)

birth_model.add_components(30.0, 1/9.0, np.array([100, 0, 100, 0]).reshape((4, 1)), np.diag([65,10,65,10]), 10.0, np.array([[50, 0],[0, 50]]), 0.7)

filt = GGIW_ExtendedKalmanFilter(forgetting_factor=3,tau=1)
filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator()) 
filt.set_measurement_model(meas_mat=np.array([[1, 0, 0, 0], [0, 0, 1, 0]]))

filt.proc_noise = np.diag([10, 1, 10, 1])
filt.meas_noise = 0.2 * np.eye(2)

filt.dt = dt

# print(birth_model) 

# birth_model.plot_distributions(plt_inds = [0,2])

# plt.show()

state_mat_args = (dt,)

RFS_base_args = {
        "prob_detection": 0.99,
        "prob_survive": 0.98,
        "in_filter": filt,
        "birth_terms": birth_model,
        "clutter_den": 1e-7,
        "clutter_rate": 1e-7,
    }
phd = GGIW_RFS.GGIW_PHD(**RFS_base_args)
phd.gating_on = False

filt_args = {"state_mat_args": state_mat_args}
phd.predict(timestep=1, filt_args=filt_args)

# print(phd._Mixture)
# phd._Mixture.plot_distributions(plt_inds = [0,2])
# plt.show()


phd.predict(timestep=1+dt, filt_args=(dt,)) 
phd._Mixture.plot_distributions(plt_inds = [0,2])
plt.show()

print(phd._Mixture)