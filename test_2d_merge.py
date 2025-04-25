import sys
import matplotlib.backends
import os
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy

import scipy.stats as stats

import gncpy.filters as gfilts
import gncpy.dynamics.basic as gdyn
import gncpy.distributions as gdistrib

from carbs.extended_targets.GGIW_Serums_Models import GGIW, GGIWMixture

from carbs.extended_targets.GGIW_EKF import GGIW_ExtendedKalmanFilter
import carbs.extended_targets.GGIW_RFS as GGIW_RFS

import carbs_clustering

def rot_ellipse(mat, rot_deg):
    ctheta = np.cos(np.deg2rad(rot_deg))
    stheta = np.sin(np.deg2rad(rot_deg))

    rot_m = np.array([[ctheta, -stheta],[stheta, ctheta]])
    return rot_m @ mat @ rot_m.transpose()

filt = GGIW_ExtendedKalmanFilter(forgetting_factor=120, tau=120, cont_cov=False)
filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator()) 
filt.set_measurement_model(meas_mat=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))
filt.proc_noise = np.diag([100, 100, 100, 100])
filt.meas_noise = 1 * np.eye(2)
filt.dt = 1

state_mat_args = (1,)

clustering_params = carbs_clustering.DBSCANParameters()
clustering = carbs_clustering.MeasurementClustering(clustering_params)
merge_thres = 5
# Set up tracker
tracker_birth_model = GGIWMixture(alphas=[15.0, 10.0, 4.0], 
            betas=[1.0, 5, 3],
            means=[np.array([1, 2, 0, 0]).reshape((4, 1)), np.array([1, 2, 3, 4]).reshape((4, 1)),np.array([3, 4, 5, 6]).reshape((4, 1))],
            covariances=[np.diag([2,2,1,3]), np.diag([2,2,2,2]), np.diag([2,3,5,6])],
            IWdofs=[8.0, 10, 5],
            IWshapes=[np.array([[100, 0],[0, 100]]), np.array([[50, 0],[0, 50]]), np.array([[20, 0],[0, 20]])],
            weights=[0.8, 0.5, 0.4])
test_mix = GGIWMixture()

RFS_base_args = {
        "prob_detection": 0.9,
        "prob_survive": 0.999,
        "in_filter": filt,
        "birth_terms": tracker_birth_model,
        "clutter_den": 1,
        "clutter_rate": 1,
    }


phd = GGIW_RFS.GGIW_PHD(clustering_obj=clustering,extract_threshold=0.0,merge_threshold=merge_thres,prune_threshold=0.0,**RFS_base_args)
phd.gating_on = False

w_list = [0.5, 0.5]
mean_list = [np.array([0.2, 0.2]).reshape(2,1), np.array([0.0, 0.0]).reshape(2,1)]
cov_list = [rot_ellipse(np.diag([0.1, 0.2]),0), rot_ellipse(np.diag([0.2, 0.25]),0)]
alpha_list = [2500, 2500]
beta_list = [50, 50]
dof_list = [int(10), int(10)]
scale_list = [rot_ellipse(np.diag([9, 1]),10)*(dof_list[0]-2-1), rot_ellipse(np.diag([9, 1]),0)*(dof_list[1]-2-1)]

mixture = GGIWMixture()
mixture.add_components(alphas=alpha_list, weights=w_list, means=mean_list, covariances=cov_list, betas=beta_list, 
                            IWdofs=dof_list, IWshapes=scale_list)

phd.test_set_mixture(mixture)

before_mix = phd.extract_mixture()
print(before_mix)

phd._merge()

after_mix = phd.extract_mixture()
print(after_mix)

fig, ax = plt.subplots(figsize=(12, 6))

before_mix._distributions[0].plot_distribution(color="r", label="p1")
before_mix._distributions[1].plot_distribution(color="g",label="p2")

after_mix._distributions[0].plot_distribution(color="b",label="merged")

title_str = str(w_list)

ax.set_aspect('equal')
ax.set_title(title_str)
ax.legend()


fig.savefig("test.png")
plt.show()