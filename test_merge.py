import sys
import matplotlib.backends
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

global_seed = 69

def calc_gaussian_pdf(mu, cov, w, low, high, num_pts):
    x_query = np.linspace(num=num_pts, start=low, stop=high).reshape((1,num_pts))
    res = w * stats.norm.pdf(x_query, loc=mu, scale = cov)
    return res, x_query

filt = GGIW_ExtendedKalmanFilter(forgetting_factor=120, tau=120, cont_cov=False)
filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator()) 
filt.set_measurement_model(meas_mat=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))
filt.proc_noise = np.diag([100, 100, 100, 100])
filt.meas_noise = 1 * np.eye(2)
filt.dt = 1

state_mat_args = (1,)

clustering_params = carbs_clustering.DBSCANParameters()
clustering = carbs_clustering.MeasurementClustering(clustering_params)

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

phd = GGIW_RFS.GGIW_PHD(clustering_obj=clustering,extract_threshold=0.0,merge_threshold=3,prune_threshold=0.0,**RFS_base_args)
phd.gating_on = False

# Generate mixture of 1D GGIW for testing
num_component = 20
rng = rnd.default_rng(global_seed)
d = 1

w_list = [rnd.uniform(low=0.05, high=0.95) for ii in range(num_component)]
mean_list = [np.array(rnd.uniform(low=0.0, high=10.0)).reshape((d,1)) for ii in range(num_component)]
cov_list = [np.array(rnd.uniform(low=0.25**2, high=0.75**2)).reshape((d,d)) for ii in range(num_component)]
gamma_list = [rnd.uniform(low=1, high=20) for ii in range(num_component)]
alpha_list = [rnd.uniform(low=1, high=40) for ii in range(num_component)]
beta_list = [a/g for a,g in zip(alpha_list, gamma_list)]
dof_list = [rnd.uniform(low=1, high=40) for ii in range(num_component)]
X_list = [rnd.uniform(low=15, high=50) for ii in range(num_component)]
shape_list = [np.array(X * (n - 2 * d -2)).reshape((d,d)) for X,n in zip(X_list, dof_list)]
test_mixture = GGIWMixture()
test_mixture.add_components(alphas=alpha_list, weights=w_list, means=mean_list, covariances=cov_list, betas=beta_list, 
                            IWdofs=dof_list, IWshapes=shape_list)



phd.test_set_mixture(test_mixture)

before_mix = phd.extract_mixture()

phd._merge()

after_mix = phd.extract_mixture()

gaus_start = 0
gaus_stop = 10
num_pt = 1000

before_gaus_pdf, x_query = calc_gaussian_pdf(mu=before_mix.means[0], cov=before_mix.covariances[0], w=before_mix.weights[0], low=gaus_start, high=gaus_stop, num_pts=num_pt)
for ii in range(1, len(before_mix.weights)):
    before_gaus_pdf += calc_gaussian_pdf(mu=before_mix.means[ii], cov=before_mix.covariances[ii], w=before_mix.weights[ii], low=gaus_start, high=gaus_stop, num_pts=num_pt)[0]

after_gaus_pdf = calc_gaussian_pdf(mu=after_mix.means[0], cov=after_mix.covariances[0], w=after_mix.weights[0], low=gaus_start, high=gaus_stop, num_pts=num_pt)[0]
for ii in range(1, len(after_mix.weights)):
    after_gaus_pdf += calc_gaussian_pdf(mu=after_mix.means[ii], cov=after_mix.covariances[ii], w=after_mix.weights[ii], low=gaus_start, high=gaus_stop, num_pts=num_pt)[0]


fig, ax = plt.subplots()
ax.plot(x_query.transpose(), before_gaus_pdf.transpose(), linewidth=2)
ax.plot(x_query.transpose(), after_gaus_pdf.transpose(), linewidth=2)
ax.set_ylim([0, 10])
ax.grid()
plt.show()






