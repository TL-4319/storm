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

global_seed = 420

def calc_gaussian_pdf(mu, cov, w, x_query):
    res = w * stats.norm.pdf(x_query, loc=mu, scale = np.sqrt(cov))
    return res

def calc_IW_pdf(dof, scale, w, x_query):
    res = w * stats.invwishart.pdf(x_query.flatten(), df=dof, scale = scale[0][0])
    return res

def calc_gamma_pdf(alpha, beta, w, x_query):
    res = w * stats.gamma.pdf(x_query, a = alpha, scale = 1/beta)
    return res



def write_list(lst, f):
    for ii in range(len(lst)):
        if type(lst[ii]) is np.ndarray:
            f.write(str(lst[ii][0][0]))
        else:
            f.write(str(lst[ii]))
        if ii < len(lst)-1:
            f.write(",")
    f.write("\n")

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

# Generate mixture of 1D GGIW for testing
num_component = 50
rng = rnd.default_rng(global_seed)
d = 1

w_list = [rng.uniform(low=0.05, high=0.95) for ii in range(num_component)]
mean_list = [np.array(rng.uniform(low=0.0, high=10.0)).reshape((d,1)) for ii in range(num_component)]
cov_list = [np.array(rng.uniform(low=0.25**2, high=0.75**2)).reshape((d,d)) for ii in range(num_component)]
gamma_list = [rng.uniform(low=10, high=50) for ii in range(num_component)]
alpha_list = [rng.uniform(low=50, high=2500) for ii in range(num_component)]
#beta_list = [rng.uniform(low=5, high=50) for ii in range(num_component)]
beta_list = [a/g for a,g in zip(alpha_list, gamma_list)]
dof_list = [float(round(rng.uniform(low=5, high=40))) for ii in range(num_component)]
X_list = [rng.uniform(low=15, high=50) for ii in range(num_component)]
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

x_query = np.linspace(start=gaus_start, stop=gaus_stop, num=num_pt).reshape((1,num_pt))

before_gaus_pdf = np.zeros(x_query.shape)
for ii in range(len(before_mix.weights)):
    before_gaus_pdf += calc_gaussian_pdf(mu=before_mix.means[ii], cov=before_mix.covariances[ii], w=before_mix.weights[ii], x_query=x_query)

after_gaus_pdf = np.zeros(x_query.shape)
for ii in range(len(after_mix.weights)):
    after_gaus_pdf += calc_gaussian_pdf(mu=after_mix.means[ii], cov=after_mix.covariances[ii], w=after_mix.weights[ii], x_query=x_query)


print(len(before_mix.weights))
print(len(after_mix.weights))


# Write before and after list to file
folder_name = "test_1"
before_file = folder_name + "/before.txt"
after_file = folder_name + "/after.txt"

if not os.path.isdir(folder_name):
    # Create dir for pre process data if it didn't exist
    os.mkdir(folder_name)

# Write before
with open(before_file, 'w') as f:
    write_list(before_mix.weights, f)
    write_list(before_mix.means, f)
    write_list(before_mix.covariances, f)
    write_list(before_mix.alphas, f)
    write_list(before_mix.betas, f)
    write_list(before_mix.IWdofs, f)
    write_list(before_mix.IWshapes, f)

with open(after_file, 'w') as f:
    write_list(after_mix.weights, f)
    write_list(after_mix.means, f)
    write_list(after_mix.covariances, f)
    write_list(after_mix.alphas, f)
    write_list(after_mix.betas, f)
    write_list(after_mix.IWdofs, f)
    write_list(after_mix.IWshapes, f)


gaus_start = 0
gaus_stop = 10
num_pt = 1000

x_query = np.linspace(start=gaus_start, stop=gaus_stop, num=num_pt).reshape((1,num_pt))

before_gaus_pdf = np.zeros(x_query.shape)
for ii in range(len(before_mix.weights)):
    before_gaus_pdf += calc_gaussian_pdf(mu=before_mix.means[ii], cov=before_mix.covariances[ii], w=before_mix.weights[ii], x_query=x_query)

after_gaus_pdf = np.zeros(x_query.shape)
for ii in range(len(after_mix.weights)):
    after_gaus_pdf += calc_gaussian_pdf(mu=after_mix.means[ii], cov=after_mix.covariances[ii], w=after_mix.weights[ii], x_query=x_query)

IW_start = 10
IW_stop = 50

X_query = np.linspace(start=IW_start, stop=IW_stop, num=num_pt).reshape((1,num_pt))
before_IW_pdf = np.zeros(X_query.shape)
for ii in range(len(before_mix.weights)):
    before_IW_pdf += calc_IW_pdf(dof=before_mix.IWdofs[ii], scale=before_mix.IWshapes[ii], w=before_mix.weights[ii], x_query=X_query)

after_IW_pdf = np.zeros(X_query.shape)
for ii in range(len(after_mix.weights)):
    after_IW_pdf += calc_IW_pdf(dof=after_mix.IWdofs[ii], scale=after_mix.IWshapes[ii], w=after_mix.weights[ii], x_query=X_query)

lamd_start = 10
lamd_stop = 50

lamd_querry = np.linspace(start=lamd_start, stop=lamd_stop, num=num_pt).reshape((1,num_pt))

before_gamma_pdf = np.zeros(lamd_querry.shape)
for ii in range(len(before_mix.weights)):
    before_gamma_pdf += calc_gamma_pdf(alpha=before_mix.alphas[ii], beta=before_mix.betas[ii],
                                       w=before_mix.weights[ii],x_query=lamd_querry)

after_gamma_pdf = np.zeros(lamd_querry.shape)
for ii in range(len(after_mix.weights)):
    after_gamma_pdf += calc_gamma_pdf(alpha=after_mix.alphas[ii], beta=after_mix.betas[ii],
                                       w=after_mix.weights[ii],x_query=lamd_querry)

title_str = "Thres = " + str(int(merge_thres)) + ". " + str(len(before_mix.weights)) + \
    " -> " + str(len(after_mix.weights))

fig, ax = plt.subplots(3,1, figsize=(8, 12))
ax[0].set_title(title_str)
ax[0].plot(x_query.transpose(), before_gaus_pdf.transpose(), linewidth=2,label="Before")
ax[0].plot(x_query.transpose(), after_gaus_pdf.transpose(), linewidth=2,label="After")
ax[0].grid()
ax[0].set_xlabel("Gaussian")
ax[0].legend()

ax[1].plot(X_query.transpose(), before_IW_pdf.transpose(), linewidth=2)
ax[1].plot(X_query.transpose(), after_IW_pdf.transpose(), linewidth=2)
ax[1].grid()
ax[1].set_xlabel("Inverse Wishart")

ax[2].plot(lamd_querry.transpose(), before_gamma_pdf.transpose(), linewidth=2)
ax[2].plot(lamd_querry.transpose(), after_gamma_pdf.transpose(), linewidth=2)
ax[2].grid()
ax[2].set_xlabel("Gamma")

fig.savefig("test.png")
plt.show()




