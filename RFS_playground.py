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

import carbs_clustering

clustering_params = carbs_clustering.DBSCANParameters(min_samples=10, eps=10)
clustering = carbs_clustering.MeasurementClustering(clustering_params)

dt = 0.5
t0, t1 = 0, 30 + dt  

# birth_model = GGIWMixture(alphas=[5.0, 5.0], 
#             betas=[1.0, 1.0],
#             means=[np.array([50, 50, -1, 0]).reshape((4, 1)),np.array([15, 15, 0, 2]).reshape((4, 1))],
#             covariances=[np.diag([100,100,10,10]),np.diag([100,100,10,10])],
#             IWdofs=[80.0, 80.0],
#             IWshapes=[np.array([[70, 25],[25, 70]]),np.array([[100, 25],[25, 100]])])

birth_model = GGIWMixture(alphas=[3.0], 
            betas=[0.5],
            means=[np.array([15, 15, 0, 0]).reshape((4, 1))],
            covariances=[np.diag([50**2,50**2,100,100])],
            IWdofs=[10.0],
            IWshapes=[np.array([[500, 0],[0, 500]])],
            weights=[1])

# birth_model.weights = [0.5]

# amount of timesteps to based gamma estimation on
w_e = 2
eta_k = w_e / (w_e - 1) # forgetting factor

filt = GGIW_ExtendedKalmanFilter(forgetting_factor=eta_k,tau=1, cont_cov=True)
filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator()) 
filt.set_measurement_model(meas_mat=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))

filt.proc_noise = np.diag([0.01, 0.01, 0.01, 0.01])
filt.meas_noise = 16 * np.eye(2)                          # Note: Higher meas noise keeps singularities from happening during the inverse wishart correction step

filt.dt = dt

state_mat_args = (dt,)

RFS_base_args = {
        "prob_detection": 0.9,
        "prob_survive": 0.99,
        "in_filter": filt,
        "birth_terms": birth_model,
        "clutter_den": 0.1,
        "clutter_rate": 1,
    }
phd = GGIW_RFS.GGIW_PHD(clustering_obj=clustering,extract_threshold=0.4,\
                        merge_threshold=0.1, prune_threshold=0.001,**RFS_base_args)
phd.gating_on = False
phd.merge_as_average = True

b_model = [(birth_model, 0.05)] # include probability of birth in tuple




filt.proc_noise = 0.0001 * np.diag([0.01, 0.01, 0.01, 0.01])
filt.meas_noise = 20 * np.eye(2)

GLMB_RFS_base_args = {
        "prob_detection": 0.9,
        "prob_survive": 0.99,
        "in_filter": filt,
        "birth_terms": b_model,
        "clutter_den": 0.1,
        "clutter_rate": 1,
    }

GLMB_args = {
        "req_births": len(b_model) + 1,
        "req_surv": 1000,
        "req_upd": 800,
        "prune_threshold": 10**-5,
        "max_hyps": 1000,
    }

glmb = GGIW_RFS.GGIW_GLMB(clustering_obj=clustering, **GLMB_args, **GLMB_RFS_base_args)

# phd.predict(timestep=1, filt_args=(dt,))

# # print(phd._Mixture)
# # phd._Mixture.plot_distributions(plt_inds = [0,2])
# # plt.show()


# phd.predict(timestep=1+dt, filt_args=(dt,)) 
# phd._Mixture.plot_distributions(plt_inds = [0,1])
# plt.show()

# print(phd._Mixture)

# # print(phd._Mixture.alphas)
# # print(phd._Mixture.betas)
# # print(phd._Mixture.means)
# # print(phd._Mixture.covariances)
# # print(phd._Mixture.IWdofs)
# # print(phd._Mixture.IWshapes)
# # print(phd._Mixture.weights)

# print(gdyn.DoubleIntegrator().state_names)

truth_kinematics = gdyn.DoubleIntegrator()

truth_model = GGIWMixture(alphas=[100.0, 100.0, 100.0], 
            betas=[1.0, 1.0, 1.0],
            means=[np.array([40, 5, -1, 0]).reshape((4, 1)),np.array([10, 5, 0, 1.5]).reshape((4, 1)),np.array([50, 50, -3, 0]).reshape((4, 1))],
            covariances=[np.diag([0,0,0,0]),np.diag([0,0,0,0]),np.diag([0,0,0,0])],
            IWdofs=[30.0, 30.0, 30.0],
            IWshapes=[np.array([[100, 50],[50, 100]]),np.array([[70, 25],[25, 70]]),np.array([[70, 25],[25, 70]])])

time = np.arange(t0, t1, dt)

# print(filt._dyn_obj)

fig, ax = plt.subplots()
fig.set_figheight(9)
fig.set_figwidth(13)

for kk, t in enumerate(time[:-1]):
    phd.predict(t, filt_args=(dt,))

    glmb.predict(t, filt_args=(dt,))

    new_mean = truth_model.means
    for ii in range(len(truth_model._distributions)):
        new_mean[ii]  = truth_kinematics.propagate_state(t,truth_model._distributions[ii].mean,state_args=(dt,)).flatten()
    truth_model.means = new_mean
    
    measurements = []
    for ii in range(len(truth_model._distributions)):
        temp = (truth_model._distributions[ii].sample_measurements(xy_inds=[0,1],random_extent=False))
        for ii in range(temp.shape[1]):
            measurements.append(temp[:,ii].reshape((2,1))) 

    
    #print(measurements)

    phd.correct(timestep=t,meas_in=measurements)

    glmb.correct(t, meas_in=measurements)

    # print(phd._Mixture)

    phd.cleanup(enable_merge=True)
    
    extract_kwargs = {"update": True, "calc_states": True} 
    glmb.cleanup(extract_kwargs=extract_kwargs)

    print_glmbs = False 
    plot_glmbs = True

    if print_glmbs:
        print(f"Time Index: {kk} \n")
        for ii in glmb.labels:
            print(ii)
        print("\n\n\n")


    # print("Cleaned up Mixture: \n\n")
    # print(phd._Mixture)

    mix = phd.extract_mixture()

    ax.clear()
    ax.plot([0],[0])
    ax.set_xlim((-40,60))
    ax.set_ylim((-20,60))
    for each_meas in measurements:
        ax.scatter(each_meas[0, :], each_meas[1, :], marker='.', label='sampled points',c='k',s=1)  
    ax.grid()

    # truth_model.plot_distributions(plt_inds=[0,1],num_std=1,ax=ax,color='k')
    
    mix.plot_confidence_extents(h=0.95, plt_inds=[0, 1], ax=ax, edgecolor='r', linewidth=1.5) #(plt_inds=[0,1],ax=ax,edgecolor='r',linewidth=3)

    if plot_glmbs:
        # for ggiw in glmb.GGIWs[-1]:
        #     ggiw.plot_confidence_extents(h=0.95, plt_inds=[0, 1], ax=ax, edgecolor='g', linewidth=1.5) 

        glmb.plot_states_labels(ax=ax)

    # phd._Mixture.plot_distributions(plt_inds=[0,1],ax=ax,edgecolor='b')

    # print("Extracted Mixture: \n\n")
    # print(mix)

    # print("Sum of weights: \n")
    # print(np.sum(phd._Mixture.weights))

    ax.set_aspect(1)

    plt.pause(0.2) 

    plt.savefig(f"image_set/{kk}.png")

plt.show()


