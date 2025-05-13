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

def truth_n_measurements(time, truth_model, truth_kinematics, FOV_lim=[-40, 60], min_max_samples = [10, 50]):
    measurements = []
    truth_model_list = []
    for ii, a in enumerate(truth_model.alphas): 
        truth_model_list.append([truth_model[ii]])
    for kk, t in enumerate(time[:-1]):
        temp_measurements = []
        for ii in range(len(truth_model._distributions)):
            temp = (truth_model._distributions[ii].sample_measurements(xy_inds=[0,1],random_extent=False)) #.round()

            # temp = np.unique(temp, axis=1)
            # Cull any measurment outside of FOV
            out_FOV = np.where(np.logical_or(temp[0,:] < FOV_lim[0], temp[0,:] > FOV_lim[1]))
            out_FOV = out_FOV[0]
            temp = np.delete(temp, out_FOV, 1)

            for ii in range(temp.shape[1]):
                temp_measurements.append(temp[:,ii].reshape((2,1))) 

        min_samples, max_samples = min_max_samples[0], min_max_samples[1]
        n_samples = np.random.randint(min_samples, max_samples + 1)
        for _ in range(n_samples):
            temp_measurements.append(np.random.uniform([FOV_lim[0], FOV_lim[0]], [FOV_lim[1], FOV_lim[1]], 2).reshape(2, 1)) 

        measurements.append(temp_measurements)

        # Propagate truth forward and append to truth list
        new_mean = truth_model.means
        new_dofs = truth_model.IWdofs
        new_alphas = truth_model.alphas
        new_shapes = truth_model.IWshapes
        for ii in range(len(truth_model._distributions)):
            new_mean[ii]  = truth_kinematics.propagate_state(t,truth_model._distributions[ii].mean,state_args=(dt,)) 
            if new_dofs[ii] > 10:
                new_dofs[ii] = 1 * new_dofs[ii] 
            new_alphas[ii] = 1 * new_alphas[ii] 
            new_shapes[ii] = new_shapes[ii] + 40 * np.eye(2)
        truth_model.means = new_mean
        truth_model.IWdofs = new_dofs
        truth_model.alphas = new_alphas 

        for ii, a in enumerate(truth_model.alphas):
            truth_model_list[ii].append(deepcopy(truth_model[ii])) 

    return measurements, truth_model_list

def initialize_filters(dt):
    clustering_params = carbs_clustering.DBSCANParameters(min_samples=10, eps=5)
    clustering = carbs_clustering.MeasurementClustering(clustering_params)

    birth_model = GGIWMixture(alphas=[10.0], 
                betas=[1],
                means=[np.array([-60, 30, 0, 0]).reshape((4, 1))],
                covariances=[np.diag([10**2,10**2,5,5])],
                IWdofs=[10.0],
                IWshapes=[np.array([[100, 0],[0, 100]])],
                weights=[1])

    # amount of timesteps to based gamma estimation on
    w_e = 2
    eta_k = w_e / (w_e - 1) # forgetting factor

    filt = GGIW_ExtendedKalmanFilter(forgetting_factor=eta_k,tau=1, cont_cov=False)
    filt.set_state_model(dyn_obj=gdyn.DoubleIntegrator()) 
    filt.set_measurement_model(meas_mat=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))

    filt.proc_noise = np.diag([0.1, 0.1, 0.01, 0.01])
    filt.meas_noise = 2 * np.eye(2) 

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
    phd = GGIW_RFS.GGIW_PHD(clustering_obj=clustering,extract_threshold=0.5,\
                            merge_threshold=4, prune_threshold=0.001,**RFS_base_args)
    phd.gating_on = False 

    b_model = [(birth_model, 0.01)]

    GLMB_RFS_base_args = {
            "prob_detection": 0.6,
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

    jglmb = GGIW_RFS.GGIW_JGLMB(clustering_obj=clustering, **GLMB_args, **GLMB_RFS_base_args) 

    return phd, glmb, jglmb


if __name__=="__main__":

    truth_kinematics = gdyn.DoubleIntegrator() 

    truth_model = GGIWMixture(alphas=[200.0, 200.0], 
                betas=[1.0, 1.0],
                means=[np.array([-60, 30, 8, 1]).reshape((4, 1)), np.array([-40, 20, 6, -1]).reshape((4, 1))],
                covariances=[np.diag([0,0,0,0]), np.diag([0,0,0,0])],
                IWdofs=[100.0, 100],
                IWshapes=[np.array([[70, 0],[0, 200]]), np.array([[200, 0],[0, 270]])],
                weights=[1,1])

    dt = 0.5
    t0, t1 = 0, 14.5 + dt 
    time = np.arange(t0, t1, dt) 

    phd, glmb, jglmb = initialize_filters(dt=dt)

    measurements, truth_model_list = truth_n_measurements(time, truth_model, truth_kinematics, FOV_lim=[-40, 60], min_max_samples = [10, 50])

    fig, ax = plt.subplots(1,1)
    fig.set_figheight(9)
    fig.set_figwidth(15)

    for kk, t in enumerate(time[:-1]):    
        phd.predict(t, filt_args=(dt,)) 
        phd.correct(timestep=t,meas_in=measurements[kk]) 
        phd.cleanup(enable_merge=True) 
        mix = phd.extract_mixture()
        ax.clear()
        ax.plot([0],[0])
        ax.set_xlim((-40,60))
        ax.set_ylim((-20,80))
        for each_meas in measurements[kk]:
            ax.scatter(each_meas[0, :], each_meas[1, :], marker='.', label='sampled points',c='k',s=1.5)  
        ax.grid()
        mix.plot_confidence_extents(h=0.95, plt_inds=[0, 1], ax=ax, edgecolor='r', linewidth=1.5) 
        plt.pause(0.2)
    
    print("done")
    plt.show()



    # glmb.predict(t, filt_args=(dt,))

    # glmb.correct(t, meas_in=measurements)

    # extract_kwargs = {"update": True, "calc_states": True} 
    # glmb.cleanup(extract_kwargs=extract_kwargs) 

    # glmb.plot_states_labels(ax=ax2, plt_inds=[0, 1], extent_plot_step=4, marker = "none", true_GGIWs=truth_model_list)


