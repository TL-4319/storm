import sys
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy

import time

import scipy.stats as stats
import gncpy.dynamics.basic as gdyn

from carbs.extended_targets.GGIW_Serums_Models import GGIW, GGIWMixture

from carbs.extended_targets.GGIW_EKF import GGIW_ExtendedKalmanFilter
import carbs.extended_targets.GGIW_RFS as GGIW_RFS

import carbs_clustering

def truth_n_measurements(time, truth_model, truth_kinematics, FOV_lim=[0, 129], min_max_samples = [10, 50]):
    measurements = []
    truth_model_list = []
    for ii, a in enumerate(truth_model.alphas): 
        truth_model_list.append([truth_model[ii]])
    for kk, t in enumerate(time):
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

        if t<time[-2]:
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
                new_shapes[ii] = new_shapes[ii] + 20 * np.eye(2)
            truth_model.means = new_mean
            truth_model.IWdofs = new_dofs
            truth_model.alphas = new_alphas 

            for ii, a in enumerate(truth_model.alphas):
                truth_model_list[ii].append(deepcopy(truth_model[ii])) 

    return measurements, truth_model_list

def initialize_filters(dt):
    clustering_params = carbs_clustering.DBSCANParameters(min_samples=10, eps=10)
    clustering = carbs_clustering.MeasurementClustering(clustering_params)

    birth_model = GGIWMixture(alphas=[100.0], 
                betas=[1],
                means=[np.array([20, 50, 0, 0]).reshape((4, 1))],
                covariances=[np.diag([5**2,5**2,5,5])],
                IWdofs=[10.0],
                IWshapes=[np.array([[200, 0],[0, 200]])],
                weights=[1])


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
    
    cphd = GGIW_RFS.GGIW_CPHD(clustering_obj=clustering,extract_threshold=0.5,\
                            merge_threshold=4, prune_threshold=0.001,**RFS_base_args)
    cphd.gating_on = False 

    b_model = [(birth_model, 0.001)]

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

    jglmb = GGIW_RFS.GGIW_JGLMB(clustering_obj=clustering, **GLMB_args, **GLMB_RFS_base_args) 

    return phd, cphd, glmb, jglmb

def test_phds(
    filter,
    time_arr,
    dt,
    measurements,
    truth_model_list,
    animated_plots=False, 
    static_plots=True, 
    ospa=True, 
    extent_plot_step=4
    ):
    """Tests both phd or cphd, since both rely on same equations"""

    out = {}

    start = time.time()

    extraction_list = [] # for plotting without animated plot and OSPA calcs

    if animated_plots:
        fig, ax = plt.subplots(1,1)
        fig.set_figheight(9)
        fig.set_figwidth(15)

    for kk, t in enumerate(time_arr[:-1]):    
        
        filter.predict(t, filt_args=(dt,)) 
        filter.correct(timestep=t,meas_in=measurements[kk]) 
        filter.cleanup(enable_merge=True) 
        mix = filter.extracted_mixture[-1]  # last extracted mixture in list

        extraction_list.append(mix)

        if animated_plots:
            ax.clear()
            ax.plot([0],[0])
            ax.set_xlim((0,129))
            ax.set_ylim((0,129))
            for each_meas in measurements[kk]:
                ax.scatter(each_meas[0, :], each_meas[1, :], marker='.', label='sampled points',c='k',s=1.5)  
            ax.grid()
            mix.plot_confidence_extents(h=0.95, plt_inds=[0, 1], ax=ax, edgecolor='r', linewidth=1.5) 
            plt.pause(0.2)

    end = time.time() # Putting this before static plot so it doesn't affect the time ran

    if static_plots:
        fig, ax = plt.subplots(1,1)
        fig.set_figheight(9)
        fig.set_figwidth(15)

        ax.clear()
        ax.plot([0],[0])
        
        ax.grid()

        for idx, g in enumerate(extraction_list):
            if idx % extent_plot_step == 0 and g is not None:
                g.plot_confidence_extents(h=0.95, plt_inds=[0, 1], ax=ax, edgecolor='r', linewidth=1.5)

        extraction_list[-1].plot_confidence_extents(h=0.95, plt_inds=[0, 1], ax=ax, edgecolor='r', linewidth=1.5)

        for ggiw_lst in truth_model_list:
            for idx, g in enumerate(ggiw_lst):
                if idx % extent_plot_step == 0 and g is not None:
                    g.plot_confidence_extents(h=0, plt_inds=[0, 1], ax=ax, edgecolor='k', linewidth=1.5)

        # plt.savefig("ProbabilityHypothesisDensityFilterResults.png")
        # plt.show()

        out["static"] = plt.gcf()

    time_ran = end-start

    print(f"Time taken to run: {round(time_ran,4)} seconds ")

    if animated_plots:
        print("Note: Time taken to run will be off due to plotting!")
        plt.show()

    if ospa:
        truth_states = [[] for _ in range(len(truth_model_list[0]))] 
        true_ggiws = [[] for _ in range(len(truth_model_list[0]))] 
        for target in truth_model_list:
            for t, ggiw in enumerate(target):
                true_ggiws[t].append(ggiw)
                truth_states[t].append(ggiw.mean) 

        # truth_states = truth_states[0:-1] # Needed to be trimmed for whatever reason to match with filter time steps

        # true_ggiws = true_ggiws[0:-1]

        # filter.calculate_ospa(truth=truth_states,c=5,p=1)
 
        filter.calculate_extended_ospa(truth=true_ggiws, c=5, p=1, 
                                       c_gamma=50, c_x=10, c_X=5, w_gamma=1, w_x=1, w_X=1)

        figs = filter.plot_extended_ospa()

        # plt.show()

        out["ospa"] = plt.gcf()

        # figs["OSPA"].savefig("PHD-OSPA.png")
        # figs["OSPA_subs"].savefig("PHD-OSPA_subs.png")

    out["time"] = time_ran

    return out

def test_glmbs(
    filter,
    time_arr,
    dt,
    measurements,
    truth_model_list,
    animated_plots=False, 
    static_plots=True,
    ospa=True
    ):
    """Tests both glmb or jglmb, since both rely on same equations"""

    out = dict()

    start = time.time()

    if animated_plots:
        fig, ax = plt.subplots(1,1)
        fig.set_figheight(9)
        fig.set_figwidth(15)

    for kk, t in enumerate(time_arr[:-1]):    
        
        filter.predict(t, filt_args=(dt,)) 
        filter.correct(timestep=t,meas_in=measurements[kk]) 
        extract_kwargs = {"update": True, "calc_states": True} 
        filter.cleanup(extract_kwargs=extract_kwargs) 

        if animated_plots:
            ax.clear()
            ax.plot([0],[0])
            ax.set_xlim((0,129))
            ax.set_ylim((0,129))
            for each_meas in measurements[kk]:
                ax.scatter(each_meas[0, :], each_meas[1, :], marker='.', label='sampled points',c='k',s=1.5)  
            ax.grid()
            filter.plot_states_labels(ax=ax, plt_inds=[0, 1], extent_plot_step=4, marker = "none") 
            plt.pause(0.2)

    end = time.time()

    if static_plots:
        fig, ax = plt.subplots(1,1)
        fig.set_figheight(9)
        fig.set_figwidth(15)

        ax.grid()

        filter.plot_states_labels(ax=ax, plt_inds=[0, 1], extent_plot_step=4, marker = "none", true_GGIWs=truth_model_list)

        # plt.savefig("GeneralizedLabeledMultiBernoulliResults.png")

        # plt.show()

        out["static"] = plt.gcf()

    time_ran = end-start

    print(f"Time taken to run: {round(time_ran,4)} seconds ")

    if animated_plots:
        print("Note: Time taken to run will be off due to plotting!")
        plt.show()

    if ospa:
        truth_states = [[] for _ in range(len(truth_model_list[0]))] 
        true_ggiws = [[] for _ in range(len(truth_model_list[0]))] 
        for target in truth_model_list:
            for t, ggiw in enumerate(target):
                true_ggiws[t].append(ggiw)
                truth_states[t].append(ggiw.mean)

        # truth_states = truth_states[0:-1] # Needed to be trimmed for whatever reason to match with filter time steps

        filter.calculate_extended_ospa(truth=true_ggiws,c=5,p=1, 
                                       c_gamma=50, c_x=10, c_X=5, w_gamma=1, w_x=1, w_X=1)

        figs = filter.plot_extended_ospa()

        # figs["OSPA2"].savefig("GLMB-OSPA.png")
        # figs["OSPA2_subs"].savefig("GLMB-OSPA_subs.png")

        # plt.show()

        out["ospa"] = plt.gcf()

    out["time"] = time_ran

    return out

if __name__=="__main__":

    truth_kinematics = gdyn.DoubleIntegrator() 

    truth_model = GGIWMixture(alphas=[300.0, 300.0], 
                betas=[1.0, 1.0],
                means=[np.array([0, 70, 8, 1]).reshape((4, 1)), np.array([20, 30, 6, -0.25]).reshape((4, 1))],
                covariances=[np.diag([0,0,0,0]), np.diag([0,0,0,0])],
                IWdofs=[45, 45], 
                IWshapes=[np.array([[70, 0],[0, 200]]), np.array([[100, 0],[0, 270]])],
                weights=[1,1])

    dt = 0.5
    t0, t1 = 0, 16.5 + dt 
    time_arr = np.arange(t0, t1, dt) 

    phd, cphd, glmb, jglmb = initialize_filters(dt=dt)

    measurements, truth_model_list = truth_n_measurements(time_arr, truth_model, truth_kinematics, FOV_lim=[0, 129], min_max_samples=[10, 50])

    phd_out = test_phds(phd,time_arr,dt,measurements,truth_model_list,animated_plots=False,static_plots=True,ospa=True)

    phd_out["static"].savefig("PHD_results.png")
    phd_out["ospa"].savefig("PHD_ospa.png")

    cphd_out = test_phds(cphd,time_arr,dt,measurements,truth_model_list,animated_plots=False,static_plots=True,ospa=True)

    cphd_out["static"].savefig("CPHD_results.png")
    cphd_out["ospa"].savefig("CPHD_ospa.png")

    glmb_out = test_glmbs(glmb,time_arr,dt,measurements,truth_model_list,animated_plots=False,static_plots=True,ospa=True)

    glmb_out["static"].savefig("glmb_results.png")
    glmb_out["ospa"].savefig("glmb_ospa.png")

    # test_glmbs(jglmb,time_arr,dt,measurements,truth_model_list,animated_plots=False,static_plots=True)
    