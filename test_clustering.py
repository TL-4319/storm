import sys
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy

import scipy.stats as stats

import gncpy.filters as gfilts
import gncpy.dynamics.basic as gdyn
import gncpy.distributions as gdistrib

from carbs.extended_targets.GGIW_Serums_Models import GGIW, GGIWMixture

from carbs.extended_targets.GGIW_EKF import GGIW_ExtendedKalmanFilter
import carbs.extended_targets.GGIW_RFS as GGIW_RFS
import carbs_clustering

DEG2RAD = 0.0174533
RAD2DEG = 57.2958

PIX2REAL = 4.26 # rough conversion from pixel unit to metric unit: ISS LIS has ~550km swath and sensor is 129 x 129 pix 
REAL2PIX = 1/PIX2REAL

global_seed = 69
debug_plots = 0

lightning_prob = 1.0

obs_window = np.array([[0, 129],
                       [0, 129]]) 
# Observation window
# obs_window = [[x_low, x_high],
#               [y_low, y_high]]
#
#
#
#   [x_low, y_high] ---------------------- [x_high, y_high]
#           |                                     |
#           |                                     |
#           |                                     |
#           |                                     |   
#   [x_low, y_low]  ---------------------- [x_low, y_high]
#


## 2D Constant velocity model

# x = [pos_x, vel_x, pos_y, vel_y]
# A = [ 1 dt  0  0
#       0  1  0  0
#       0  0  1  dt
#       0  0  0  1 ]


## Shape states include 
# shape_state = [theta, theta_dot, major, major_dot, minor, minor_dot]

# A = [1 dt 0  0 0 0
#      0  1 0  0 0 0
#      0  0 1 dt 0 0
#      0  0 0  1 0 0
#      0  0 0  0 1 dt
#      0  0 0  0 0 1]
# Shape states will have random walk on state second derivative

def _draw_frame(true_agents, ax):
    for agent in true_agents:
        if agent[0] == 0:
            continue
        shape_mat = _params2shapemat(agent[2][2,0], agent[2][4,0], agent[2][0,0])
        cur_ellipse = _gen_ellipse(agent[1], shape_mat)
        ax.plot(cur_ellipse[0,:], cur_ellipse[1,:])
    return ax

def _gen_ellipse(x, shape):
    # Return array of X and Y coordinates of points on the edge of ellipse
    # x         | 4 x 1 numpy array state vector
    # shape     | 2 x 2 numpy array matrix of object ellipse extend

    eig_val, eig_vec = np.linalg.eig(shape)
    theta = np.linspace(0, 2*np.pi, 100)
    ellipsis = (np.sqrt(eig_val[None,:]) * eig_vec) @ [np.sin(theta), np.cos(theta)]
    ellipsis [0,:] += x[0]
    ellipsis [1,:] += x[1]
    return ellipsis

def _shapemat2params(shape):
    # Return the semimajor axes and rotation angle from x to the semimajor axis

    # Decompose shape to principal axes
    eig_val, eig_vec = np.linalg.eig(shape)
    semi_maj = np.linalg.norm(eig_vec[:,0])
    semi_min = np.linalg.norm(eig_vec[:,1])

    # Calc rotation from x to semimajor axis
    theta_deg = np.atan2(semi_maj[1],semi_maj[0]) * RAD2DEG

    return np.sqrt(semi_maj), np.sqrt(semi_min), theta_deg

def _params2shapemat(semi_maj, semi_min, theta_deg):
    # Return the shape matrix given ellipses principle axes and rotation from x to semi major axis

    sq_maj = semi_maj * semi_maj
    sq_min = semi_min * semi_min

    stheta = np.sin(DEG2RAD * theta_deg)
    ctheta = np.cos(DEG2RAD * theta_deg)

    rot_mat = np.array([[ctheta, -stheta],
                        [stheta, ctheta]])
    
    major_vector = np.array([[sq_maj],
                            [0]])
    minor_vector = np.array([[0],
                            [sq_min]])
    
    rot_eig_vecx = rot_mat @ major_vector.reshape((2,1))
    rot_eig_vecy = rot_mat @ minor_vector.reshape((2,1))
    rot_eigv_mat = np.hstack((rot_eig_vecx,rot_eig_vecy))
    inv_rot_eigv_mat = np.linalg.inv(rot_eigv_mat)
    return rot_eigv_mat @ np.diag(np.array([sq_maj, sq_min])) @inv_rot_eigv_mat

def _rotate_shape_mat (shape, rot_deg:float):
    stheta = np.sin(DEG2RAD * rot_deg)
    ctheta = np.cos(DEG2RAD * rot_deg)

    rot_mat = np.array([[ctheta, -stheta],
                        [stheta, ctheta]])
    
    # Decompose shape to principal axes
    eig_val, eig_vec = np.linalg.eig(shape)
    eig_vecx = eig_vec[:,0]
    eig_vecy = eig_vec[:,1]

    rot_eig_vecx = rot_mat @ eig_vecx.reshape((2,1))
    rot_eig_vecy = rot_mat @ eig_vecy.reshape((2,1))
    rot_eigv_mat = np.hstack((rot_eig_vecx,rot_eig_vecy))
    inv_rot_eigv_mat = np.linalg.inv(rot_eigv_mat)
    return rot_eigv_mat @ np.diag(eig_val) @ inv_rot_eigv_mat

def _state_mat_fun(t, dt, useless):
    return np.array([[1.0,  0.0,     dt,   0],
                     [0,   1.0,     0,   dt],
                     [0,    0,    1.0,  0],
                     [0,    0,      0, 1.0]])

def _shape_mat_fun(t, dt, useless):
    return np.array ([[1, dt, 0,  0, 0, 0],
                      [0,  1, 0,  0, 0, 0],
                      [0,  0, 1, dt, 0, 0],
                      [0,  0, 0,  1, 0, 0],
                      [0,  0, 0,  0, 1, dt],
                      [0,  0, 0,  0, 0, 1]])

def _shape_mat_acc(t, dt, useless):
    return np.array ([[0,  0, 0,  0, 0, 0],
                      [0,  dt * 0.1, 0,  0, 0, 0],
                      [0,  0, 0, 0, 0, 0],
                      [0,  0, 0,  dt * 0.1, 0, 0],
                      [0,  0, 0,  0, 0, 0],
                      [0,  0, 0,  0, 0, dt * 0.1]])

def _lamda_fun(shape, rng):
    # Implement Flash Rate Parameterization Scheme using updraft volume model https://doi.org/10.1029/2007JD009598
    # f (flash/min) = 6.75 x 1e-11 x vol (m^3) - 13.9
    # Assume that the volume span the storm area and reach from 10km to 15km in altitude (roughly)
    
    shape_mat = _params2shapemat(abs(shape[2,0]), abs(shape[4,0]), shape[0,0])
    # Updraft volume calc
    eig, eig_val = np.linalg.eig(shape_mat)
    uv_m3 = (np.prod(eig) * np.pi * PIX2REAL**2) * 5 * 1e9

    # flash rate per sec
    f_ps = (6.75 * uv_m3 * 1e-11 - 13.9)/60
    return max(0,f_ps)


class toyExtendedAgentBirth(object):
    """Object containing the birth model for toy example"""
    def __init__(self, num_agent:int, birth_time:np.ndarray, state_mean:np.ndarray, state_std:np.ndarray, shape_mean:np.ndarray, shape_std:np.ndarray, rng:np.random):
        """ Initialize
        Generate multiple agent with random birth time and intial mean/shape that is sampled around the base_mean/base_shape

        Input Parameters
        -----------
        num_agent : int
            Number of agents to simulate and birth
        birth_time : 2 x 1 numpy array
            Range of time where births happen
        state_mean : N x 1 numpy array
            Mean of the state vector for all agent
        state_std : N x 1 numpy array
            Standard variation of the birth state vector
        shape_mean : R x 1 numpy array
            Mean shape matrix for all agent. 
        shape_std : R x 1 numpy array
            Standard deviation of agent shape
        rng : numpy.random object
            Seeded random number generator

        Member Parameters
        -----------
        birth_time : num_agent x 1 numpy array
            Birth time of of each agent
        state_mean : N x 1 numpy array
            Mean of the state vector for all agent
        state_cov : N x N numpu array
            Covariance matrix of state 
        shape_mean : R x 1 numpy array
            Base shape state for all agent. 
        shape_cov : R x R numpy array
            Covariance matrix of shape
        """

        self.birth_time = rng.integers(birth_time[0], birth_time[1], num_agent)
        self.state_mean = state_mean
        self.shape_mean = shape_mean
        self.state_cov = np.diag(np.square(state_std).flatten())
        self.shape_cov = np.diag(np.square(shape_std).flatten())

def _prop_true(true_agents, tt, dt, rng):
    
    if true_agents is None:
        return []

    out = []
    for agent in true_agents:
        # Update kinematic state
        updated_state = _state_mat_fun(tt,dt,"useless") @ agent[1]

        #Update shape state with random accel on rot, and principle axes
        updated_shape_state = _shape_mat_fun(tt,dt,"useless") @ agent[2] + \
            _shape_mat_acc(tt, dt, "useless") @ rng.multivariate_normal(np.zeros(6), np.diag(np.array([0, 0.2, 0, 0.0001, 0, 0.0001]))).reshape(6,1)
        print(updated_shape_state)

        # Update lambda based on shape
        updated_lamda = _lamda_fun(updated_shape_state,rng)
        

        out.append([updated_lamda, updated_state, updated_shape_state])
    return out

def _update_true_agents(true_agents:list, tt:float, dt:float, b_model:toyExtendedAgentBirth, rng:np.random.Generator):
    # Propagate existing target
    out = _prop_true(true_agents, tt, dt, rng)

    # Add new targets
    if any(np.abs(tt - b_model.birth_time) < 1e-8):
        # Birth description
        # Random state norm distributed around state_mean
        x = b_model.state_mean + (rng.multivariate_normal(np.zeros((b_model.state_mean.shape[0])), b_model.state_cov)).reshape(4,1)

        # Random shape state
        shape_delta = rng.multivariate_normal(np.zeros((b_model.shape_mean.shape[0])), b_model.shape_cov)
        shape_state = b_model.shape_mean + shape_delta.reshape(6,1)
        shape_state[0,0] = rng.uniform(0,360) # Uniformly random rotation angle

        # Calculate lamda rate based on current shape
        rate = _lamda_fun(shape_state, rng)
        out.append([rate, x.copy(), shape_state.copy()])
    return out

def _check_in_FOV(true_agents, obs_window):
    agent_in_FOV = []
    for agent in true_agents:
        shape_mat = _params2shapemat(agent[2][2,0], agent[2][4,0], agent[2][0,0])
        el = _gen_ellipse(agent[1], shape_mat)
        is_in_FOV = np.any(el[0,:] > obs_window[0,0]) and np.any(el[0,:] < obs_window[0,1]) and \
            np.any(el[1,:] > obs_window[1,0]) and np.any(el[1,:] < obs_window[1,1]) and agent[0] > 0
        if is_in_FOV:
            agent_in_FOV.append(deepcopy(agent))
    return agent_in_FOV
        
def _gen_extented_meas(tt, agents_in_FOV, obs_window, rng:np.random.Generator):
    meas_in = []
    
    for agent in agents_in_FOV:
        if rng.uniform(0, 1) > lightning_prob:
            continue
        num_meas = rng.poisson(agent[0])
        agent_pos = np.array([agent[1][0,0], agent[1][1,0]])
        shape_mat = _params2shapemat(agent[2][2,0], agent[2][4,0], agent[2][0,0])
        m = rng.multivariate_normal(agent_pos, 0.25 * shape_mat,num_meas).reshape(num_meas,2).transpose().round() # Detection are rounded to int to simulate pixel index
        m = np.unique(m,axis=1) # Cull repeated measurement 
        # Cull any measurment outside of FOV
        out_FOV = np.where(np.logical_or(m[1,:] < obs_window[1,0], m[1,:] > obs_window[1,1]))
        out_FOV = out_FOV[0]
        m = np.delete(m, out_FOV, 1)
        out_FOV = np.where(np.logical_or(m[0,:] < obs_window[0,0], m[0,:] > obs_window[0,1]))
        out_FOV = out_FOV[0]
        m = np.delete(m, out_FOV, 1)
        if m.shape[1] > 0:
            # This to ensure list structure is identical
            for ii in range (m.shape[1]):
                meas_in.append(m[:,ii].copy().reshape(2,1)) 
    return meas_in

def test_clustering():
    rng = rnd.default_rng(global_seed)

    tt = 0

    num_agent = 5
    birth_time = np.array([0, 20])

    state_mean = np.array([65.0, 65.0, 0.0, -2.0]).reshape((4,1))
    state_std = np.array([30.0, 30.0, 0.0, 0.1]).reshape((4,1))

    shape_mean = np.array([0, 0, (20 * REAL2PIX), 0, (20 * REAL2PIX), 0]).reshape((6,1)) # Assume average storm diameter of 24km with some variation in shape
    shape_std = np.array([0,1, 10 * REAL2PIX, 0, 10 * REAL2PIX,0]).reshape((6,1))

    b_model = toyExtendedAgentBirth(num_agent, birth_time, state_mean, state_std, shape_mean, shape_std, rng)

    targets = []

    for ii in range(num_agent):
        # Birth description
        # Random state norm distributed around state_mean
        x = b_model.state_mean + (rng.multivariate_normal(np.zeros((b_model.state_mean.shape[0])), b_model.state_cov)).reshape(4,1)

        # Random shape state
        shape_delta = rng.multivariate_normal(np.zeros((b_model.shape_mean.shape[0])), b_model.shape_cov)
        shape_state = b_model.shape_mean + shape_delta.reshape(6,1)
        shape_state[0,0] = rng.uniform(0,360) # Uniformly random rotation angle

        # Calculate lamda rate based on current shape
        rate = _lamda_fun(shape_state, rng)
        targets.append([rate, x.copy(), shape_state.copy()])
    
    meas_gen = _gen_extented_meas(tt, targets, obs_window, rng)

    # Create clustering parameter object
    clustering_params = carbs_clustering.DBSCANParameters(eps=3, min_samples=5, ignore_noise=True)

    # Create clustering object using the parameters above
    clustering = carbs_clustering.MeasurementClustering(clustering_params)
    
    # Partition a set.
    meas_list = clustering.cluster(meas_gen)

    # For visualization
    fig, ax = plt.subplots(figsize=(10,10))
    ax.clear()
    ax.plot([0],[0])
    ax.set_xlim(tuple(obs_window[0,:]))
    ax.set_ylim(tuple(obs_window[1,:]))
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.set_aspect(1)
    #ax = _draw_frame(targets, ax)
    #for meas in meas_gen:
    #    ax.scatter(meas[0,:],meas[1,:], 10, "r" ,marker="*")
    for ii in range(len(meas_list)):
        cluster = meas_list[ii]
        cluster_array = np.array(cluster).reshape(len(cluster), 2).transpose()
        ax.scatter(cluster_array[0,:],cluster_array[1,:], 20 ,marker="*")
        centroid_pos = np.mean(cluster_array, axis=1)
        #ax.scatter(centroid_pos[0] ,centroid_pos[1], 20 ,marker="o")
        ax.text(centroid_pos[0] ,centroid_pos[1]+1, str(ii),dict(size=20))



if __name__ == "__main__":
    from timeit import default_timer as timer

    plt.close("all")

    debug_plots = True

    start = timer()
    #############################################
    # Test function here
    #############################################
    test_clustering()


    ############################################
    end = timer()
    print("{:.2f} s".format(end - start))
    print("Close all plots to exit")
    plt.show()