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

import carbs.extended_targets.GGIW_Serums_Models as gmodels

DEG2RAD = 0.0174533

global_seed = 69
debug_plots = 0

obs_window = np.array([[0, 100],
                       [0, 100]]) 
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
def _gen_ellipse(x, shape):
    # Return array of X and Y coordinates of points on the edge of ellipse
    # x         | 4 x 1 numpy array state vector
    # shape     | 2 x 2 numpy array matrix of object ellipse extend

    eig_val, eig_vec = np.linalg.eig(shape)
    theta = np.linspace(0, 2*np.pi, 100)
    ellipsis = (np.sqrt(eig_val[None,:]) * eig_vec) @ [np.sin(theta), np.cos(theta)]
    ellipsis [0,:] += x[0]
    ellipsis [1,:] += x[2]
    return ellipsis

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
    return np.array([[1.0,  dt,     0,   0],
                     [0,   1.0,     0,   0],
                     [0,    0,    1.0,  dt],
                     [0,    0,      0, 1.0]])

def _lamda_fun(shape, rng):
    # Can implement lamda rate for meas based on shape matrix here
    rate = rng.integers(10, 30)
    return rate

def _shape_fun(shape, rng):
    # Can implement time varying shape matrix here
    return shape.copy()

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
        shape_mean : R x N numpy array
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
        state_cov : N x X numpu array
            Covariance matrix of state 
        shape_mean : R x R numpy array
            Base shape matrix for all agent. 
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
        updated_lamda = _lamda_fun(agent[2],rng)
        updated_shape = _shape_fun(agent[2],rng)
        out.append([updated_lamda, _state_mat_fun(tt,dt,"useless") @ agent[1], updated_shape])
    return out

def _update_true_agents(true_agents:list, tt:float, dt:float, b_model:toyExtendedAgentBirth, rng:np.random.Generator):
    # Propagate existing target
    out = _prop_true(true_agents, tt, dt, rng)

    # Add new targets
    if any(np.abs(tt - b_model.birth_time) < 1e-8):
        # Birth description
        # Random state norm distributed around state_mean
        # Random rate
        # Random shape with size sample around shape_mean and have random rotation
        x = b_model.state_mean + (rng.multivariate_normal(np.zeros((b_model.state_mean.shape[0])), b_model.state_cov)).reshape(4,1)
        shape_delta = rng.multivariate_normal(np.zeros((b_model.shape_mean.shape[0])), b_model.shape_cov)
        shape = b_model.shape_mean + np.diag(shape_delta)
        rot = rng.uniform(0,360)
        rot_shape = _rotate_shape_mat(shape, rot)
        rate = _lamda_fun(rot_shape, rng)
        out.append([rate, x.copy(), rot_shape.copy()])
    return out

def _draw_frame(true_agents, ax):
    for agent in true_agents:
        cur_ellipse = _gen_ellipse(agent[1], agent[2])
        ax.plot(cur_ellipse[0,:], cur_ellipse[1,:])
    return ax


def test_GGIW_PHD():
    print ("Test GGIW-PHD")
    
    rng = rnd.default_rng(global_seed)

    dt = 1
    t0, t1 = 0, 120 + dt

    num_agent = 3

    birth_time = np.array([t0, 50])

    state_mean = np.array([50.0, -0.1, 120.0, -1.5]).reshape((4,1))
    state_std = np.array([10.0, 0.2, 1.0, 0.1]).reshape((4,1))

    shape_mean = np.diag(np.array([30, 15]))
    shape_std = np.array([5.0, 5.0]).reshape((2,1))

    b_model = toyExtendedAgentBirth(num_agent, birth_time, state_mean, state_std, shape_mean, shape_std, rng)

    time = np.arange(t0, t1, dt)
    true_agents = []    # Each agent is a list [lambda, x, shape]
    global_true = []
    fig, ax = plt.subplots()
    artist = []
    for kk,tt in enumerate(time):
        ax.clear()
        ax.set_title(str(tt) + "s")
        ax.plot([0],[0])
        ax.set_xlim(tuple(obs_window[0,:]))
        ax.set_ylim(tuple(obs_window[1,:]))
        ax.set_aspect(1)
        true_agents = _update_true_agents(true_agents, tt, dt, b_model, rng)
        ax = _draw_frame(true_agents, ax)
        plt.pause(0.1)
    
        
        

if __name__ == "__main__":
    from timeit import default_timer as timer
    import matplotlib

    matplotlib.use("WebAgg")

    plt.close("all")

    debug_plots = True

    start = timer()
    #############################################
    # Test function here
    #############################################
    test_GGIW_PHD()
    

    ############################################
    end = timer()
    print("{:.2f} s".format(end - start))
    print("Close all plots to exit")
    plt.show()