import sys
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
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

def _calc_shape_mat(shape, rot_deg:float):
    stheta = np.sin(DEG2RAD * rot_deg)
    ctheta = np.cos(DEG2RAD * rot_deg)

    rot_mat = np.array([[ctheta, -stheta],
                        [stheta, ctheta]])
    return rot_mat @ shape

def _state_mat_fun(t, dt, useless):
    return np.array([[1.0,  dt,     0,   0],
                     [0,   1.0,     0,   0],
                     [0,    0,    1.0,  dt],
                     [0,    0,      0, 1.0]])

def _lamda_fun(shape):
    # Can implement lamda rate for meas based on shape matrix here
    return 1

def _shape_fun(shape):
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

def _prop_true(true_agents, tt, dt):
    
    if true_agents is None:
        return []

    out = []
    for agent in true_agents:
        updated_lamda = _lamda_fun(agent[2])
        updated_shape = _shape_fun(agent[2])
        out.append([updated_lamda, _state_mat_fun(tt,dt,"useless") @ agent[1], updated_shape])
    return out

def _update_true_agents(true_agents:list, tt:float, dt:float, b_model:toyExtendedAgentBirth, rng:np.random.Generator):
    # Propagate existing target
    out = _prop_true(true_agents, tt, dt)

    # Add new targets
    if any(np.abs(tt - b_model.birth_time) < 1e-8):
        x = b_model.state_mean + (rng.multivariate_normal(np.zeros((b_model.state_mean.shape[0])), b_model.state_cov))
        shape_delta = rng.multivariate_normal(np.zeros((b_model.shape_mean.shape[0])), b_model.shape_cov)
        shape = b_model.shape_mean + np.diag(shape_delta)
        rate = rng.integers(10, 30)
        out.append([rate, x.copy(), shape.copy()])
    return out

def test_GGIW_PHD():
    print ("Test GGIW-PHD")
    
    rng = rnd.default_rng(global_seed)

    dt = 1
    t0, t1 = 0, 100 + dt

    num_agent = 3

    birth_time = np.array([t0, 25])

    state_mean = np.array([50.0, -2.0, 120.0, -1.0]).reshape((4,1))
    state_std = np.array([30.0, 1, 1.0, 0]).reshape((4,1))

    shape_mean = np.diag(np.array([30, 10]))
    shape_std = np.array([10.0, 10.0]).reshape((2,1))

    b_model = toyExtendedAgentBirth(num_agent, birth_time, state_mean, state_std, shape_mean, shape_std, rng)

    time = np.arange(t0, t1, dt)
    true_agents = []    # Each agent is a list [lambda, x, shape]
    global_true = []
    for kk,tt in enumerate(time):
        true_agents = _update_true_agents(true_agents, tt, dt, b_model, rng)
        

if __name__ == "__main__":
    from timeit import default_timer as timer
    import matplotlib

    #matplotlib.use("WebAgg")

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