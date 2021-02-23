import numpy as np
import pandas as pd

def get_simple_uplift_data(num_obs):
    """Creates sample uplift dataset with 3 variables.
    First two variables are of form y_i = x_i*t + e for two responses
    Thrid response is just noise

    Args:
        num_obs (int): number of observations to simulate from
    Returns:
        responses, explanatory variables, and treatment
    """

    tmt = np.random.binomial(1, .5, num_obs)
    x = np.concatenate([np.random.uniform(0, 1, num_obs).reshape(-1, 1),
                        np.random.uniform(0, 1, num_obs).reshape(-1, 1)], axis=1)

    y_1 = tmt * x[:, 0] + np.random.normal(0, .1, num_obs)
    y_2 = tmt * x[:, 1] + np.random.normal(0, .1, num_obs)
    y_3 = np.random.normal(0, 1, num_obs).reshape(-1, 1)

    y = np.concatenate(
        [y_1.reshape(-1, 1), y_2.reshape(-1, 1), y_3.reshape(-1, 1)], axis=1)

    return y, x, tmt


def get_sin_uplift_data(num_obs):
    """Creates sample uplift dataset with 3 variables.
    First two variables are of form y_i = x_i*t + e for two responses
    Thrid response is just noise

    Args:
        num_obs (int): number of observations to simulate from
    Returns:
        responses, explanatory variables, and treatment
    """

    tmt = np.random.binomial(1, .5, num_obs)
    x = np.concatenate([np.random.uniform(0, 1, num_obs).reshape(-1,1),
    np.random.uniform(0, 1, num_obs).reshape(-1,1)], axis = 1)

    y_1 = np.sin(tmt*20*x[:,0]) + np.random.normal(0, .1, num_obs)
    y_2 = np.sin(tmt*20*x[:,1]) + np.random.normal(0, .1, num_obs)
    y_3 =  np.random.normal(0, 1, num_obs).reshape(-1,1)

    y = np.concatenate([y_1.reshape(-1,1), y_2.reshape(-1,1),y_3.reshape(-1,1)], axis = 1)

    return y, x, tmt



def get_no_noise_data(num_obs):
    """Creates sample uplift dataset with 3 variables.
    First two variables are of form y_i = x_i*t for two responses
    Thrid response is just noise

    Args:
        num_obs (int): number of observations to simulate from
    Returns:
        responses, explanatory variables, and treatment
    """

    tmt = np.random.binomial(1, .5, num_obs)
    x = np.concatenate([np.random.uniform(0, 1, num_obs).reshape(-1, 1),
                        np.random.uniform(0, 1, num_obs).reshape(-1, 1)], axis=1)

    noise = np.random.normal(0, .1, num_obs)

    y_1 = 2 * tmt * x[:, 0]
    y_2 = tmt * x[:, 1]

    y = np.concatenate([y_1.reshape(-1, 1), y_2.reshape(-1, 1), ], axis=1)

    return y, x, tmt


def get_no_noise_data_2(num_obs):
    """Creates sample uplift dataset with 3 variables.
    First two variables are of form y_i = x_i*t for two responses
    Thrid response is just noise

    Args:
        num_obs (int): number of observations to simulate from
    Returns:
        responses, explanatory variables, and treatment
    """

    tmt = np.random.choice([0, 1, 2, 3], num_obs,
                           [.25, .25, .25, .25]).reshape(-1, 1)
    x = np.concatenate([
        np.random.uniform(0, 1, num_obs).reshape(-1, 1),
        np.random.uniform(0, 1, num_obs).reshape(-1, 1),
        np.random.uniform(0, 1, num_obs).reshape(-1, 1),
        np.random.uniform(0, 1, num_obs).reshape(-1, 1),

    ], axis=1)

    noise = np.random.normal(0, .1, num_obs)

    y = 1 * (tmt == 0) * x[:, 0].reshape(-1, 1) + (tmt == 0) * 1 + \
        2 * (tmt == 1) * x[:, 1].reshape(-1, 1) + (tmt == 1) * 2 + \
        3 * (tmt == 2) * x[:, 2].reshape(-1, 1) + (tmt == 2) * 3 + \
        4 * (tmt == 3) * x[:, 3].reshape(-1, 1) + (tmt == 3) * 4

    return y, x.T, tmt

def get_observational_uplift_data(num_obs):
    """Creates sample uplift dataset with 3 variables.
    First two variables are of form y_i = x_i*t + e for two responses
    Thrid response is just noise. Treatment is a function of x_1

    Args:
        num_obs (int): number of observations to simulate from
    Returns:
        responses, explanatory variables, and treatment
    """

    x = np.concatenate([np.random.uniform(0, 1, num_obs).reshape(-1, 1),
                        np.random.uniform(0, 1, num_obs).reshape(-1, 1)], axis=1) - .5

    tmt = np.random.binomial(1, .2+.6*(x[:,0]>0), num_obs)
    tmt[np.where(x[:,0]< -.25)] = 0

    y_1 = -tmt * x[:, 0] + np.random.normal(0, .1, num_obs)
    y_2 = tmt * x[:, 1] + np.random.normal(0, .1, num_obs)
    y_3 = np.random.normal(0, 1, num_obs).reshape(-1, 1)

    y = np.concatenate(
        [y_1.reshape(-1, 1), y_2.reshape(-1, 1), y_3.reshape(-1, 1)], axis=1)

    return y, x, tmt



def get_observational_uplift_data(num_obs):
    """Creates sample uplift dataset with 3 variables.
    First two variables are of form y_i = x_i*t + e for two responses
    Thrid response is just noise. Treatment is a function of x_1

    Args:
        num_obs (int): number of observations to simulate from
    Returns:
        responses, explanatory variables, and treatment
    """

    segment = np.random.choice(['a','b'], size = num_obs)
    segment = pd.get_dummies(segment)
    x = np.concatenate([np.random.uniform(0, 1, num_obs).reshape(-1, 1),
                        np.random.uniform(0, 1, num_obs).reshape(-1, 1)],
                        axis=1)

    rule_assignment = np.random.choice(['control','assignment_1','assignment_2'], size = num_obs, p=[.1, 0.45, 0.45] )
    tmt = np.zeros(num_obs)
    #model_1
    tmt[np.where((rule_assignment=='assignment_1')*(x[:,0]<.33))[0]] = 1
    tmt[np.where((rule_assignment=='assignment_1')*(x[:,0]>.33))[0]] = 2
    #model_2
    tmt[np.where((rule_assignment=='assignment_2')*(x[:,0]<.5))[0]] = 1
    tmt[np.where((rule_assignment=='assignment_2')*(x[:,0]>.5))[0]] = 2
    #control
    tmt[np.where(x[:,1]>.8)] = 0

    tmt = pd.get_dummies(tmt)

    y = np.array((segment['a'] == 1) * (tmt[1.0])).astype('float')
    y = y + np.array((segment['b'] == 1) * (tmt[2.0])).astype('float')

    y = y - np.array((segment['b'] == 1) * (tmt[1.0])).astype('float')
    y = y - np.array((segment['a'] == 1) * (tmt[2.0])).astype('float')

    y = np.array(y).reshape(-1,1)

    x = np.concatenate([np.array(segment), x], axis = 1)

    tmt = np.array(tmt)

    return y, x, tmt, rule_assignment
