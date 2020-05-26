import numpy as np


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
