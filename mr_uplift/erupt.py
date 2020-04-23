import numpy as np
import pandas as pd
import random


def weighted_avg_and_std(values, weights):
    """computes weighted averages and stdevs
    Args:
        values (np.array): variable of interest
        weight (np.array): weight to assign each observation
    Returns:
        list of weighted mean and stdev
    """

    average = np.average(values, weights=weights)

    variance = np.average((values - average)**2, weights=weights)
    ess = np.sum(weights)**2 / np.sum(weights**2)

    return [average, np.sqrt(variance) / np.sqrt(ess)]


def erupt(y, tmt, optim_tmt, weights=None, names=None):
    """Calculates Expected Response Under Proposed Treatments
    Args:
        y (np.array): response variables
        tmt (np.array): treatment randomly assigned to observation
        optim_tmt (np.array): proposed treatment model assigns
        weights (np.array): weights for each observation. useful if treatment
            was not uniformly assigned
        names (list of str): names of response variables
    Returns:
        Expected means and stdevs of proposed treatments
    """

    if weights is None:
        weights = np.ones(y.shape[0])

    equal_locs = np.where(optim_tmt == tmt)[0]

    erupts = [weighted_avg_and_std(y[equal_locs][:,
                                                 x].reshape(-1,
                                                            1),
                                   weights[equal_locs].reshape(-1,
                                                               1)) for x in range(y.shape[1])]

    erupts = pd.DataFrame(erupts)
    erupts.columns = ['mean', 'std']

    if names is not None:
        erupts['response_var_names'] = names
    else:
        erupts['response_var_names'] = [
            'var_' + str(x) for x in range(y.shape[1])]

    return erupts


def get_best_tmts(objective_weights, ice, unique_tmts):
    """Calulcates Optimal Treatment for a set of counterfactuals and weights.
    Args:
        objective_weights (np.array): list of weights to maximize in objective function
        ice (np.array): 3-d array of counterfactuals.
            num_tmts x num_observations x num_responses
        unique_tmts (np.array): list of treatments
    Returns:
        Optimal Treatment with specific objective function
    """

    weighted_ice = [x * objective_weights for x in ice]
    sum_weighted_ice = np.array(weighted_ice).sum(axis=2)

    max_values = sum_weighted_ice.T.argmax(axis=1)

    best_tmt = unique_tmts[np.array(max_values)]

    return(best_tmt)


def get_weights(tmts):
    """Calculates weights to apply to tmts. Weight is inversely proportional to
        how common each weight is.
    Args:
        tmts (np.array): treatments randomly assigned to
    Returns:
        array of length tmts that is weight to assign in erupt calculation
    """
    tmts_pd = pd.DataFrame(tmts)
    tmts_pd.columns = ['tmt']
    weight_mat = pd.DataFrame(pd.DataFrame(
        tmts).iloc[:, 0].value_counts() / len(tmts))
    weight_mat['tmt'] = weight_mat.index
    weight_mat.columns = ['weight', 'tmt']
    tmts_pd = tmts_pd.merge(weight_mat, on='tmt', how='left')
    observation_weights = 1.0 / np.array(tmts_pd['weight']).reshape(-1, 1)

    return observation_weights


def get_erupts_curves_aupc(y, tmt, ice, unique_tmts, objective_weights,
                           names=None):
    """Calculates optimal treatments and returns erupt
    Args:
        y (np.array): response variables
        tmt (np.array): treatment randomly assigned to observation
        ice (np.array): 3-d array of counterfactuals.
            num_tmts x num_observations x num_responses
        objective_weights (np.array): list of weights to maximize in objective
        function
        names (list of str): names of response variables
    Returns:
        Dataframe of ERUPT metrics for each response variable for a given set
        of objective weights
    """

    all_erupts = []
    all_distributions = []
    observation_weights = get_weights(tmt)

    for obj_weight in objective_weights:

        optim_tmt = get_best_tmts(obj_weight, ice, unique_tmts)
        random_tmt = optim_tmt.copy()[
            np.random.choice(
                len(optim_tmt),
                len(optim_tmt),
                replace=False)]

        str_obj_weight = ','.join([str(q) for q in obj_weight])

        erupts = erupt(y, tmt, optim_tmt, weights=observation_weights,
                       names=names)

        erupts_random = erupt(y, tmt, random_tmt, weights=observation_weights,
                              names=names)

        erupts_random['weights'] = str_obj_weight
        erupts['weights'] = str_obj_weight

        erupts_random['assignment'] = 'random'
        erupts['assignment'] = 'model'

        erupts = pd.concat([erupts, erupts_random], axis=0)

        dists = pd.DataFrame(optim_tmt).iloc[:, 0].value_counts()
        dist_treatments = pd.DataFrame(np.array(dists).reshape(-1, 1))
        dist_treatments['tmt'] = dists.index
        dist_treatments['weights'] = str_obj_weight
        dist_treatments['percent_tmt'] = np.array(dists) / np.array(dists).sum()
        dist_treatments.columns = ['num_observations', 'tmt', 'weights',
                                   'percent_tmt']

        all_erupts.append(erupts)
        all_distributions.append(dist_treatments)

    all_erupts = pd.concat(all_erupts)
    all_distributions = pd.concat(all_distributions)

    return all_erupts, all_distributions
