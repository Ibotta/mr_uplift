import numpy as np
import pytest

from mr_uplift.dataset.data_simulation import get_no_noise_data, get_simple_uplift_data, get_observational_uplift_data
from mr_uplift.mr_uplift import MRUplift, get_t_data
from mr_uplift.keras_model_functionality import prepare_data_optimized_loss
import sys
import pandas as pd

class TestMRUplift(object):

    def test_get_t_data(self):

        num_obs_1 = 10
        num_obs_2 = 3

        test_1 = get_t_data(0, num_obs_1)
        test_2 = get_t_data(np.array([0, 1]), num_obs_2)

        test_1_values = np.zeros(num_obs_1).reshape(-1, 1)
        test_2_values = np.concatenate([np.zeros(num_obs_2).reshape(-1, 1),
                                        np.ones(num_obs_2).reshape(-1, 1)], axis=1)

        assert np.mean(test_1 == test_1_values) == 1
        assert np.mean(test_2 == test_2_values) == 1

    def test_model_mean_outputs(self):
        true_ATE = np.array([[0, 0], [1, .5]])
        rmse_tolerance = .05
        num_obs = 10000

        y_no_noise, x_no_noise, tmt_no_noise = get_no_noise_data(num_obs)

        uplift_model = MRUplift()
        uplift_model.fit(x_no_noise, y_no_noise, tmt_no_noise.reshape(-1, 1),
                         n_jobs=1)
        oos_ice = uplift_model.predict_ice(response_transformer = True)

        assert np.sqrt(np.mean((oos_ice.mean(axis=1) -true_ATE)**2)) < rmse_tolerance

    def test_model_pred_oos_shapes(self):
        num_obs = 1000

        y, x, t = get_simple_uplift_data(num_obs)

        t = np.concatenate([t.reshape(-1, 1),
        np.random.binomial(1, .5, num_obs).reshape(-1, 1)], axis=1)
        param_grid = dict(num_nodes=[8], dropout=[.1], activation=[
                                  'relu'], num_layers=[1], epochs=[1], batch_size=[1000])

        uplift_model = MRUplift()
        uplift_model.fit(x, y, t, param_grid = param_grid, n_jobs=1)

        assert uplift_model.predict_ice().shape == (
            np.unique(t, axis=0).shape[0], num_obs * .7, y.shape[1])

        assert uplift_model.predict_ice(x=x).shape == (np.unique(t,axis=0).shape[0],
            num_obs,
            y.shape[1])

        assert uplift_model.get_erupt_curves()

        assert uplift_model.get_erupt_curves(x = x, y = y, t = t)

    def test_model_pred_oos_shapes_single_col_tmt(self):
        num_obs = 1000

        y, x, t = get_simple_uplift_data(num_obs)

        t = t.reshape(-1, 1)

        param_grid = dict(num_nodes=[8], dropout=[.1], activation=[
                                  'relu'], num_layers=[1], epochs=[1], batch_size=[1000])

        uplift_model = MRUplift()
        uplift_model.fit(x, y, t, param_grid = param_grid, n_jobs=1)

        assert uplift_model.predict_ice().shape == (
            np.unique(t, axis=0).shape[0], num_obs * .7, y.shape[1])

        assert uplift_model.predict_ice(x=x).shape == (np.unique(t,axis=0).shape[0],
            num_obs,
            y.shape[1])

        assert uplift_model.get_erupt_curves()

        assert uplift_model.get_erupt_curves(x = x, y = y, t = t)

    def test_model_pred_oos_shapes_single_col_tmt_propensity(self):
        num_obs = 1000

        y, x, t = get_simple_uplift_data(num_obs)

        t = t.reshape(-1, 1)

        param_grid = dict(num_nodes=[8], dropout=[.1], activation=['relu'], num_layers=[2],
                                epochs=[1], batch_size=[100],
                                  alpha = [.5], copy_several_times = [1])

        uplift_model = MRUplift()
        uplift_model.fit(x, y, t, param_grid = param_grid, n_jobs=1,
            optimized_loss = True, use_propensity = True)

        assert uplift_model.predict_ice().shape == (
            np.unique(t, axis=0).shape[0], num_obs * .7, y.shape[1])

        assert uplift_model.predict_ice(x=x).shape == (np.unique(t,axis=0).shape[0],
            num_obs,
            y.shape[1])

        assert uplift_model.get_erupt_curves()

        assert uplift_model.get_erupt_curves(x = x, y = y, t = t)


    def test_prepare_data_optimized_loss_one_col_tmt(self):
        num_obs = 1000

        y, x, t = get_simple_uplift_data(num_obs)
        t = t.reshape(len(t),1)

        unique_treatments = np.unique(t, axis = 0)

        masks = np.ones(num_obs).reshape(num_obs,1)

        x, utility_weights, missing_utility, missing_y_mat, masks, weights = prepare_data_optimized_loss(x,y,t, masks ,unique_treatments)

        assert(utility_weights.shape == (num_obs, y.shape[1]))
        assert(missing_y_mat.shape == (num_obs, unique_treatments.shape[0], y.shape[1]))
        for q in range(unique_treatments.shape[0]):
            assert( ((missing_utility[:,q]==0) == (missing_y_mat[:,q,0] == -999)).mean() ==1 )

    def test_prepare_data_optimized_loss_two_col_tmt(self):
        num_obs = 1000

        y, x, t = get_simple_uplift_data(num_obs)

        t = np.concatenate([t.reshape(-1, 1),
        np.random.binomial(1, .5, num_obs).reshape(-1, 1)], axis=1)
        unique_treatments = np.unique(t, axis = 0)
        masks = np.ones(num_obs*len(unique_treatments)).reshape(num_obs,len(unique_treatments))

        x, utility_weights, missing_utility, missing_y_mat, masks, weights = prepare_data_optimized_loss(x,y,t,masks, unique_treatments)

        assert(utility_weights.shape == (num_obs, y.shape[1]))
        assert(missing_y_mat.shape == (num_obs, unique_treatments.shape[0], y.shape[1]))
        for q in range(unique_treatments.shape[0]):
            assert( ((missing_utility[:,q]==0) == (missing_y_mat[:,q,0] == -999)).mean() ==1 )

    def test_model_optim_mean_outputs(self):
        true_ATE = np.array([[0, 0], [1, .5]])
        rmse_tolerance = .05
        num_obs = 10000
        param_grid = dict(num_nodes=[8], dropout=[.1], activation=['relu'], num_layers=[2], epochs=[30], batch_size=[100])

        y_no_noise, x_no_noise, tmt_no_noise = get_no_noise_data(num_obs)

        uplift_model = MRUplift()
        uplift_model.fit(x_no_noise, y_no_noise, tmt_no_noise.reshape(-1, 1),
                         n_jobs=1, param_grid = param_grid, optimized_loss = False)
        oos_ice = uplift_model.predict_ice(response_transformer = True)

        assert np.sqrt(np.mean((oos_ice.mean(axis=1) - true_ATE)**2)) < rmse_tolerance

    def test_model_get_random_erupts(self):
        true_ATE = np.array([[0, 0], [1, .5]])
        rmse_tolerance = .05
        num_obs = 10000
        param_grid = dict(num_nodes=[8], dropout=[.1], activation=['relu'], num_layers=[2], epochs=[30], batch_size=[100],
                                  alpha = [.5], copy_several_times = [2])

        y_no_noise, x_no_noise, tmt_no_noise = get_no_noise_data(num_obs)

        uplift_model = MRUplift()
        uplift_model.fit(x_no_noise, y_no_noise, tmt_no_noise.reshape(-1, 1),
                         n_jobs=1, param_grid = param_grid, optimized_loss = True)
        oos_re = uplift_model.get_random_erupts()

        uplift_model_propensity = MRUplift()
        uplift_model_propensity.fit(x_no_noise, y_no_noise, tmt_no_noise.reshape(-1, 1),
                         n_jobs=1, param_grid = param_grid,
                         optimized_loss = True, use_propensity = True)
        oos_re_propensity = uplift_model_propensity.get_random_erupts()


        assert oos_re['mean'].iloc[0] > 0
        assert oos_re_propensity['mean'].iloc[0] > 0

    def test_varimp(self):
        num_obs = 10000
        param_grid = dict(num_nodes=[8], dropout=[.1], activation=['relu'], num_layers=[2], epochs=[30], batch_size=[100])

        y, x, t = get_simple_uplift_data(num_obs)

        uplift_model = MRUplift()
        uplift_model.fit(x, y, t.reshape(-1, 1),
                         n_jobs=1, param_grid = param_grid)
        varimp = uplift_model.permutation_varimp(objective_weights = np.array([.7,-.3,0]).reshape(1,-1))

        param_grid = dict(num_nodes=[8], dropout=[.1], activation=['relu'], num_layers=[2], epochs=[30], batch_size=[100],
                                  alpha = [.5], copy_several_times = [2])

        uplift_model_propensity = MRUplift()
        uplift_model_propensity.fit(x, y, t.reshape(-1, 1),
                         n_jobs=1, param_grid = param_grid,
                         optimized_loss = True, use_propensity = True)
        varimp_propensity = uplift_model_propensity.permutation_varimp(objective_weights = np.array([.7,-.3,0]).reshape(1,-1))

        assert varimp['permutation_varimp_metric'].iloc[0]>varimp['permutation_varimp_metric'].iloc[1]
        assert varimp_propensity['permutation_varimp_metric'].iloc[0]>varimp_propensity['permutation_varimp_metric'].iloc[1]


    def test_model_pred_oos_shapes_single_col_tmt_propensity(self):
        num_obs = 10000
        propensity_score_cutoff = 100
        TOLERANCE = .98
        y, x, t, rule_assignment = get_observational_uplift_data(num_obs)

        param_grid = dict(num_nodes=[8], dropout=[.1], activation=['relu'], num_layers=[1],
                                epochs=[20], batch_size=[512],
                                  alpha = [.9999,.99], copy_several_times = [1])

        uplift_model = MRUplift()
        uplift_model.fit(x, y[:,0].reshape(-1,1), t, param_grid = param_grid, n_jobs=1,
            optimized_loss = True, use_propensity = True, test_size = 0)
        uplift_model.best_params_net
        y_test, x_test, t_test, rule_assignment_test = get_observational_uplift_data(num_obs)


        experiment_groups = np.zeros(num_obs)+2
        experiment_groups[np.where(x_test[:,-2]<.5)[0]] = 1
        experiment_groups[np.where(x_test[:,-2]<.33)[0]] = 0
        experiment_groups[np.where(x_test[:,-1]>.8)[0]] = 3


        optim_treatments_no_cuttoff = uplift_model.predict_optimal_treatments(x = x_test, use_propensity_score_cutoff = False)
        optim_treatments_cuttoff = uplift_model.predict_optimal_treatments(x = x_test, use_propensity_score_cutoff = True,
            propensity_score_cutoff = propensity_score_cutoff)

        optim_treatments_cuttoff_cat = optim_treatments_cuttoff.argmax(axis = 1)
        optim_treatments_no_cuttoff_cat = optim_treatments_no_cuttoff.argmax(axis = 1)

        correct_tmts_1 = np.array([x in [0,1] for x in optim_treatments_cuttoff_cat[np.where(experiment_groups == 0)[0]] ]).mean()
        correct_tmts_2 = np.array([x in [1,2] for x in optim_treatments_cuttoff_cat[np.where(experiment_groups == 1)[0]] ]).mean()
        correct_tmts_3 = np.array([x in [0,2] for x in optim_treatments_cuttoff_cat[np.where(experiment_groups == 2)[0]] ]).mean()
        correct_tmts_4 = np.array([x in [0] for x in optim_treatments_cuttoff_cat[np.where(experiment_groups == 3)[0]] ]).mean()

        correct_tmts_experiment_groups_1 = ((optim_treatments_cuttoff_cat[np.where(experiment_groups == 1)[0]] == 1)  == x_test[np.where(experiment_groups == 1)[0],0]).mean()
        correct_tmts_no_cutoff =  np.mean((optim_treatments_no_cuttoff_cat==1    ) == x_test[:,0])

        assert correct_tmts_1>TOLERANCE
        assert correct_tmts_2>TOLERANCE
        assert correct_tmts_3>TOLERANCE
        assert correct_tmts_4>TOLERANCE

        assert correct_tmts_experiment_groups_1>TOLERANCE

        assert np.array_equal(optim_treatments_cuttoff_cat,optim_treatments_no_cuttoff_cat) is False
        assert correct_tmts_no_cutoff>TOLERANCE
