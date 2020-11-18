import numpy as np
import pytest

from mr_uplift.dataset.data_simulation import get_no_noise_data, get_simple_uplift_data
from mr_uplift.mr_uplift import MRUplift, get_t_data
from mr_uplift.keras_model_functionality import prepare_data_optimized_loss

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


    def test_prepare_data_optimized_loss_one_col_tmt(self):
        num_obs = 1000

        y, x, t = get_simple_uplift_data(num_obs)
        unique_treatments = np.unique(t, axis = 0)

        x, utility_weights, missing_utility, missing_y_mat = prepare_data_optimized_loss(x,y,t,unique_treatments)

        assert(utility_weights.shape == (num_obs, y.shape[1]))
        assert(missing_y_mat.shape == (num_obs, unique_treatments.shape[0], y.shape[1]))
        for q in range(unique_treatments.shape[0]):
            assert( ((missing_utility[:,q]==0) == (missing_y_mat[:,q,0] == -999)).mean() ==1 )

    def test_prepare_data_optimized_loss_two_col_tmt(self):
        num_obs = 1000

        y, x, t = get_simple_uplift_data(num_obs)
        unique_treatments = np.unique(t, axis = 0)

        t = np.concatenate([t.reshape(-1, 1),
        np.random.binomial(1, .5, num_obs).reshape(-1, 1)], axis=1)

        unique_treatments = np.unique(t, axis = 0)
        x, utility_weights, missing_utility, missing_y_mat = prepare_data_optimized_loss(x,y,t,unique_treatments)

        assert(utility_weights.shape == (num_obs, y.shape[1]))
        assert(missing_y_mat.shape == (num_obs, unique_treatments.shape[0], y.shape[1]))
        for q in range(unique_treatments.shape[0]):
            assert( ((missing_utility[:,q]==0) == (missing_y_mat[:,q,0] == -999)).mean() ==1 )

    def test_model_optim_mean_outputs(self):
        true_ATE = np.array([[0, 0], [1, .5]])
        rmse_tolerance = .05
        num_obs = 10000
        param_grid = dict(num_nodes=[8], dropout=[.1], activation=['relu'], num_layers=[2], epochs=[30], batch_size=[100],
                                  alpha = [.5], copy_several_times = [2])

        y_no_noise, x_no_noise, tmt_no_noise = get_no_noise_data(num_obs)

        uplift_model = MRUplift()
        uplift_model.fit(x_no_noise, y_no_noise, tmt_no_noise.reshape(-1, 1),
                         n_jobs=1, param_grid = param_grid, optimized_loss = True)
        oos_ice = uplift_model.predict_ice(response_transformer = True)

        assert np.sqrt(np.mean((oos_ice.mean(axis=1) -true_ATE)**2)) < rmse_tolerance

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

        assert oos_re['mean'].iloc[0] > 0
