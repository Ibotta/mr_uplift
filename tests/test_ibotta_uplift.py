import numpy as np
import pytest

from ibotta_uplift.dataset.data_simulation import get_no_noise_data, get_simple_uplift_data
from ibotta_uplift.ibotta_uplift import IbottaUplift, get_t_data


class TestIbottaUplift(object):

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

        uplift_model = IbottaUplift()
        uplift_model.fit(x_no_noise, y_no_noise, tmt_no_noise.reshape(-1, 1),
                         n_jobs=1)
        oos_ice = uplift_model.predict_ice()

        assert np.sqrt(np.mean((oos_ice.mean(axis=1) -true_ATE)**2)) < rmse_tolerance

    def test_model_pred_oos_shapes(self):
        num_obs = 1000

        y, x, t = get_simple_uplift_data(num_obs)

        t = np.concatenate([t.reshape(-1, 1),
        np.random.binomial(1, .5, num_obs).reshape(-1, 1)], axis=1)

        uplift_model = IbottaUplift()
        uplift_model.fit(x, y, t, n_jobs=1)

        assert uplift_model.predict_ice().shape == (
            np.unique(t, axis=0).shape[0], num_obs * .7, y.shape[1])

        assert uplift_model.predict_ice(x=x).shape == (np.unique(t,axis=0).shape[0],
            num_obs,
            y.shape[1])
