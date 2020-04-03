import numpy as np
import pytest

from ibotta_uplift.data_simulation import get_no_noise_data, get_simple_uplift_data
from ibotta_uplift.calibrate_uplift import UpliftCalibration
from ibotta_uplift.erupt import weighted_avg_and_std, erupt,\
    get_best_tmts, get_weights
from ibotta_uplift.ibotta_uplift import IbottaUplift, get_t_data


class TestCalibrate(object):

    @pytest.fixture(scope='class')
    def calib(self):
        num_obs = 100
        y_no_noise, x_no_noise, tmt_no_noise = get_no_noise_data(num_obs)
        fake_counters = np.array(
            [np.array([np.zeros(num_obs), np.zeros(num_obs)]).T, x_no_noise])

        calib = UpliftCalibration()
        calib.fit(fake_counters, y_no_noise, tmt_no_noise)

        return calib

    def test_calibration(self, calib):

        assert np.allclose(np.sum(calib.calibrated_intercepts), 0)
        assert np.allclose(calib.calibrated_coefs[:, 1], np.array([2.0, 1.0]))

    def test_calibrate_new_data(self, calib):
        num_obs = 100
        true_coefs = np.array([2.0, 1.0])
        y_no_noise, x_no_noise, tmt_no_noise = get_no_noise_data(num_obs)
        fake_counters = np.array(
            [np.array([np.zeros(num_obs), np.zeros(num_obs)]).T, x_no_noise])

        calib_x = calib.transform(fake_counters)
        assert np.allclose((calib_x[1] / x_no_noise).mean(axis=0), true_coefs)

    def test_uplift_scores(self, calib):
        true_coefs = np.array([[1., 0.], [1., 0.]])
        calib.uplift_scores()
        assert np.allclose(calib.reg_results, true_coefs)


class TestErupt(object):

    def test_weighting(self):
        tmt = np.random.binomial(1, .25, 1000000)
        weights = get_weights(tmt)

        tmt_1_weight = weights[np.where(tmt == 1)[0]][0]
        tmt_0_weight = weights[np.where(tmt == 0)[0]][0]

        assert np.abs(tmt_1_weight - 1 / .25) < .01
        assert np.abs(tmt_0_weight - 1 / .75) < .01

    def test_get_best_tmts(self):
        num_obs = 100
        y_no_noise, x_no_noise, tmt_no_noise = get_no_noise_data(num_obs)
        fake_counters = np.array(
            [np.array([np.zeros(num_obs), np.zeros(num_obs)]).T, x_no_noise])

        optim = get_best_tmts([1, -1], fake_counters, np.array([0, 1]))

        assert (optim == (x_no_noise[:, 0] - x_no_noise[:, 1] > 0) * 1).mean()

    def test_weighted_avg_and_std(self):
        y = np.concatenate([np.ones(50) * .75, np.ones(25) * 2])
        weights = np.concatenate([np.ones(50), np.ones(25) * .5])
        assert weighted_avg_and_std(y, weights)[0] == 1

    def test_erupt(self):
        num_obs = 100
        y_no_noise, x_no_noise, tmt_no_noise = get_no_noise_data(num_obs)
        fake_counters = np.array(
            [np.array([np.zeros(num_obs), np.zeros(num_obs)]).T, x_no_noise])

        optim_tmt = get_best_tmts([1, -1], fake_counters, np.array([0, 1]))

        expected_means = y_no_noise[np.where(
            (tmt_no_noise == (x_no_noise[:, 0] - x_no_noise[:, 1] > 0)))[0]].mean(axis=0)

        erupts = erupt(y_no_noise, tmt_no_noise, optim_tmt,
                       weights=None, names=None).iloc[:, 0]
        erupts = np.array(erupts)

        assert np.allclose(erupts, expected_means)


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

        assert np.sqrt(
            np.mean(
                (oos_ice.mean(
                    axis=1) -
                    true_ATE)**2)) < rmse_tolerance

    def test_model_pred_oos_shapes(self):
        num_obs = 1000

        y, x, t = get_simple_uplift_data(num_obs)

        t = np.concatenate(
            [t.reshape(-1, 1), np.random.binomial(1, .5, num_obs).reshape(-1, 1)], axis=1)

        uplift_model = IbottaUplift()
        uplift_model.fit(x, y, t, n_jobs=1)

        assert uplift_model.predict_ice().shape == (
            np.unique(t, axis=0).shape[0], num_obs * .7, y.shape[1])
        assert uplift_model.predict_ice(
            x=x).shape == (
            np.unique(
                t,
                axis=0).shape[0],
            num_obs,
            y.shape[1])
