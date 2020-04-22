import numpy as np
import pytest

from mr_uplift.dataset.data_simulation import get_no_noise_data, get_simple_uplift_data
from mr_uplift.calibrate_uplift import UpliftCalibration
from mr_uplift.erupt import weighted_avg_and_std, erupt,\
    get_best_tmts, get_weights


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
        assert np.allclose(calib.regression_results, true_coefs)
