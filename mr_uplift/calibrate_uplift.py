import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_reg_scores(ice, y, tmt):
    """Increase in Scores of using HETE vs control
    Args:
        ice (np array): Individual Conditional Expections 2-d array with
            num_observations, num_tmts.
        This is the counterfactuals for all treatment, response variable pairs.
        y (np array): response variables
        tmt (np array): treatment column
    Returns:
        Builds and returns regression of form y = b0 + b1*y_0_hat +
        b2*(y_1_hat - y_0_hat) + ... + bp*(y_p_hat - y_0_hat)
        and a regression of form y = b0 + b1*y_0_hat.
        It returns in sample scores of each model
    """

    hete = (ice - ice[0, :])

    # if tmt is a binary get_dummies will not expand. turn into string and it
    # will
    tmt = [str(x) for x in tmt]
    temp_data = pd.DataFrame(hete.T * pd.get_dummies(tmt))
    temp_data = temp_data.iloc[:, 1:]
    temp_data['base'] = ice[0, :]

    reg = LinearRegression()
    reg.fit(temp_data, y)

    reg_base = LinearRegression()
    reg_base.fit(temp_data['base'].values.reshape(-1, 1), y)

    score_hete = reg.score(temp_data, y)
    score_base = reg_base.score(
        temp_data['base'].values.reshape(-1, 1), y.reshape(-1, 1))

    return [score_hete, score_base]


def build_calibration_models(ice, y, tmt):
    """Returns Calibrated HETE Models.
    Args:
      ice (np array): Individual Conditional Expections 2-d array with
        num_observations, num_tmts.
      This is the counterfactuals for all treatment, response variable pairs.
      y (np array): response variables
      tmt (np array): treatment column
    Returns:
      Builds and returns regression of form y = b0 + b1*y_0_hat +
      b2*y_1_hat + ... + b(p+1)*(tmt == 0) + ...
    """

    temp_data = pd.DataFrame(ice.T * pd.get_dummies(tmt.reshape(-1,)))
    temp_dummies = pd.get_dummies(tmt.reshape(-1,))
    temp_data = pd.concat([temp_data, temp_dummies], axis=1)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(temp_data, y)

    return reg


class UpliftCalibration:
    """A calibration class for uplift models. Currently has a .fit and
    .transform method to get calibrated effects for each treatment.
    """

    def fit(self, ice, y, tmt):
        """Fits Calibrated HETE Models.
        Args:
          ice (np array): Individual Conditional Expections 2-d array with
            num_observations, num_tmts.
          This is the counterfactuals for all treatment, response variable pairs.
          y (np array): response variables
          tmt (np array): treatment column
        Returns:
          Builds and returns regression of form y = b0 + b1*y_0_hat +
          b2*(y_1_hat - y_0_hat) + ... + bp*(y_p_hat - y_0_hat)
        """

        self.ice = ice
        self.y = y
        self.tmt = tmt
        self.num_tmts = len(np.unique(tmt))
        self.num_responses = y.shape[1]

        self.regs = [build_calibration_models(
            self.ice[:, :, index],
            self.y[:, index],
            self.tmt)
            for index in range(self.num_responses)]

        calibrated_coefs = np.array(
            [self.regs[x].coef_ for x in range(self.num_responses)])
        self.calibrated_coefs = calibrated_coefs[:, :self.num_tmts]
        self.calibrated_intercepts = calibrated_coefs[:, self.num_tmts:]

    def transform(self, new_ice):
        """Transforms counterfactual predictions to be calibrated
        Args:
          new_ice (np array): Individual Conditional Expections 2-d array with
            num_observations, num_tmts.
        Returns:
          Calibrated counterfactuals. Thef first treatment will be considered
          the 'base' and is set to zero. All other counterfactuals are relative
          to this treatment.
        """

        counters_scaled_diff_mats_calibrated = [
            self.calibrated_intercepts.T + new_ice[:, x, :] * self.calibrated_coefs.T for x in range(new_ice.shape[1])]
        counters_scaled_diff_mats_calibrated = np.swapaxes(
            np.array(counters_scaled_diff_mats_calibrated), 1, 0)

        return counters_scaled_diff_mats_calibrated

    def uplift_scores(self):
        """Calculates increase in Scores of using HETE vs control
        Returns:
          Two scores for each response variable. One with all treatments and
          one with only first treatment. This should give an idea of how well
          uplift model does for each treatment
        """

        regression_results = [get_reg_scores(
            self.ice[:, :, index],
            self.y[:, index],
            self.tmt)
            for index in range(self.num_responses)]
        regression_results = pd.DataFrame(regression_results)
        regression_results.columns = [
            'with_uplift_effects',
            'without_uplift_effects']

        self.regression_results = regression_results
