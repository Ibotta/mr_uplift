import numpy as np
import pandas as pd
import argparse
import functools

from ds_util.io import save_object, load_object

from ibotta_uplift.keras_model_functionality import save_keras_model, load_keras_model,\
    train_model_multi_output_w_tmt
from ibotta_uplift.erupt import get_erupts_curves_aupc, get_best_tmts

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_t_data(values, num_obs):
    """expands single treatment to many rows.
    Args:
      values (array): treatments to predict
    Returns:
      repeated treatment values
    """

    if isinstance(values, np.ndarray):
        len_values = len(values)
    else:
        len_values = 1

    values = np.array(values)
    values_rep = np.full((num_obs, len_values), values.reshape(1, len_values))

    return values_rep


def reduce_concat(x, sep=""):
    """concatenates object into one string
    Args:
      x (array): values to concatenate into one string
    Returns:
      string of objects
    """
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)


class IbottaUplift:
    def __init__(self, **kw):
        """I'm putting this in here because I think its necessary for .copy()
        function later. Not sure if thats the case.
        """
        self.kw = kw
        self.__dict__.update(kw)

    def fit(self, x, y, t, test_size=0.7, random_state=22, param_grid=None,
            n_jobs = -1, cv = 5):
        """Fits a Neural Network Model of the form y ~ f(t,x). Scales all variables
        and creates seperate transformers for y, t, and x. Assigns train / test split.

        Args:
        x (np array or pd.dataframe): Explanatory Data of shape num_observations
          by num_explanatory_variables
        y (np array or pd.dataframe): Response Variables of shape
          num_observations by num_response_variables
        t (np array or pd.dataframe): Treatment Variables of shape
          num_observations by num_treatment columns
        test_size (float): Percentage of observations to be in test set
        random_state (int): Random state for train test split. This is used in other parts
        of class.
        param_grid (dict): Parameters of keras model to gridsearch over.
        n_jobs (int): number of cores to run on.
        cv (int): number of cross vaildation_folds

        Returns:
        Builds a regression of form net of for y ~ f(t,x) and assigns it to
        self.model
        """

        self.unique_t = np.unique(np.array(t), axis=0)

        self.num_t = len(np.unique(t))
        self.num_responses = y.shape[1]

        self.x = np.array(x)
        self.y = np.array(y)
        self.t = np.array(t)

        self.test_size = test_size
        self.random_state = random_state

        if isinstance(x, pd.DataFrame):
            self.x_names = x.columns.values
        else:
            self.x_names = None

        if isinstance(y, pd.DataFrame):
            self.y_names = y.columns.values
        else:
            self.y_names = None

        if isinstance(t, pd.DataFrame):
            self.t_names = t.columns.values
        else:
            self.t_names = None

        x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(
            self.x, self.y, self.t, test_size=self.test_size,
            random_state=self.random_state)

        self.y_ss = StandardScaler()
        self.x_ss = StandardScaler()
        self.t_ss = StandardScaler()

        self.y_ss.fit(y_train)
        self.x_ss.fit(x_train)
        self.t_ss.fit(t_train)

        y_train_scaled = self.y_ss.transform(y_train)
        x_train_scaled = self.x_ss.transform(x_train)
        t_train_scaled = self.t_ss.transform(t_train)

        x_t_train = np.concatenate([t_train_scaled, x_train_scaled], axis=1)

        net = train_model_multi_output_w_tmt(x_t_train, y_train_scaled,
                                             param_grid=param_grid,
                                             n_jobs = n_jobs,
                                             cv = cv)

        self.best_score_net = net.best_score_
        self.best_params_net = net.best_params_
        self.model = net.best_estimator_.model

    def predict(self, x, t):
        """Returns predictions of Fitted Model

        Args:
        x (np array or pd.dataframe): Explanatory Data of shape num_observations
          by num_explanatory_variables
        t (np array or pd.dataframe): Treatment Variables of shape
          num_observations by num_treatment columns

        Returns:
        Builds and returns regression of form net of for y ~ f(t,x)
        """

        x_t_new = np.concatenate(
            [self.t_ss.transform(t), self.x_ss.transform(x)], axis=1)
        preds = self.model.predict(x_t_new)
        scaled_preds = self.y_ss.inverse_transform(preds)

        return scaled_preds

    def predict_ice(self, x=None, treatments=None):
        """Predicts all counterfactuals with new data. If no new data is
            assigned it will use test set data.
            The 'ice' refers to Individual Conditional Expectations. A better
            description of this can be found here:
            https://arxiv.org/pdf/1309.6392.pdf
        Args:
          x (np.array): new data to predict. Will use test data if not given
          treatments (np.array): Treatments to predict on. If none assigned then
          original training treatments are used.
        Returns:
          Counterfactuals for all treatments and response variables. an arrary of
          num_treatments by num_observations by num_responses
        """

        if x is None:
            x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(
                self.x, self.y, self.t, test_size=self.test_size,
                random_state=self.random_state)
        else:
            x_test = x

        if treatments is None:
            treatments = self.unique_t

        ice = np.array([self.predict(x_test, get_t_data(
            t, x_test.shape[0])) for t in treatments])

        return ice

    def get_erupt_curves(self, x=None, y=None, t=None, objective_weights=None,
                         treatments=None):
        """Returns ERUPT Curves and distributions. If either x or y is not inputted
        it will use test data.

        An example is described in more detail here:
        https://medium.com/building-ibotta/estimating-and-visualizing-business-tradeoffs-in-uplift-models-80ff845a5698

        Args:
          x (np.array): x data
          y (np.array): responses
          objective_weights (np.array): of dim (num_weights by num_response_variables).
          if none is assigned it will trade off costs of first two response variables and
          assume that first column is to be maximized and second column is to be minimized
          treatments (np.array): treatments to use in erupt calculations
        Returns:
          ERUPT Curves and Treatment Distributions
        """

        if self.num_responses == 1:
            raise Exception(
                "No Tradeoffs are available with one response variable.")

        if objective_weights is None:
            objective_weights = np.zeros((11, self.num_responses))
            objective_weights[:, 0] = np.linspace(0.0, 1.0, num=11)
            objective_weights[:, 1] = -np.linspace(1.0, 0.0, num=11)

        if x or y or t is None:
            x_train, x, y_train, y, t_train, t = train_test_split(
                self.x, self.y, self.t, test_size=self.test_size,
                random_state=self.random_state)

        if treatments is None:
            treatments = self.unique_t

        if t.shape[1] > 1:
            t = np.array([reduce_concat(x, '_') for x in t])
            unique_t = np.array([reduce_concat(x, '_') for x in treatments])
        else:
            unique_t = treatments

        to_keep_locs = np.where([z in unique_t for z in t])[0]
        y = y[to_keep_locs]
        t = t[to_keep_locs]
        x = x[to_keep_locs]

        ice_preds = self.predict_ice(x, treatments)

        return get_erupts_curves_aupc(
            y,
            t,
            ice_preds,
            unique_t,
            objective_weights,
            names=self.y_names)

    def copy(self):
        """Copies IbottaUplift Class. Not sure if this is the best way but it
        works.
        """
        return IbottaUplift(**self.kw)

    def save(self, file_location):
        """Saves IbottaUplift Class to location. Will save two outputs:
        keras model and rest of IbottaUplift class.

        Args:
          file_location (str): File location to save data
        Returns:
          Nothin. Saves file to location
        """
        uplift_class_copy = self.copy()
        uplift_class_copy.model = None

        save_object(
            uplift_class_copy,
            file_location +
            '/ibotta_uplift_class.pkl')
        save_keras_model(
            uplift_model.model,
            file_location +
            '/ibotta_uplift_model.pkl')

    def load(self, file_location):
        """Loads IbottaUplift Class from location.

        Args:
          file_location (str): File location to load data
        Returns:
          Updated Uplift Class
        """
        uplift_model = load_keras_model(
            file_location + '/ibotta_uplift_model.pkl')
        uplift_class = load_object(file_location + '/ibotta_uplift_class.pkl')
        uplift_class.model = uplift_model

        return uplift_class


    def predict_optimal_treatments(self, x, weights, treatments = None):
        """Predicts optimal treatments given explanatory variables and weights

        Args:
          x (np.array): new data to predict. Will use test data if not given
          weight (np.array): set of weights of length num_responses to maximize
          treatments (np.array): Treatments to predict on. If none assigned then
          original training treatments are used.
        Returns:
          Optimal Treatment Values
        """

        if treatments is None:
            treatments = self.unique_t

        ice = self.predict_ice(x, treatments)

        if treatments.shape[1] > 1:
            unique_t = np.array([reduce_concat(x, '_') for x in treatments])
        else:
            unique_t = treatments
        return get_best_tmts(weights, ice, treatments)
