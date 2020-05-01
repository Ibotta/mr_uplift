import numpy as np
import pandas as pd
import dill
import copy
from keras.models import load_model
from mr_uplift.keras_model_functionality import train_model_multi_output_w_tmt
from mr_uplift.erupt import get_erupts_curves_aupc, get_best_tmts

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mr_uplift.calibrate_uplift import UpliftCalibration


def get_t_data(values, num_obs):
    """Repeats treatment to several rows and reshapes it. Used to append treatments
    to explanatory variable dataframe to predict counterfactuals.
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


def reduce_concat(x):
    """concatenates object into one string
    Args:
      x (array): values to concatenate into one string
    Returns:
      string of objects
    """
    return np.array(['_'.join(map(str, q)) for q in x])


class MRUplift(object):
    def __init__(self, **kw):

        """I'm putting this in here because I think its necessary for .copy()
        function later. Not sure if thats the case.
        """
        self.kw = kw
        self.__dict__.update(kw)

    def fit(self, x, y, t, test_size=0.7, random_state=22, param_grid=None,
            n_jobs=-1, cv=5):
        """Fits a Neural Network Model of the form y ~ f(t,x). Creates seperate
        transformers for y, t, and x and scales each. Assigns train / test split.

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
        Builds a neural network and assigns it to self.model
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
                                             n_jobs=n_jobs,
                                             cv=cv)

        self.best_score_net = net.best_score_
        self.best_params_net = net.best_params_
        self.model = net.best_estimator_.model

    def predict(self, x, t):
        """Returns predictions of the fitted model. Transforms both x and t then
        concatenates those two variables into an array to predict on. Finally,
        an inverse transformer is applied on predictions to transform to original
        response means and standard deviations.

        Args:
        x (np array or pd.dataframe): Explanatory Data of shape num_observations
          by num_explanatory_variables
        t (np array or pd.dataframe): Treatment Variables of shape
          num_observations by num_treatment columns

        Returns:
        Predictions fitted model
        """

        x_t_new = np.concatenate(
            [self.t_ss.transform(t), self.x_ss.transform(x)], axis=1)
        preds = self.model.predict(x_t_new)

        scaled_preds = self.y_ss.inverse_transform(preds)

        return scaled_preds

    def predict_ice(self, x=None, treatments=None, calibrator=False):
        """Predicts all counterfactuals with new data. If no new data is
            assigned it will use test set data. Can subset to particular treatments
            using treatment assignment. Can also apply calibrator function (experimental)
            to predictions.

            The 'ice' refers to Individual Conditional Expectations. A better
            description of this can be found here:
            https://arxiv.org/pdf/1309.6392.pdf
        Args:
          x (np.array): new data to predict. Will use test data if not given
          treatments (np.array): Treatments to predict on. If none assigned then
          original training treatments are used.
        calibrator (boolean): If true will use the trained calibrator to transform
          responses. Otherwise will use the response inverse transformer
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

        if calibrator:
            ice = self.calibrator.transform(ice)

        return ice

    def get_erupt_curves(self, x=None, y=None, t=None, objective_weights=None,
                         treatments=None, calibrator=False):
        """Returns ERUPT Curves and distributions of treatments. If either x or
        y or t is not inputted it will use test data.

        If there is only one response variable then it will assume we want to maximize the response.
        It will calculate and return the ERUPT metric and distribution of treatments.
        An introduction to ERUPT metric can be found here https://medium.com/building-ibotta/erupt-expected-response-under-proposed-treatments-ff7dd45c84b4

        If there are mutliple responses it will use objective_weights to create a weighted sum of response
        variables. It will then use this new weighted sum to determine optimal treatments and calculate ERUPT
        metrics accordingly. Repeatedly doing this with different weights will lead estimations of tradeoffs.

        ERUPT curves are described in more detail here:
        https://medium.com/building-ibotta/estimating-and-visualizing-business-tradeoffs-in-uplift-models-80ff845a5698

        Args:
          x (np.array): new data to predict. Will use test data if not given
          y (np.array): responses
          objective_weights (np.array): of dim (num_weights by num_response_variables).
          if none is assigned it will trade off costs of first two response variables and
          assume that first column is to be maximized and second column is to be minimized
          treatments (np.array): treatments to use in erupt calculations
          calibrator (boolean): If true will use the trained calibrator to transform
            responses. Otherwise will use the response inverse transformer

        Returns:
          ERUPT Curves and Treatment Distributions
        """

        if self.num_responses == 1:
            objective_weights = np.array([1]).reshape(1,-1)

        if objective_weights is None:
            objective_weights = np.zeros((11, self.num_responses))
            objective_weights[:, 0] = np.linspace(0.0, 1.0, num=11)
            objective_weights[:, 1] = -np.linspace(1.0, 0.0, num=11)

        if any([q is None for q in [x, y, t] ]):
            print("Using Test Data Set")
            x_train, x, y_train, y, t_train, t = train_test_split(
                self.x, self.y, self.t, test_size=self.test_size,
                random_state=self.random_state)

        if treatments is None:
            treatments = self.unique_t

        if t.shape[1] > 1:
            t = reduce_concat(t)
            unique_t = reduce_concat(treatments)
        else:
            unique_t = treatments

        to_keep_locs = np.where([z in unique_t for z in t])[0]
        y = y[to_keep_locs]
        t = t[to_keep_locs]
        x = x[to_keep_locs]

        ice_preds = self.predict_ice(x, treatments, calibrator)

        return get_erupts_curves_aupc(
            y,
            t,
            ice_preds,
            unique_t,
            objective_weights,
            names=self.y_names)

    def copy(self):
        """Copies MRUplift Class. Not sure if this is the best way but it
        works.
        """
        return copy.copy(self)

    def save(self, file_location):
        """Saves MRUplift Class to location. Will save two outputs:
        keras model and rest of MRUplift class.

        Args:
          file_location (str): File location to save data
        Returns:
          Nothin. Saves file to location
        """
        model = self.model
        uplift_class_copy = self.copy()
        uplift_class_copy.model = None


        dill.dump(uplift_class_copy, file = open(file_location + '/mr_uplift_class.pkl', "wb"))

        model.save(file_location + '/mr_uplift_model.h5')



    def load(self, file_location):
        """Loads MRUplift Class from location.

        Args:
          file_location (str): File location to load data
        Returns:
          Updated Uplift Class
        """

        uplift_class = dill.load(open(file_location + '/mr_uplift_class.pkl', "rb"))
        uplift_class.model = load_model(file_location + '/mr_uplift_model.h5')
        self.__dict__.update(uplift_class.__dict__)


    def predict_optimal_treatments(self, x, weights=None, treatments=None,
                                   calibrator=False):
        """Calculates optimal treatments of model output given explanatory
            variables and weights

        Args:
          x (np.array): new data to predict. Will use test data if not given
          weight (np.array): set of weights of length num_responses to maximize.
            is required for multi output decisions
          treatments (np.array): Treatments to predict on. If none assigned then
          original training treatments are used.
          calibrator (boolean): If true will use the trained calibrator to transform
          responses. Otherwise will use the response inverse transformer

        Returns:
          Optimal Treatment Values
        """

        if treatments is None:
            treatments = self.unique_t

        ice = self.predict_ice(x, treatments, calibrator)

        if self.num_responses > 1:

            if treatments.shape[1] > 1:
                unique_t = reduce_concat(treatments)
            else:
                unique_t = treatments
            best_treatments = get_best_tmts(weights, ice, treatments)
        else:
            best_treatments = np.argmax(ice, axis=0)

        return best_treatments

    def calibrate(self, treatments=None):
        """(Experimental)
        This fits a calibrator on training dataset. This of the form
        y = b0y_pred_0*t_0+b1*y_pred_1*t_1 + ... + b_num_tmts*y_pred_numtmts*t_num_tmts for all treatments.

        Args:
            None
        Returns:
          None
        """
        x_train, x, y_train, y, t_train, t = train_test_split(
            self.x, self.y, self.t, test_size=self.test_size,
            random_state=self.random_state)

        if t_train.shape[1] > 1:
            t_train = reduce_concat(t_train)
        t_train = t_train.reshape(-1, 1)

        ice = self.predict_ice(x_train, treatments)

        calib = UpliftCalibration()
        calib.fit(ice, y_train, t_train)
        calib.uplift_scores()

        self.calibrator = calib


    def permutation_varimp(self, weights=None, x=None, treatments=None, calibrator=False):
        """Variable importance metrics. This is based on permutation tests. For variable this permutes the column
        and then predicts and finds the optimal value given a set of weights. For each user it compares the optimal treatment of
        permuted column data with optimal treatment of non-permuted data and averages the result. The output is an index of how often
        the permuted column disagrees with unpermuted columns.

        Args:
          weights (np.array): set of weights of length num_responses to maximize.
            is required for multi output decisions
          x (np.array): new data to predict. Will use test data if not given
            treatments (np.array): Treatments to predict on. If none assigned then
            original training treatments are used.
          calibrator (boolean): If true will use the trained calibrator to transform
            responses. Otherwise will use the response inverse transformer
        Returns:
          df of variable importance metrics
        """
        if x is None:
            x_train, x, y_train, y, t_train, t = train_test_split(
                self.x, self.y, self.t, test_size=self.test_size,
                random_state=self.random_state)


        original_decisions = self.predict_optimal_treatments(x, weights=weights, treatments=treatments,
                                       calibrator=calibrator)
        varimps = []
        for p in range(x.shape[1]):

            shuffled_index = np.arange(x.shape[0])
            np.random.shuffle(shuffled_index)

            x_copy = x.copy()
            x_copy[:,p] = x_copy[:,p][shuffled_index]

            temp_decisions = self.predict_optimal_treatments(x_copy,
                weights=weights, treatments=treatments,
                calibrator=calibrator)
            temp_varimp = (original_decisions == temp_decisions).mean()

            varimps.append(temp_varimp)

        #make varimps a 'larger number -> more important' metric
        varimps = 1 - np.array(varimps)

        varimps_pd = pd.DataFrame(np.array(varimps))
        varimps_pd.columns = ['permuation_varimp_metric']

        if self.x_names is not None:
            varimps_pd['var_names'] = self.x_names

        return varimps_pd
