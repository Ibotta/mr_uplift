from tempfile import NamedTemporaryFile

import pandas as pd
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, concatenate,Add, Multiply, RepeatVector, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from numpy.random import seed
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.activations import softmax
from sklearn.model_selection import KFold, ParameterGrid


def reduce_concat(x):
    """concatenates object into one string
    Args:
      x (array): values to concatenate into one string
    Returns:
      string of objects
    """
    return np.array(['_'.join(map(str, q)) for q in x])

def mse_loss_multi_output_np(yTrue, yPred):
    """Returns Loss for Keras. This is essentially a multi-output regression that
    ignores missing data. It's similar in concept to matrix factorization with
    missing data in that it only includes non missing values in loss function.
    Anything with -999.0 is ignored
    Args:
        yTrue (keras array): True Values
        yPred (keras array): Predictions
    Returns:
        Function to minimize
    """
    output = (yTrue  - yPred )**2
    return np.mean(output)


mse_loss_multi_output_scorer = make_scorer(
    mse_loss_multi_output_np, greater_is_better=False)


def hete_mse_loss_multi_output(
        input_shape,
        output_shape,
        num_nodes=256,
        dropout=.5,
        num_layers=1,
        activation='relu'):
    """Returns standard neural network
    Args:
        input_shape (int): number of features in training dataset
        num_nodes (int): Number of nodes in layers
        dropout (real): dropout percentage
        num_layers (real): number of num_layer
        activation (keras function): nonlinear activation function
    Returns:
        Keras Model
    """
    input = Input(shape=(input_shape,))

    x = Dense(num_nodes, activation=activation)(input)
    x = Dropout(dropout)(x)

    if(num_layers > 1):
        for q in range(num_layers - 1):
            x = Dense(num_nodes, activation=activation)(x)
            x = Dropout(dropout)(x)

    x = Dense(output_shape)(x)

    model = Model(input, x)
    model.compile(
        optimizer='rmsprop',
        loss=mean_squared_error,
        metrics=[mean_squared_error])

    return model


def train_model_multi_output_w_tmt(x, y, param_grid=None, n_jobs=-1, cv=5):
    """Trains and does gridsearch on a keras nnet. Response can be multi output
    Args:
        x (np array): explanatory variables
        y (np array): response variables. could be more than 1
        n_jobs (int): number of cores to run on.
        cv (int): number of cross vaildation_folds
    Returns:
        Gridsearched multi output model
    """

    if param_grid is None:
        param_grid = dict(num_nodes=[8, 128], dropout=[.1, .25, .5], activation=[
                          'relu'], num_layers=[1, 2], epochs=[50], batch_size=[400, 4000, 8000])

    model = KerasRegressor(
        build_fn=hete_mse_loss_multi_output,
        input_shape=x.shape[1],
        output_shape=y.shape[1],
        num_nodes=32,
        num_layers=1,
        epochs=100,
        verbose=False,
        dropout=.2,
        batch_size=1000)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=n_jobs,
        scoring=mse_loss_multi_output_scorer,
        verbose=False,
        cv=cv)

    grid_result = grid.fit(x, y)

    return grid_result






def create_base_network(input_shape, num_responses,
  num_layers = 1, num_nodes = 8, dropout = .1, activation = 'relu'):
  '''Base network to be shared (eq. to feature extraction).
  '''
  input = Input(shape=(input_shape,))

  x = Dense(num_nodes, activation=activation)(input)
  x = Dropout(dropout)(x)

  if(num_layers > 1):
     for q in range(num_layers - 1):
         x = Dense(num_nodes, activation=activation)(x)
         x = Dropout(dropout)(x)

  x = Dense(num_responses)(x)

  model = Model(input, x, name = 'net_model')

  return model



def optim_loss(yTrue,yPred):
    """Returns Loss for Keras. This function essentially asks the question:
    'what is the probability that a particular treatment yields the maximum?'.
    It does this by multiplying a softmax probability to the response variable
    of interest. If a response is missing then it is not included in loss.
    Args:
        yTrue (keras array): True Values
        yPred (keras array): Predictions
    Returns:
        Function to minimize
    """

    output = yTrue*yPred
    output = K.sum(output,axis = -1)
    return -K.mean(output)


def create_mo_optim_model(input_shape, num_responses, unique_treatments, num_layers,
  num_nodes, dropout):

  inputs_x = Input(shape=(input_shape,))
  util_weights = Input(shape=(num_responses,))

  batch_size = K.shape(inputs_x)[0]

  unique_tmts = [K.variable(x.reshape(1,unique_treatments.shape[1])) for x in unique_treatments]
  unique_tmts_full = [K.tile(x, (batch_size,1)) for x in unique_tmts]

  x_ts = [concatenate([tmts_full, inputs_x]) for tmts_full in unique_tmts_full]


  model = create_base_network(input_shape+unique_treatments.shape[1] ,num_responses,
  num_layers, num_nodes, dropout )

  outputs = [model(x) for x in x_ts]

  outputs_weighted = [Multiply()([outs,util_weights]) for outs in outputs]

  #expand to 3 dimensions. Used to concatenate in next line.
  outputs_weighted = [RepeatVector(1)(data) for data in outputs_weighted]
  outputs_weighted = concatenate(outputs_weighted, axis = 1)

  util_by_tmts = Lambda(lambda x: K.sum(x, axis=-1))(outputs_weighted)
  util_by_tmts_prob = Lambda(lambda x: softmax(x))(util_by_tmts)

  model = Model([inputs_x, util_weights], util_by_tmts_prob)
  model.compile(optimizer='rmsprop',
                loss=optim_loss)

  return model





def gridsearch_mo_optim(x, y, t, param_grid = None, copy_several_times = None):

    if copy_several_times is not None:
        x = np.concatenate([x.copy() for q in range(copy_several_times)])
        y = np.concatenate([y.copy() for q in range(copy_several_times)])
        t = np.concatenate([t.copy() for q in range(copy_several_times)])

    t_categorical = reduce_concat(t)
    unique_treatments = np.unique(t, axis = 0)

    grid = ParameterGrid(param_grid)

    kf = KFold(n_splits=2, shuffle=True, random_state=22)
    kf.get_n_splits(y)

    results = []
    n_obs = y.shape[0]
    utility_weights = np.concatenate([np.random.uniform(-1,1,n_obs).reshape(-1,1) for q in range(y.shape[1])], axis = 1)
    utility_weights = utility_weights/np.abs(utility_weights).sum(axis=1).reshape(-1,1)
    new_y = (utility_weights*np.array(y)).sum(axis=1)

    new_response = new_y.reshape(-1,1)*np.array(pd.get_dummies(pd.DataFrame(t_categorical).iloc[:,0]))

    for params in grid:
        print(params)
        temp_results = []
        for train_index, test_index in kf.split(y):

            mod = create_mo_optim_model(input_shape = x.shape[1],
              num_responses = y.shape[1],
              unique_treatments = unique_treatments, num_layers = params['num_layers'],
              num_nodes = params['num_nodes'],
              dropout = params['dropout'])

            mod.fit([x[train_index], utility_weights[train_index]], new_response[train_index], epochs = params['epochs'],
            batch_size = params['batch_size'], verbose = False)

            evaluation = mod.evaluate([x[test_index], utility_weights[test_index]], new_response[test_index])

            temp_results.append(evaluation)

        results.append(np.mean(temp_results).mean())

    optim_grid = [x for x in grid][np.argmin(np.array(results))]

    mod  = create_mo_optim_model(input_shape = x.shape[1],
            num_responses = y.shape[1],
            unique_treatments = unique_treatments, num_layers = optim_grid['num_layers'],
            num_nodes = optim_grid['num_nodes'],
            dropout = optim_grid['dropout'])

    mod.fit([x, utility_weights] , new_response, epochs = optim_grid['epochs'],
        batch_size = optim_grid['batch_size'], verbose = False)


    return mod, optim_grid, results[np.argmin(np.array(results))]
