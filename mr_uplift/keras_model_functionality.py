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


def missing_mse_loss_multi_output(yTrue,yPred):
    """Returns Loss for Keras. This is essentially a multi-output regression that
    ignores missing data. It's similar in concept to matrix factorization with
    missing data in that it only includes non missing values in loss function.
    Args:
        yTrue (keras array): True Values
        yPred (keras array): Predictions
    Returns:
        Function to minimize
    """
    equals = K.cast(K.not_equal(yTrue, K.cast_to_floatx(-999.0)), dtype='float32')
    output = (yTrue * equals  - yPred*equals )**2
    return K.mean(output)

def create_mo_optim_model(input_shape, num_responses, unique_treatments, num_layers,
  num_nodes, dropout, alpha):

  inputs_x = Input(shape=(input_shape,))
  util_weights = Input(shape=(num_responses,))

  batch_size = K.shape(inputs_x)[0]

  unique_tmts = [K.variable(x.reshape(1,unique_treatments.shape[1])) for x in unique_treatments]
  unique_tmts_full = [K.tile(x, (batch_size,1)) for x in unique_tmts]

  x_ts = [concatenate([tmts_full, inputs_x]) for tmts_full in unique_tmts_full]


  model = create_base_network(input_shape+unique_treatments.shape[1] ,num_responses,
  num_layers, num_nodes, dropout )

  outputs = [model(x) for x in x_ts]


  outputs_mse = [RepeatVector(1)(data) for data in outputs]
  outputs_mse = concatenate(outputs_mse, axis = 1, name = 'outputs_mse')
  #outputs_mse = Dense(num_responses, name = 'outputs_mse')(outputs_mse)

  outputs_weighted = [Multiply()([outs,util_weights]) for outs in outputs]

  #expand to 3 dimensions. Used to concatenate in next line.
  outputs_weighted = [RepeatVector(1)(data) for data in outputs_weighted]
  outputs_weighted = concatenate(outputs_weighted, axis = 1)

  util_by_tmts = Lambda(lambda x: K.sum(x, axis=-1))(outputs_weighted)
  util_by_tmts_prob = Lambda(lambda x: softmax(x), name = 'util_by_tmts_prob')(util_by_tmts)

  model = Model([inputs_x, util_weights], [util_by_tmts_prob,outputs_mse])
  model.compile(optimizer='rmsprop',
                loss={'util_by_tmts_prob':optim_loss, 'outputs_mse': missing_mse_loss_multi_output},
               loss_weights={'util_by_tmts_prob': alpha, 'outputs_mse': (1-alpha)})

  return model

def copy_data(x,y,t, n):
        x = np.concatenate([x.copy() for q in range(n)])
        y = np.concatenate([y.copy() for q in range(n)])
        t = np.concatenate([t.copy() for q in range(n)])
        return x, y, t


def prepare_data(x, y ,t, copy_several_times = None):
    t_categorical = reduce_concat(t)
    unique_treatments = np.unique(t, axis = 0)

    new_dict = dict(zip(np.unique(t_categorical, axis= 0), [x for x in range(len(np.unique(t_categorical, axis= 0)))]))
    big_y = np.zeros((y.shape[0], len(np.unique(t_categorical, axis= 0)), y.shape[1] )) - 999
    for index in range(big_y.shape[0]):
        big_y[index, new_dict[t_categorical[index]] , :] = np.array(y)[index,:]

    n_obs = y.shape[0]
    utility_weights = np.concatenate([np.random.uniform(-1,1,n_obs).reshape(-1,1) for q in range(y.shape[1])], axis = 1)
    utility_weights = utility_weights/np.abs(utility_weights).sum(axis=1).reshape(-1,1)
    new_y = (utility_weights*np.array(y)).sum(axis=1)

    new_response = new_y.reshape(-1,1)*np.array(pd.get_dummies(pd.DataFrame(t_categorical).iloc[:,0]))

    if copy_several_times is not None:
        x = np.concatenate([x.copy() for q in range(copy_several_times)])
        new_response = np.concatenate([new_response.copy() for q in range(copy_several_times)])
        big_y = np.concatenate([big_y.copy() for q in range(copy_several_times)])
        utility_weights = np.concatenate([utility_weights.copy() for q in range(copy_several_times)])

    return x, utility_weights, new_response, big_y


def gridsearch_mo_optim(x, y, t, param_grid = None, copy_several_times = None):

    unique_treatments = np.unique(t, axis = 0)

    grid = ParameterGrid(param_grid)

    kf = KFold(n_splits=5, shuffle=True, random_state=22)
    kf.get_n_splits(y)

    results = []
    for params in grid:
        print(params)
        temp_results = []

        for train_index, test_index in kf.split(y):

            x_train, utility_weights_train, new_response_train, big_y_train  = prepare_data(x[train_index],y[train_index], t[train_index], copy_several_times)
            x_test, utility_weights_test, new_response_test, big_y_test   = prepare_data(x[test_index], y[test_index], t[test_index], copy_several_times)

            mod = create_mo_optim_model(input_shape = x.shape[1],
              num_responses = y.shape[1],
              unique_treatments = unique_treatments, num_layers = params['num_layers'],
              num_nodes = params['num_nodes'],
              dropout = params['dropout'],
              alpha = params['alpha'])

            mod.fit([x_train, utility_weights_train], [new_response_train, big_y_train],
            epochs = params['epochs'],
            batch_size = params['batch_size'],
            verbose = False)

            preds = mod.predict([x_test, utility_weights_test])[0]

            optim_value_location = np.argmax(preds, axis = 1)
            tmt_location = np.argmax(np.abs(new_response_test), axis = 1)

            shuffled_optim_value_location = optim_value_location.copy()
            np.random.shuffle(shuffled_optim_value_location)

            proposed_model = np.where(optim_value_location.reshape(-1,1) == tmt_location )[0]
            random_proposed_model = np.where(shuffled_optim_value_location.reshape(-1,1) == tmt_location )[0]

            gains = new_response_test.sum(axis = 1)[proposed_model].mean() - new_response_test.sum(axis = 1)[random_proposed_model].mean()
            print(gains)
            temp_results.append(gains)
            #evaluation = mod.evaluate([x[test_index], utility_weights[test_index]], [new_response[test_index],big_y[test_index,:,:]])
            #temp_results.append(evaluation)

        results.append(np.mean(temp_results).mean())

    optim_grid = [x for x in grid][np.argmax(np.array(results))]

    mod  = create_mo_optim_model(input_shape = x.shape[1],
            num_responses = y.shape[1],
            unique_treatments = unique_treatments, num_layers = optim_grid['num_layers'],
            num_nodes = optim_grid['num_nodes'],
            dropout = optim_grid['dropout'],
            alpha = optim_grid['alpha'])


    x, utility_weights, new_response, big_y  = prepare_data(x,y, t, copy_several_times)

    mod.fit([x, utility_weights] , [new_response,big_y], epochs = optim_grid['epochs'],
        batch_size = optim_grid['batch_size'], verbose = False)



    return mod, optim_grid, results[np.argmin(np.array(results))]
