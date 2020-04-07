from tempfile import NamedTemporaryFile

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from numpy.random import seed
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from keras.losses import mean_squared_error




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
