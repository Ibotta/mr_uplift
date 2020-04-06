.. figure:: https://github.com/Ibotta/pure-predict/blob/master/doc/images/pure-predict.png
   :alt: pure-predict

ibotta_uplift: Machine learning uplift model package
========================================================

|License| |Build Status| |PyPI Package| |Python Versions|

``ibotta_uplift``
give example

Introduction
-----------------
There are currently several packages for uplift models. However they tend to rely on the single response, singe treatment scenario. In addition they do not report on an out-of-sample model metrics.


This package includes the following features to :

#. It allows for Multiple Treatments. In addition one can incorporate meta features for each treatment.
#. ERUPT functionality that estimates model performance on OOS data
#. Support for multiple responses. Will estimate tradeoffs between maximizing / minimizing weighted sums of responses.



Quick Start Example
-------------------

In a python enviornment :

.. code-block:: python
    import numpy as np
    import pandas as pd

    from dataset.data_simulation import get_simple_uplift_data
    from ibotta_uplift.ibotta_uplift import IbottaUplift

    #Generate Data
    y, x, t = get_simple_uplift_data(10000)
    y = pd.DataFrame(y)
    y.columns = ['revenue','cost', 'noise']
    y['profit'] = y['revenue'] - y['cost']

    #Build / Gridsearch model
    uplift_model = IbottaUplift()
    param_grid = dict(num_nodes=[8], dropout=[.1, .5], activation=[
                          'relu'], num_layers=[1, 2], epochs=[25], batch_size=[30])

    #OOS ERUPT Curves
    erupt_curves, dists = uplift_model.get_erupt_curves()

    #predict optimal treatments with new observations
    _, x_new ,_  = get_simple_uplift_data(5)
    uplift_model.predict_optimal_treatments(x_new, weights = np.array([.6,-.4,0,0]).reshape(1,-1))
