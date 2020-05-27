.. figure:: https://github.com/Ibotta/mr_uplift/blob/master/doc/images/mr_uplift_logo.png
   :alt: mr_erupt

mr_uplift: Machine learning Uplift Package
========================================================


Introduction
-----------------
There are currently several packages for uplift models (see `EconML <https://github.com/microsoft/EconML>`__ ,  `GRF <https://github.com/grf-labs/grf>`__, `PTE <https://cran.r-project.org/web/packages/PTE/index.html>`__). They tend to focus on interesting ways of estimating the heterogeneous treatment effect. However models in their current state tend to focus on the single response, singe treatment scenario. In addition the metrics they use do not give estimates to the expectations of response variables if the models were used in practice (PTE is an exception).

This package attempts to build an automated solution for Uplift modeling that includes the following features:

#. It allows for Multiple Treatments. In addition one can incorporate meta features for each treatment. For example; a particular treatment might have several shared features with other bonuses. Instead of creating a dummy indicator for each bonus the user can create a vector of categorial or continuous variables to represent the treatment.
#. `ERUPT <https://medium.com/building-ibotta/erupt-expected-response-under-proposed-treatments-ff7dd45c84b4>`__ functionality that estimates model performance on OOS data. This metric calculates the expected response if the model were given to the average user similar to .
#. Support for multiple responses. This allows estimation of tradeoffs between maximizing / minimizing weighted sums of responses. An example can be found `here <https://medium.com/building-ibotta/estimating-and-visualizing-business-tradeoffs-in-uplift-models-80ff845a5698>`__

It does so by estimating a neural network of the form y ∼ f(t,x) where y, x, and t are the response, explanatory variables and treatment variables. If `optim_loss=True` then an experimental loss function is used to estimate the function (see `here <https://github.com/Ibotta/mr_uplift/blob/master/examples/mr_uplift_new_optimized_loss.ipynb>`__). It is assumed the treatment was randomly assigned. There is functionality to predict counterfactuals for all treatments and calculates ERUPT metrics on out of sample data. 


Quick Start Example
-------------------

In a python enviornment :

.. code-block:: python

    import numpy as np
    import pandas as pd

    from mr_uplift.dataset.data_simulation import get_simple_uplift_data
    from mr_uplift.mr_uplift import MRUplift

    #Generate Data
    y, x, t = get_simple_uplift_data(10000)
    y = pd.DataFrame(y)
    y.columns = ['revenue','cost', 'noise']
    y['profit'] = y['revenue'] - y['cost']

    #Build / Gridsearch model
    uplift_model = MRUplift()
    param_grid = dict(num_nodes=[8], dropout=[.1, .5], activation=[
                          'relu'], num_layers=[1, 2], epochs=[25], batch_size=[30])
    uplift_model.fit(x, y, t.reshape(-1,1), param_grid = param_grid, n_jobs = 1)

    #OOS ERUPT Curves
    erupt_curves, dists = uplift_model.get_erupt_curves()

    #predict optimal treatments with new observations
    _, x_new ,_  = get_simple_uplift_data(5)
    uplift_model.predict_optimal_treatments(x_new, weights = np.array([.6,-.4,0,0]).reshape(1,-1))


.. figure:: https://github.com/Ibotta/mr_uplift/blob/master/doc/images/erupt_curves.png
   :alt: erupt-curves

Relevant Papers and Blog Posts
------------------------------

For Discussion on the metric used to calculate how model performs see:

`ERUPT: Expected Response Under Proposed Treatments <https://medium.com/building-ibotta/erupt-expected-response-under-proposed-treatments-ff7dd45c84b4>`__

`Uplift Modeling with Multiple Treatments and General Response Types <https://arxiv.org/pdf/1705.08492.pdf>`__

`Heterogeneous Treatment Effects and Optimal Targeting Policy Evaluation <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111957>`__

`A comparison of methods for model selection when estimating individual treatment effects <https://arxiv.org/pdf/1804.05146.pdf>`__

`Inference for the Effectiveness of Personalized Medicine with Software <https://arxiv.org/pdf/1404.7844.pdf>`__

For tradeoff analysis see:

`Estimating and Visualizing Business Tradeoffs in Uplift Models <https://medium.com/building-ibotta/estimating-and-visualizing-business-tradeoffs-in-uplift-models-80ff845a5698>`__

`Experimental Evaluation of Individualized Treatment Rules <https://imai.fas.harvard.edu/research/files/indtreat.pdf>`__

For optimized loss see:

`Maximizing The ERUPT Metric for Uplift Models <https://medium.com/building-ibotta/maximizing-the-erupt-metric-for-uplift-models-f8d7e57bfdf2>`__

`Methods for Individual Treatment Assignment: An Application and Comparison for Playlist Generation <https://arxiv.org/pdf/2004.11532.pdf>`__

Acknowledgements
~~~~~~~~~~~~~~~~
Thanks to `Evan Harris <https://github.com/denver1117>`__, `Andrew Tilley <https://github.com/tilleyand>`__, `Matt Johnson <https://github.com/mattsgithub>`__, and `Nicole Woytarowicz <https://github.com/nicolele>`__  for internal review before open source. Thanks to `James Foley <https://github.com/chadfoley36>`__ for logo artwork.
