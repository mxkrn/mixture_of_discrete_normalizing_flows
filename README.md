# Mixture of Discrete Normalizing Flows for Variational Inference

## Sources and demos in Jupyter notebooks:

The repository includes:

1. [notebooks](notebooks) - Jupyter notebooks illustrating use of MDNF with various models:
 * [notebooks/GaussianMixture.ipynb](GaussianMixture) - GMM with MDNF
 * [notebooks/BayesianNetwork.ipynb](BayesianNetwork) - BN with MDNF
 * [notebooks/VAEFlows.ipynb](VAEFlows) - VAE with MDNF
 * [notebooks/PartialFlows.ipynb](PartialFlows) - partial vs. location-scale  flows

2. [mdnf](mdnf) - main files implementing flows, mixtures, inference etc.:
 * [one_hot.py]() - Operations on one-hot encoded vectors. 
 * [flows_mixture.py]() - Mixture of discrete normalizing flows. 
 * [inference.py]() - Variational inference algorithms for discrete normalizing flows. 
  
 * Base distributions:
   * [base_mixtures.py]() - Base for mixture of categorical distributions. 
   * [base_categorical.py]() - Factorized categorical distribution. 
   * [base_constructors.py]() - Creating base mixtures of categorical distributions. 
   
 * Individual discrete flows:
   * [flows_transformations.py]() - Networks calculating transformations for discrete flows. 
   * [flows.py]() - Basic flows.
   * [flows_factorized.py]() - Discrete flows for factorized distributions.  
   * [flows_edward2_made.py]() - Masked autoencoders.  
   * [flows_edward2.py]() - Discrete autoregressive flows.

 * Models:
   * [bayesian_networks.py]() - Evaluation of joint probability of x and y for arbitrary Bayesian networks. 
   * [gmvi.py]() - Variational Gaussian Mixture using Discrete Normalizing Flows. 
 
 * Auxiliary:
   * [prob_recovery.py]() - Recovering probability tables from samples or flows. 
   * [aux.py]() - General auxiliary functions. 
   * [time_profiling.py]() - Auxiliary functions for measuring time. 

 * Unit tests:
   * [one_hot_test.py]() 
   * [flows_test.py]() 
   * [flows_mixture_test.py]() 


## Specification of dependencies

The code was tested with Python 3.7.4 (on a Linux platform),
using *tensorflow 2.2.0* and *tensorflow_probability 0.9.0*
(can be installed with `pip install tensorflow tensorflow_probability`).
It also requires *numpy*, *pandas*, *sklearn* and *scipy*,
that can be installed with `pip install numpy pandas sklearn scipy`,
 but are also available by default in for example,
 [python Anaconda distributions](https://www.anaconda.com/products/individual).
Potential problems with *scipy 1.4.1* can be solved by downgrading it to version 1.2.1 with 
`pip install scipy==1.2.1`.

Notebooks *.ipynb* can be previewed using *Jupyter Notebook* and run from a command line with *runipy*. 
Visualizing results requires *matplotlib* and *seaborn* to be available (`pip install matplotlib seaborn`).

Parts of the code for Bayesian networks require PGMPY
(`pip install pgmpy==0.1.10`) and 
code for Gaussian mixture models builds on 
[Python codes implementing algorithms described in Bishop's book](https://github.com/ctgk/PRML) 
(can by installed with `git clone https://github.com/ctgk/PRML; cd PRML; python setup.py install`). 

Finally, the code comparing partial and location-scale flows uses 
[Edward2](https://github.com/google/edward2) that can be installed with 
`pip install "git+https://github.com/google/edward2.git#egg=edward2"`

