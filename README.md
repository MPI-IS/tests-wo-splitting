[![Build status](https://raw.githubusercontent.com/MPI-IS-BambooAgent/sw_badges/master/badges/plans/testswithoutsplitting/build.svg?sanitize=true)](https://atlas.is.localnet/bamboo/browse/BAMEI-TWS/latest/)

# Learning kernel tests without data splitting

Code for the experiments of the paper "Learning kernel tests without
data splitting".
[https://arxiv.org/abs/2006.02286](https://arxiv.org/abs/2006.02286)
which will be presented at NeurIPS2020.

The implementations of the methods as described in the paper are in
the directory 'methods'.

## Installation

We strongly suggest you to install the package in a separate virtual
environment. You can create one by executing

    python -m venv --copies my_venv

from the root of the project and then activate it by running

    . my_venv/bin/activate

You can then install the package as usual with the help of `pip` by calling

    pip install .

or using the `install` target in the `Makefile` by simply running

    make install


## Computing p-values

If you want perform a two sample test on your own samples X and Y you can use the function `pvalue()` in 
`tests-wo-split/methods/pvalue`.
A simple test of the validity of our method is to see whether the p-values are uniformly distributed 
under the null hypothesis (samples come from the same distribution).
#### Example: uniform distribution of p-values
    import matplotlib.pyplot as plt
    from tests_wo_split.methods.pvalue import pvalue
    import numpy as np
    runs = 1000
    size = 1000
    p = []
    for i in range(runs):
        x = np.random.normal(0,1, size=size)
        y = np.random.normal(0,1, size=size)
        p.append(pvalue(x=x, y=y))
    plt.hist(p)
    plt.show()



## Reproducing Figure 2

To reproduce our results of Figure 2 you can use the provided
`Makefile`. Simply execute

    make fig

from the root of the project. This will run all the experiments,
render the figure and leave it as `evaluation.pdf` in the root of the
project. You can run the experiments in parallel by calling this task as

    make -j4 fig

The number after `-j` determines the number of parallel processes. It
should be possible to run at least 4 processes on an average laptop.

The default setting is to reproduce the experiments for d=6
and the dataset 'diff_var'. To exactly reproduce the upper right plot
of Figure 2, please set `runs: 5000` in the config file `config.yml`
(this increases the execution time linearly!). In order to create the
other subplots, please un-comment the corresponding section of the
config file. To asses type-I errors, change the parameter 'hypothesis'
to 'null'.

## Your own dataset

To test the method on your own distributions P and Q, go to the file
'config.yml' and set 'dataset' to 'own_dataet'. Further please go to
'datasets/generate_data.py' and specify how to draw samples from your
custom distribution.

## Author

[Jonas Kübler](https://github.com/jmkuebler),
Empirical Inference Department - Max Planck Institute for Intelligent Systems

## License

MIT License (see LICENSE.md)

## Copyright

© 2020, Max Planck Society - Max Planck Institute for Intelligent Systems
