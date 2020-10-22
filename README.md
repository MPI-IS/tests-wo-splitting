# Learning kernel tests without data splitting

Code for the experiments of the paper "Learning kernel tests without
data splitting".
[https://arxiv.org/abs/2006.02286](https://arxiv.org/abs/2006.02286)
which will be presented at NeurIPS2020.

The implementations of the methods as described in the paper are in
the directory 'methods'.

## Installation

You can install the package as usual with the help of `pip` by calling

    pip install .

There is also `install` target in the `Makefile`, so you can simply run

    make install

We strongly suggest you install the package in a separate virtual
environment. You can create one by executing

    python -m venv venv

from the root of the project and then activate it by running

    . venv/bin/activate

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
