# Learning kernel tests without data splitting

Code for the experiments of the paper "Learning kernel tests without data splitting".
For the camera-ready version we will create a Github repository and make the code publicly available.

The implementations of the methods as described in the paper are in the directory 'methods'.

##### Reproduce Figure 1
To reproduce our results of Figure 1, please run the script experiment.sh in the directory results (please adapt the path to your virtualenv before).

To make the plot please navigate to 'experiments/results' and execute the file 'evaluation.py'.

The default setting is to reproduce the experiments for d=6 and the dataset 'diff_var'. 
Please change the parameters in the file 'config.yml' in order to create the other subplots.
To asses type-I errors, change the parameter 'hypothesis' to 'null'.



##### Dependencies
Specified in the file requirements.txt