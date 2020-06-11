# Learning kernel tests without data splitting

Code for the experiments of the paper "Learning kernel tests without data splitting"

To reproduce our results of Figure 1, please run the script experiment.sh in the directory results.
To make the plot please navigate to 'experiments/results' and execute the file 'evaluation.py'.

The default setting is to produce the plot for d=6 and the dataset 'diff_var'. 
Please change the parameters in the file 'config.yml' in order to create the other subplots.
To asses type-I errors, change the parameter 'hypothesis' to 'null'.



# Dependencies
Specified in the file requirements.txt