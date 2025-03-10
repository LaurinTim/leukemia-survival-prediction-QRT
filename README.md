# leukemia-survival-prediction
Attempt to train model which predicts leukemia survival rate as outlined in https://challengedata.ens.fr/participants/challenges/162/.

Training data is in the X_train directory and the target in the target_train.csv file. The data for the submissions is in the X_train directory. The files for the submission are in the submission_files folder. Different models are saved in the saved_models folder.

The current best model is trained in model_rsf.py and uses the RandomSurvivalForest method from sklearn.

In the paper at 10.3324/haematol.2021.280027 a similar problem was investigated and it is used as a reference.

Other methods to obtain a model are in the "Pytorch DL models" and "Sklearn cox models" directories.

In "Pytorch DL models", learning was used. As outlined in the paper, this is most likely not ideas as there is not enough training data available.

In "Sklearn cox models" the CoxPHSurvivalAnalysis method from sklearn is used. This method uses the assumption that the hazard function has a log-linear relationship to the features.