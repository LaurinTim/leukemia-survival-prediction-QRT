# leukemia-survival-prediction
Attempt to train model which predicts leukemia survival rate as outlined in https://challengedata.ens.fr/participants/challenges/162/.

Training data is in the X_train directory and the target in the target_train.csv file. The data for the submissions is in the X_train directory. The files for the submission are in the submission_files folder. Different models are saved in the saved_models folder.

The current best model is trained in model_rsf.py and uses the RandomSurvivalForest method from sklearn.

In the paper at 10.3324/haematol.2021.280027 a similar problem was investigated and it is used as a reference.

Other methods to obtain a model are in the "Pytorch DL models" and "Sklearn cox models" directories.

In "Pytorch DL models", learning was used. As outlined in the paper, this is most likely not ideas as there is not enough training data available.

In "Sklearn cox models" the CoxPHSurvivalAnalysis method from sklearn is used. This method uses the assumption that the hazard function has a log-linear relationship to the features.

The main improvement would be to implement a different criteria on what features to use. Currently for each feature, only this feature is used to train RandomSurvivalForest and the ffeatures with IPCW index above some value are chosen to be kept. The problem with this is, that if a feature only occurs in few patients it will result in a bad IPCW index when only using this feature, but it might actually be a very good indicator for the few people that have this feature. It is also possible that features are selected that do not occur in any patients in the test data. Better ways to select features are mentioned in the paper that is mentioned above and they are: linear correlation, chi square, recursive elimination, lasso regularization and finally random forest ranking which is the only one that is currently used.

Another possible improvement would be to use XGboost survival tree methods.
