# leukaemia-survival-prediction
Attempt to train model which predicts leukaemia survival rate as outlined in https://challengedata.ens.fr/participants/challenges/162/.

Training data is in the X_train directory and the target in the target_train.csv file. The data for the test set is in the X_test directory. The files for the prediction of the test set are in the submission_files folder. Different models are saved in the saved_models folder.

The current best model is trained in the script *model_rsf.py* and uses the `RandomSurvivalForest` method from sklearn. An IPCW Concordance Index of 0.744 was achieved using this script.

The paper [Prediction of complete remission and survival in acute myeloid leukemia using supervised machine learning](https://doi.org/10.3324/haematol.2021.280027) investigates a similar problem and is used as a reference.

Other methods to train a model are in the *Pytorch DL models* and *Sklearn cox models* directories.

In the *Pytorch DL models* directory, scripts for models trained with deep learning for survival analysis learning can be found. As outlined in the paper, this is most likely not ideal as there is not enough training data available.

In the *Sklearn cox models* directory, scripts using the `CoxPHSurvivalAnalysis` method from sklearn can be found. This method uses the assumption that the hazard function has a log-linear relationship to the features.

The main improvement to the current best model would be to implement a different criteria for the feature selection. For the feature selection of the final model, a separate model is trained for each feature using the `RandomSurvivalForest` method currently. The IPCW score of these models is then determined and features, for which the model scores above some fixed threshold, are kept. The problem with this approach is, that if a feature only occurs in few patients it will result in a bad IPCW index when only using this feature, but it might actually be a very good indicator for the few people that have this feature. Also features that are only useful in combination with other features are discarded. It is also possible that features are selected that do not occur in any patients in the test data. Other ways to select features are shown in the paper mentioned above. These methods are: linear correlation, chi square, recursive elimination, lasso regularization and random forest ranking, which is the only one that is currently used.

Another possible improvement would be to use XGboost survival tree methods.
