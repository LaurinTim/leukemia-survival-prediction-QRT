# leukemia-survival-prediction
Attempt to train model which predicts leukemia survival rate as outlined in https://challengedata.ens.fr/participants/challenges/162/.

Different models are saved in the saved_models folder. The files for the submission are in the submission_files folder.

The current best model is trained in model_2nn2.py and uses a NN. This is most likely not ideal as outlined in 10.3324/haematol.2021.280027 since there is too little training data available for a deep learning approach. A random forest or XGboost model would likely be better.
