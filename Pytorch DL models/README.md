# Pytoch DL models
In this directory there are different routines to get models which use deep learning. The most recent one is model_2nn2.py, older versions are in the "old models" directory.

Because of the limited amount of available training data, it is not ideal to use deep learning and newer models use RandomSurvivalForest from sklearn. The deep learning models were mainly created for educational purposes.

model_2nn2.py uses 2 NNs (one for clinical data, one for molecular data) and then combines them in one model to get the final output. For the genes affected by the somatic mutations a pretrained embedding model obtained from https://github.com/jingcheng-du/Gene2vec was used (paper explaining the process of obtaining the model: https://doi.org/10.1186/s12864-018-5370-x). The expected IPCW Concordance Index using the model obtained from model_2nn2.py is between 0.68 and 0.70.