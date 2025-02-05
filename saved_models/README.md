Saved Models:

model1: 
	cox_model.py was used to get this model
	Use BM_BLAST, HB and PLT, 100 epochs

  	Sequential:
   		torch.nn.BatchNorm1d(num_features),  # Batch normalization
		torch.nn.Linear(num_features, 32),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(32, 64),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(64, 1)

	Test Data Indices calculated while training:
		Concordance Index: 0.709
		IPCW Concordance Index: 0.734

  	Obtained score from submission: 0.648

  model11: 

	cox_model.py was used to get this model
 
	Use BM_BLAST, HB and PLT, 100 epochs

 	Sequential:
   		torch.nn.BatchNorm1d(num_features),  # Batch normalization
		torch.nn.Linear(num_features, 32),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(32, 64),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(64, 1)

	Test Data Indices calculated while training:

		Concordance Index: 0.698

		IPCW Concordance Index: 0.726

  	Obtained score from submission: 0.653
