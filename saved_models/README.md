Saved Models:

model1: 

	cox_model.py was used to get this model
	Use BM_BLAST, HB and PLT, 200 epochs

  	Sequential:
   		torch.nn.BatchNorm1d(3),
		torch.nn.Linear(3, 32),
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
	Use BM_BLAST, HB and PLT, 200 epochs

 	Sequential:
   		torch.nn.BatchNorm1d(3),
		torch.nn.Linear(3, 32),
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

model20: 

	4features.py was used to get this model
	Use BM_BLAST, HB, PLT and number of Somatic mutations (NSM), 100 epochs

 	Sequential:
   		torch.nn.BatchNorm1d(4),
		torch.nn.Linear(4, 32),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(32, 64),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(64, 1)

	Test Data Indices calculated while training:
		Concordance Index: 0.713
		IPCW Concordance Index: 0.750

  	Obtained score from submission: -

model21: 

	4features.py was used to get this model
	Use BM_BLAST, HB, PLT and number of Somatic mutations (NSM), 100 epochs

 	Sequential:
   		torch.nn.BatchNorm1d(4),
		torch.nn.Linear(4, 32),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(32, 64),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(64, 1)

	Test Data Indices calculated while training:
		Concordance Index: 0.741
		IPCW Concordance Index: 0.766

  	Obtained score from submission: -
