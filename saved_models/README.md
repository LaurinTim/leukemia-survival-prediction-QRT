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
		IPCW Concordance Index: 0.734 (WRONG)

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
		IPCW Concordance Index: 0.726 (WRONG)

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
		IPCW Concordance Index: 0.750 (WRONG)

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
		IPCW Concordance Index: 0.766 (WRONG)

  	Obtained score from submission: -

model30: 

	4features.py was used to get this model
	Use BM_BLAST, HB, PLT and number of Somatic mutations (NSM), 30 epochs

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
		Concordance Index: ?
		IPCW Concordance Index: 0.695

  	Obtained score from submission: -

model31: 

	4features.py was used to get this model
	Use BM_BLAST, HB, PLT and number of Somatic mutations (NSM), 30 epochs

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
		Concordance Index: 0.731
		IPCW Concordance Index: 0.714

  	Obtained score from submission: 0.656 (why the large discrepancy?)

model31: 

	4features.py was used to get this model
	Use BM_BLAST, HB, PLT and number of Somatic mutations (NSM), 30 epochs

 	Sequential:
   		torch.nn.BatchNorm1d(4),
		torch.nn.Linear(4, 32),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(32, 64),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
  		torch.nn.Linear(64, 128),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(128, 1)

	Test Data Indices calculated while training:
		Concordance Index: 0.717
		IPCW Concordance Index: 0.703

  	Obtained score from submission: 0.654 (why the large discrepancy?)

model40: 

	4features.py was used to get this model
	Use BM_BLAST, HB, PLT and number of Somatic mutations (NSM), 50 epochs, learning rate was lowered to 0.001

 	Sequential:
   		torch.nn.BatchNorm1d(4),
		torch.nn.Linear(4, 32),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(32, 64),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
  		torch.nn.Linear(64, 128),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(128, 1)

	Test Data Indices calculated while training:
		Concordance Index: 0.699
		IPCW Concordance Index: 0.673

  	Obtained score from submission: -

model50: 

	4features.py was used to get this model
	Use BM_BLAST, HB, PLT and number of Somatic mutations (NSM), 1000 epochs, learning rate was lowered to 0.0001

 	Sequential:
   		torch.nn.BatchNorm1d(4),
		torch.nn.Linear(4, 32),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(32, 64),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
  		torch.nn.Linear(64, 128),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(128, 1)

	Test Data Indices calculated while training:
		Concordance Index: 0.695
		IPCW Concordance Index: 0.676

  	Obtained score from submission: -

model60: 

	4features.py was used to get this model
	Use BM_BLAST, HB, PLT and number of Somatic mutations (NSM), 200 epochs, learning rate 1e-4

 	Sequential:
   		torch.nn.BatchNorm1d(4),
		torch.nn.Linear(4, 32),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(32, 64),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
  		torch.nn.Linear(64, 128),
		torch.nn.ReLU(),
		torch.nn.Dropout(),
		torch.nn.Linear(128, 1)

	Test Data Indices calculated while training:
		Concordance Index: 0.708
		IPCW Concordance Index: 0.677

  	Obtained score from submission: -

model70: 

	4features.py was used to get this model
	Use BM_BLAST, HB, PLT and number of Somatic mutations (NSM), 2000 epochs, learning rate 5e-5

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
		Concordance Index: 0.694
		IPCW Concordance Index: 0.663

  	Obtained score from submission: -

model80: 

	4features.py was used to get this model
	Use BM_BLAST, HB, PLT and number of Somatic mutations (NSM), 2000 epochs, learning rate 5e-5

 	Sequential:
   		torch.nn.BatchNorm1d(4),
            	torch.nn.Linear(4, 32),
            	torch.nn.ReLU(),
            	torch.nn.Dropout(),
            	torch.nn.Linear(32, 64),
            	torch.nn.ReLU(),
           	torch.nn.Dropout(),
            	torch.nn.Linear(64, 128),
            	torch.nn.ReLU(),
            	torch.nn.Dropout(),
            	torch.nn.Linear(128, 1)

	Test Data Indices calculated while training:
		Concordance Index: 0.695
		IPCW Concordance Index: 0.684

  	Obtained score from submission: -
