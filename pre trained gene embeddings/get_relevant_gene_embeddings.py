import numpy as np
import pandas as pd

#pre trained gene embeddings from https://doi.org/10.1186/s12864-018-5370-x
gene_embeddings = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT\\pre trained gene embeddings\\gene2vec_dim_200_iter_9.txt", 
                              delimiter="\s+", header=None, index_col=0).T.reset_index(drop=True)

#The "KMT2A" gene is the same as "MLL", our data used "MLL"
gene_embeddings = gene_embeddings.rename({"KMT2A": "MLL"}, axis=1)

molecular_data = pd.read_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT\\X_train\\molecular_train.csv")
unique_genes = pd.unique(molecular_data["GENE"])
gene_embeddings_present = pd.DataFrame(0, index = np.arange(0,gene_embeddings.shape[0]), columns = unique_genes)

for gene in unique_genes:
    curr_embedding = gene_embeddings[gene]
    gene_embeddings_present[gene] = curr_embedding
    
gene_embeddings_present.to_csv("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT\\pre trained gene embeddings\\gene2vec_relevant_genes.txt", index=False)