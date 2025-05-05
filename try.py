import pandas as pd
import numpy as np

# Path to directory containing the project
data_dir = "C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\leukemia-survival-prediction-QRT"

# Paths to files used for the training
target_file = data_dir+'\\target_train.csv'  # Contains survival information about patients, used as training target
clinical_file = data_dir+'\\X_train\\clinical_train.csv'  # Clinical information of patients used for training
molecular_file = data_dir+'\\X_train\\molecular_train.csv'  # Molecular information of patients used for training

# Paths to the test files used for submissions
clinical_file_test = data_dir+'\\X_test\\clinical_test.csv'  # Clinical test data for model submission
molecular_file_test = data_dir+'\\X_test\\molecular_test.csv'  # Molecular test data for model submission

# Load training datasets
clinical_train = pd.read_csv(clinical_file)      # clinical features with patient ID
molecular_train = pd.read_csv(molecular_file)    # mutation list per patient ID
target_train = pd.read_csv(target_file)          # survival outcomes: OS_YEARS and OS_STATUS

target_train = target_train.dropna(subset=["OS_YEARS", "OS_STATUS"])
clinical_train = clinical_train[[True if val in list(target_train["ID"]) else False for val in list(clinical_train["ID"])]]
molecular_train = molecular_train[[True if val in list(target_train["ID"]) else False for val in molecular_train["ID"]]]

# Merge clinical and target data on ID (inner join to include only patients with outcome data)
train_df = clinical_train.merge(target_train, on="ID")
print("Training set size:", train_df.shape)
train_df.head(3)

# %%

# Feature engineering for molecular data:
# 1. Binary mutation indicators for each gene
gene_mutations = molecular_train.groupby(['ID','GENE']).size().unstack(fill_value=0)
gene_mutations = (gene_mutations > 0).astype(int)  # 1 if gene mutated, else 0
print("Number of unique genes in data:", gene_mutations.shape[1])

# 2. Total mutation count per patient
mutation_count = molecular_train.groupby('ID').size()  # series with ID index
mutation_count.name = "mutation_count"

# Merge gene mutation features into the main training DataFrame
train_df = train_df.set_index('ID').join(gene_mutations, how='left').join(mutation_count, how='left')
train_df.fillna(0, inplace=True)  # fill missing gene indicators (patients with no mutations) with 0
train_df.reset_index(inplace=True)

print("Sample of engineered features:")
feature_cols = ['mutation_count'] + list(gene_mutations.columns[:5])  # show mutation count and first 5 genes
print(train_df[feature_cols].head(3))

# %%

# Impute missing numeric values in clinical features with median
numeric_cols = ['BM_BLAST','WBC','ANC','MONOCYTES','HB','PLT']
for col in numeric_cols:
    train_df.loc[:,col] = train_df[col].fillna(train_df[col].median())

# Encode CYTOGENETICS into categorical flags
def classify_cytogenetics(cyto_str):
    if pd.isna(cyto_str) or cyto_str == "" or type(cyto_str) == int:
        return "unknown"
    s = cyto_str.lower()
    if "complex" in s or ">3abnormal" in s:
        return "complex"
    # Consider it normal if it contains no typical abnormality markers
    # (We assume purely "46,xx" or "46,xy" with no other aberration means normal)
    abnormalities = ["del(", "dup(", "inv(", "t(", "+", "-"]
    if any(mark in s for mark in abnormalities):
        return "abnormal"
    return "normal"

train_df['cyto_category'] = train_df['CYTOGENETICS'].apply(classify_cytogenetics)
# One-hot encode the cytogenetics category (complex/normal/abnormal)
train_df['cyto_normal'] = (train_df['cyto_category'] == 'normal').astype(int)
train_df['cyto_complex'] = (train_df['cyto_category'] == 'complex').astype(int)
# We can drop 'cyto_category' and let 'other abnormal' be implied when both flags are 0
train_df.drop(columns=['CYTOGENETICS','cyto_category'], inplace=True)

# One-hot encode the Center category
train_df = pd.get_dummies(train_df, columns=['CENTER'], drop_first=True)
print("Final training feature columns:", train_df.columns.tolist())

# %%

from sksurv.util import Surv

# Separate features X and outcome y
# Drop ID and outcome columns from features
X_train = train_df.drop(columns=['ID','OS_YEARS','OS_STATUS'])
# Create structured array for survival outcome
y_train = Surv.from_dataframe(event='OS_STATUS', time='OS_YEARS', data=train_df)
print("X_train shape:", X_train.shape)
print("y_train[0]:", y_train[0])  # example of (event, time) structure

# %%

from sklearn.model_selection import KFold, GridSearchCV
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import as_concordance_index_ipcw_scorer

class MaxTimeHoldoutKFold:
    """K‐fold CV that always holds out the same one patient (with max time) from splitting."""
    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, groups=None):
        # y is a structured array with fields ('event', 'time')
        times = y['OS_YEARS']
        # find the single index of the patient with maximum observed time
        max_idx = np.argmax(times)
        # the rest of the indices
        rest_idx = np.delete(np.arange(len(times)), max_idx)
        # now apply a standard KFold to the "rest" indices
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_rest, test_rest in kf.split(rest_idx):
            # always include max_idx in the training set
            train_idx = np.concatenate(([max_idx], rest_idx[train_rest]))
            test_idx  = rest_idx[test_rest]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# %%

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

tau = train_df["OS_YEARS"].max()

cox_model = CoxPHSurvivalAnalysis()  # Cox proportional hazards model
rsf_model = RandomSurvivalForest(n_estimators=1000, min_samples_split=10, min_samples_leaf=3,
                                 random_state=0)
gb_model  = GradientBoostingSurvivalAnalysis(n_estimators=1000, learning_rate=0.1,
                                            max_depth=3, random_state=0)

# %%

# Define IPCW C-index scorer (tau set to max observed time to consider full range)
ipcw_scorer = as_concordance_index_ipcw_scorer(rsf_model, tau=tau)

# Hyperparameter grid for Random Survival Forest
# best params: n_extimators=300, max_depth=None, min_samples_leaf=3
rsf_param_grid = {
    "estimator__max_features": [None, "log2", "sqrt"]
    #"estimator__max_depth": [None, 10]
}

rsf_grid = GridSearchCV(
    as_concordance_index_ipcw_scorer(rsf_model, tau=tau),
    param_grid=rsf_param_grid,
    cv=MaxTimeHoldoutKFold(n_splits=10, random_state=0),
    verbose=3
)

rsf_grid.fit(X_train, y_train)
print("Best RSF parameters:", rsf_grid.best_params_)
print("Best RSF IPCW C-index:", rsf_grid.best_score_)

# %%

# Hyperparameter grid for Gradient Boosting Survival
# best params: loss="coxph", learning_rate=0.1, n_estimators=200
# best score: 0.7060
gb_param_grid = {
    "estimator__max_features": [None, "log2", "sqrt"]
    #"estimator__max_depth": [None, 10]
}

gb_grid = GridSearchCV(
    as_concordance_index_ipcw_scorer(gb_model, tau=tau),
    param_grid=gb_param_grid,
    cv=MaxTimeHoldoutKFold(n_splits=10, random_state=0),
    verbose=3
)
gb_grid.fit(X_train, y_train)
print("Best Gradient Boosting parameters:", gb_grid.best_params_)
print("Best Gradient Boosting IPCW C-index:", gb_grid.best_score_)

# %%

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics      import as_concordance_index_ipcw_scorer
from sklearn.model_selection import cross_val_score

# 1) Build a ridge‐penalized Cox (no random_state here)
cox_ridge = CoxnetSurvivalAnalysis(
    l1_ratio         = 0.01    # pure L2 penalty
)

# 2) Wrap it so its .score() returns the IPCW C-index
tau = train_df["OS_YEARS"].max()
wrapped_cox = as_concordance_index_ipcw_scorer(cox_ridge, tau=tau)

# 3) Run cross‐validation without specifying scoring
scores = cross_val_score(
    wrapped_cox,
    X_train,
    y_train,
    cv=MaxTimeHoldoutKFold(n_splits=5, random_state=0),  # use your custom CV
    n_jobs=-1
)

cox_ipcw = scores.mean()

print(f"Cox PH IPCW C-index: {cox_ipcw:.3f}")
print(f"RSF (tuned) IPCW C-index: {rsf_grid.best_score_:.3f}")
print(f"Gradient Boosting (tuned) IPCW C-index: {gb_grid.best_score_:.3f}")

# %%

# Load test data
clinical_test = pd.read_csv('clinical_test.csv')
molecular_test = pd.read_csv('molecular_test.csv')

# Aggregate molecular features for test
gene_mutations_test = molecular_test.groupby(['ID','GENE']).size().unstack(fill_value=0)
gene_mutations_test = (gene_mutations_test > 0).astype(int)
mutation_count_test = molecular_test.groupby('ID').size()
mutation_count_test.name = "mutation_count"

# Merge clinical and molecular for test
test_df = clinical_test.set_index('ID').join(gene_mutations_test, how='left').join(mutation_count_test, how='left')
test_df.fillna(0, inplace=True)
test_df.reset_index(inplace=True)

# Apply same clinical feature processing:
for col in numeric_cols:  # use numeric_cols from training step
    test_df[col].fillna(train_df[col].median(), inplace=True)  # impute with training median
test_df['cyto_category'] = test_df['CYTOGENETICS'].apply(classify_cytogenetics)
test_df['cyto_normal'] = (test_df['cyto_category'] == 'normal').astype(int)
test_df['cyto_complex'] = (test_df['cyto_category'] == 'complex').astype(int)
test_df.drop(columns=['CYTOGENETICS','cyto_category'], inplace=True)

# One-hot encode center in test, aligning with training columns
test_df = pd.get_dummies(test_df, columns=['CENTER'], drop_first=True)
# Add any center dummy columns that were in training but not in test (set them to 0)
for col in [c for c in X_train.columns if c.startswith('CENTER_')]:
    if col not in test_df.columns:
        test_df[col] = 0

# Add any gene columns missing in test (set to 0)
for gene in gene_mutations.columns:
    if gene not in test_df.columns:
        test_df[gene] = 0

# Ensure test_df has the same feature columns as X_train
X_test = test_df.set_index('ID')[X_train.columns]  # align columns by selecting in training order

# %%

# Train the final model on the full training data
best_rsf = rsf_grid.best_estimator_   # RandomSurvivalForest with best params (already refit on full data by GridSearchCV)
# If needed (in case refit=False in GridSearchCV), we would do: best_rsf.fit(X_train, y_train)

# Predict risk scores for test set
risk_scores = best_rsf.predict(X_test)

# Create output DataFrame
output_df = pd.DataFrame({'ID': X_test.index, 'risk_score': risk_scores})
output_df.set_index('ID', inplace=True)
output_df.to_csv(data_dir + "\\submission_files\\submissions2\\sub0.csv")
print("Saved predictions.csv with columns ID and risk_score")


























































































