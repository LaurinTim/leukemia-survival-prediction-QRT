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

import os
os.chdir(data_dir)

# Import utility functions from model_rsf_utils
import other.utils as u

# %%

# Load training datasets
clinical_train = pd.read_csv(clinical_file)      # clinical features with patient ID
molecular_train = pd.read_csv(molecular_file)    # mutation list per patient ID
target_train = pd.read_csv(target_file)          # survival outcomes: OS_YEARS and OS_STATUS

target_train = target_train.dropna(subset=["OS_YEARS", "OS_STATUS"])
clinical_train = clinical_train[[True if val in list(target_train["ID"]) else False for val in list(clinical_train["ID"])]]
molecular_train = molecular_train[[True if val in list(target_train["ID"]) else False for val in molecular_train["ID"]]]

# Merge clinical and target data on ID (inner join to include only patients with outcome data)
train_df = clinical_train.merge(target_train, on="ID")

# Impute missing numeric values in clinical features with median
numeric_cols = ['BM_BLAST','WBC','ANC','MONOCYTES','HB','PLT']
for col in numeric_cols:
    train_df.loc[:,col] = train_df[col].fillna(train_df[col].median())
    #train_df.loc[:,col] = train_df[col].fillna(0)

print("Training set size:", train_df.shape)
train_df.head(3)

# %%

#train_df.loc[:, "BM_BLAST"] = np.log(train_df["BM_BLAST"]+1e-9)
train_df.loc[:, "PLT"] = np.log(train_df["PLT"]+1e-9)
train_df.loc[:, "WBC"] = np.log(train_df["WBC"]+1e-9)
train_df.loc[:, "ANC"] = np.log(train_df["ANC"]+1e-9)
train_df.loc[:, "MONOCYTES"] = np.log(train_df["MONOCYTES"]+1e-9)
#train_df.loc[:, "HB"] = np.log(train_df["HB"]+1e-9)

#train_df.drop(columns=['BM_BLAST','WBC','ANC','MONOCYTES','HB','PLT'], inplace=True)
#better results if WBC is dropped (only for aft_loss_distribution=extreme)
#train_df.drop(columns=['WBC'], inplace=True)

# %%

# Feature engineering for molecular data:
# 1. Binary mutation indicators for each gene
gene_mutations = molecular_train.groupby(['ID','GENE']).size().unstack(fill_value=0)
gene_mutations = (gene_mutations > 0).astype(int)  # 1 if gene mutated, else 0
print("Number of unique genes in data:", gene_mutations.shape[1])

# 2. Total mutation count per patient
mutation_count = molecular_train.groupby('ID').size()  # series with ID index
mutation_count.name = "mutation_count"

effects_map = u.effect_to_survival_map()
effects_survival = pd.DataFrame([list(molecular_train["ID"]), [effects_map.get(val) for val in molecular_train["EFFECT"]]], index=["ID", "EFFECTS_SURVIVAL"]).T
effects_survival = effects_survival.groupby("ID")["EFFECTS_SURVIVAL"].mean()

effects = molecular_train.groupby(["ID", "EFFECT"]).size().unstack(fill_value=0)
effects = (effects > 0).astype(int)

# Merge gene mutation features into the main training DataFrame
#train_df = train_df.set_index('ID').join(gene_mutations, how='left').join(mutation_count, how='left').join(effects, how="left").join(effects_survival, how="left")
train_df = train_df.set_index('ID').join(mutation_count, how="left").join(gene_mutations, how='left').join(effects, how="left").join(effects_survival, how="left")
train_df.fillna(0, inplace=True)  # fill missing gene indicators (patients with no mutations) with 0
train_df.reset_index(inplace=True)

print("Sample of engineered features:")
#feature_cols = ['mutation_count'] + list(gene_mutations.columns[:5])  # show mutation count and first 5 genes
#print(train_df[feature_cols].head(3))

# %%

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
train_df = train_df.drop(columns=['cyto_category'])

'''
def classify_cytogenetics(cyto_str):
    if pd.isna(cyto_str) or cyto_str == "" or type(cyto_str) == int:
        return "unknown"
    s = cyto_str.lower()
    adverse_markers = ["complex", "-5", "del(5q)", "MONOSOMY 5", "-7", "del(7)", "MONOSOMY 7", "17p", "-17", "inv(3)/t(3;3)", "t(6;9)", "t(9;22)"]
    favorable_markers = ["t(8;21)", "(q22;q22.1)", " inv(16)", "(p13.1q22)", "t(16;16)", "t(15;17) (apl)", "t(15;17)(apl)"]
    if any(mark in s for mark in adverse_markers):
        #print([mark for mark in adverse_markers if mark in s])
        #adverse = True
        return "adverse"
    if any(mark in s for mark in favorable_markers):
        print("yeet")
        return "favorable"
    return "normal"

train_df['cyto_category'] = train_df['CYTOGENETICS'].apply(classify_cytogenetics)

# One-hot encode the cytogenetics category (complex/normal/abnormal)
train_df['cyto_normal'] = (train_df['cyto_category'] == 'normal').astype(int)
train_df['cyto_adverse'] = (train_df['cyto_category'] == 'adverse').astype(int)
train_df['cyto_favorable'] = (train_df['cyto_category'] == 'favorable').astype(int)
# We can drop 'cyto_category' and let 'other abnormal' be implied when both flags are 0
train_df.drop(columns=['CYTOGENETICS','cyto_category'], inplace=True)
'''

# One-hot encode the Center category
#train_df = pd.get_dummies(train_df, columns=['CENTER'], drop_first=True)
train_df.drop(columns=["CENTER", "CYTOGENETICS"], inplace=True)
#train_df.drop(columns=["CENTER"], inplace=True)
print("Final training feature columns:", train_df.columns.tolist())

# %%

from sksurv.util import Surv

# Separate features X and outcome y
# Drop ID and outcome columns from features
X_data = train_df.drop(columns=['ID','OS_YEARS','OS_STATUS'])
# Create structured array for survival outcome
y_data = Surv.from_dataframe(event='OS_STATUS', time='OS_YEARS', data=train_df)
print("X_data shape:", X_data.shape)
print("y_data[0]:", y_data[0])  # example of (event, time) structure

# %%

def sets(X, y, validation_file='Validation_IDs.csv', complete_train=False):
    val_ids = pd.read_csv(data_dir + '\\' + validation_file)
    
    if complete_train:
        X_train = X
        y_train = y
        X_val = X[[True if val in list(val_ids['ID']) else False for val in list(train_df['ID'])]]
        y_val = y[[True if val in list(val_ids['ID']) else False for val in list(train_df['ID'])]]
    
    else:
        X_train = X[[False if val in list(val_ids['ID']) else True for val in list(train_df['ID'])]]
        y_train = y[[False if val in list(val_ids['ID']) else True for val in list(train_df['ID'])]]
        X_val = X[[True if val in list(val_ids['ID']) else False for val in list(train_df['ID'])]]
        y_val = y[[True if val in list(val_ids['ID']) else False for val in list(train_df['ID'])]]
        
    return X_train, X_val, y_train, y_val

X_train, X_val, y_train, y_val = sets(X_data, y_data)

# %%

count = pd.Series(data = np.sum((X_data != 0).astype(int), axis=0), index=X_data.columns)

X_data = X_data.loc[:, list(count > 0)]

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

estimators = 200

cox_model = CoxPHSurvivalAnalysis()  # Cox proportional hazards model
rsf_model = RandomSurvivalForest(n_estimators=estimators, min_samples_split=10, min_samples_leaf=3,
                                 max_features="sqrt", random_state=0)
gb_model  = GradientBoostingSurvivalAnalysis(n_estimators=estimators, learning_rate=0.1,
                                            max_depth=3, max_features=None, random_state=0)

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
    cv=MaxTimeHoldoutKFold(n_splits=5, random_state=0),
    verbose=3
)

rsf_grid.fit(X_data, y_data)
print("Best RSF parameters:", rsf_grid.best_params_)
print("Best RSF IPCW C-index:", rsf_grid.best_score_)

# %%

# Hyperparameter grid for Gradient Boosting Survival
# best params: loss="coxph", learning_rate=0.1, n_estimators=200
# best score: 0.7060
gb_param_grid = {
    "estimator__max_features": [None, "log2", "sqrt"]
    #"estimator__dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    #"estimator__max_depth": [None, 10]
}

gb_grid = GridSearchCV(
    as_concordance_index_ipcw_scorer(gb_model, tau=tau),
    param_grid=gb_param_grid,
    cv=MaxTimeHoldoutKFold(n_splits=5, random_state=0),
    verbose=3
)

gb_grid.fit(X_data, y_data)
print("Best Gradient Boosting parameters:", gb_grid.best_params_)
print("Best Gradient Boosting IPCW C-index:", gb_grid.best_score_)

# %%

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw

Xt, Xv, yt, yv = train_test_split(X_data, y_data, test_size=0.3, random_state=1)

xg = XGBRegressor(objective="survival:cox", eval_metric="cox-nloglik", tree_mode="hist",
                  n_estimators=200, max_depth=4, max_leaves=20, max_bin=9, gamma=0.1, min_child_weight=23,
                  learning_rate=0.06, n_jobs=-1, )

xg.fit(Xt, yt["OS_YEARS"], sample_weight=yt["OS_STATUS"])

pred = xg.predict(Xv)

print(concordance_index_ipcw(yt, yv, pred))

pred = xg.predict(Xt)

print(concordance_index_ipcw(yt, yt, pred))

# %%

import xgboost as xgb

#Xt, Xv, yt, yv = train_test_split(X_data, y_data, test_size=0.3, random_state=1)
Xt, Xv, yt, yv = sets(X_data, y_data, validation_file='Validation_IDs_90.csv', complete_train=False)

# y_data is your structured array of dtype [('status',bool),('time',float)]
times  = yt['OS_YEARS']               # observed time or follow‐up time
status = yt['OS_STATUS'].astype(int)  # 1=event (death), 0=censored

# 1) Build lower‐ and upper‐bound arrays:
y_lower = times.copy()
#   – if event happened, upper bound = exact time
#   – if censored, upper bound = +inf (or a very large number)
y_upper = np.where(status==1,
                   times,
                   np.inf)

# 2) Create a DMatrix with these bounds
dtrain = xgb.DMatrix(Xt,
                     label_lower_bound=y_lower,
                     label_upper_bound=y_upper)

# If you also want a validation split:
times_val  = yv['OS_YEARS']
status_val = yv['OS_STATUS'].astype(int)
y_lower_val = times_val
y_upper_val = np.where(status_val==1, times_val, np.inf)
dval = xgb.DMatrix(Xv,
                   label_lower_bound=y_lower_val,
                   label_upper_bound=y_upper_val)

# 3) Set up AFT parameters
'''
params = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': "normal",  # or 'logistic', 'extreme'
    'aft_loss_distribution_scale':  1.1, #1.1
    'tree_method': 'gpu_hist',   # or 'gpu_hist' if you have a GPU
    'learning_rate': 0.0901, #0.0901
    "max_depth": 6,
    "max_leaves":8,
    "max_bin": 10,
    "gamma": 0.7, #0.7
}

'''
'''
params = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': "extreme",
    'aft_loss_distribution_scale':  1.0,
    'tree_method': 'gpu_hist',
    'learning_rate': 0.0999,
    "max_depth": 6,
    "max_leaves":8,
    "max_bin": 10,
    "gamma": 0.47,
}
'''
params = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': "extreme",
    'aft_loss_distribution_scale':  1.0,
    'tree_method': 'gpu_hist',
    'learning_rate': 0.1,
    "max_depth": 6,
    "max_leaves":8,
    "max_bin": 10,
    "gamma": 0.47,
}


# 4) Train with xgb.train (sklearn wrapper doesn’t yet expose the lower/upper labels)
bst = xgb.train(params,
                dtrain,
                num_boost_round=10000,
                evals=[(dval, 'validation')],
                early_stopping_rounds=100,
                verbose_eval=0)

# 5) Predicting
# bst.predict returns the model’s estimate of T(x) = log(Y) if you use a log‐link
pred_log_time = bst.predict(dval)
# If you want actual time estimates, take exp():
pred_time = np.exp(pred_log_time)

print()
print(concordance_index_ipcw(yt, yv, pred_time))
print(concordance_index_ipcw(yt, yv, 1/pred_time))
print()

pred_log_time = bst.predict(dtrain)
# If you want actual time estimates, take exp():
pred_time = np.exp(pred_log_time)

print(concordance_index_ipcw(yt, yt, pred_time))
print(concordance_index_ipcw(yt, yt, 1/pred_time))

#print()
#print(f"Best iteration: {bst.best_iteration}\nBest score: {bst.best_score:1.6}")

#best_score = 1.16106 #best score with learning rate=0.0901, gamma=0.7
best_score = 1.15132

#print(f"New best score: {round(bst.best_score, 5) < best_score}")
#print(f"{round(bst.best_score, 5)}, {best_score}")

#with aft_loss_distribution=normal best parameters:
#'aft_loss_distribution_scale'=1.0, learning_rate=0.0999, max_depth=6, gamma=0.47
#best score: 1.15132

# %%

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics      import as_concordance_index_ipcw_scorer
from sklearn.model_selection import cross_val_score

# 1) Build a ridge‐penalized Cox (no random_state here)
cox_ridge = CoxnetSurvivalAnalysis(
    l1_ratio = 0.01,
    alpha_min_ratio=0.0005
)

# 2) Wrap it so its .score() returns the IPCW C-index
tau = train_df["OS_YEARS"].max()
wrapped_cox = as_concordance_index_ipcw_scorer(cox_ridge, tau=tau)

# 3) Run cross‐validation without specifying scoring
scores = cross_val_score(
    wrapped_cox,
    X_data,
    y_data,
    cv=MaxTimeHoldoutKFold(n_splits=5, random_state=0),  
    n_jobs=-1
)

cox_ipcw = scores.mean()

#print(scores)
print(f"Cox PH IPCW C-index: {cox_ipcw:.3f}")
print(f"RSF (tuned) IPCW C-index: {rsf_grid.best_score_:.3f}")
print(f"Gradient Boosting (tuned) IPCW C-index: {gb_grid.best_score_:.3f}")

# %%

cox = CoxnetSurvivalAnalysis(l1_ratio = 0.01, alpha_min_ratio=0.0005)
cox.fit(X_train, y_train)
cox_pred = cox.predict(X_val)
cox_ind = concordance_index_ipcw(y_train, y_val, cox_pred)[0]


rsf = RandomSurvivalForest(n_estimators=estimators, min_samples_split=10, min_samples_leaf=3, max_features="sqrt", random_state=0)
rsf.fit(X_train, y_train)
rsf_pred = rsf.predict(X_val)
rsf_ind = concordance_index_ipcw(y_train, y_val, rsf_pred)[0]

gb = GradientBoostingSurvivalAnalysis(n_estimators=estimators, learning_rate=0.1, max_depth=3, max_features=None, random_state=0)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_val)
gb_ind = concordance_index_ipcw(y_train, y_val, gb_pred)[0]

times  = y_train['OS_YEARS']
status = y_train['OS_STATUS'].astype(int)

y_lower = times.copy()
y_upper = np.where(status==1, times, np.inf)

dtrain = xgb.DMatrix(X_train,
                     label_lower_bound=y_lower,
                     label_upper_bound=y_upper)

times_val  = y_val['OS_YEARS']
status_val = y_val['OS_STATUS'].astype(int)
y_lower_val = times_val
y_upper_val = np.where(status_val==1, times_val, np.inf)
dval = xgb.DMatrix(X_val,
                   label_lower_bound=y_lower_val,
                   label_upper_bound=y_upper_val)

xg = xgb.train(params, dtrain, num_boost_round=10000, evals=[(dval, 'validation')], early_stopping_rounds=100, verbose_eval=0)
xg_pred = xg.predict(dval)
xg_pred = 1/xg_pred
xg_ind = concordance_index_ipcw(y_train, y_val, xg_pred)[0]

print(f"COX: {cox_ind:1.5}")
print(f"RSF: {rsf_ind:1.5}")
print(f"GB:  {gb_ind:1.5}")
print(f"XGB: {xg_ind:1.5}")

# %%

# Load test data
clinical_test = pd.read_csv(clinical_file_test)
molecular_test = pd.read_csv(molecular_file_test)

# Aggregate molecular features for test
gene_mutations_test = molecular_test.groupby(['ID','GENE']).size().unstack(fill_value=0)
gene_mutations_test = (gene_mutations_test > 0).astype(int)
mutation_count_test = molecular_test.groupby('ID').size()
mutation_count_test.name = "mutation_count"

effects_survival_test = pd.DataFrame([list(molecular_test["ID"]), [effects_map.get(val) for val in molecular_test["EFFECT"]]], index=["ID", "EFFECTS_SURVIVAL"]).T
effects_survival_test = effects_survival_test.groupby("ID")["EFFECTS_SURVIVAL"].mean()

effects_test = molecular_test.groupby(["ID", "EFFECT"]).size().unstack(fill_value=0)
effects_test = (effects_test > 0).astype(int)

# Merge clinical and molecular for test
test_df = clinical_test.set_index('ID').join(mutation_count_test, how='left').join(gene_mutations_test, how='left').join(effects_test, how='left').join(effects_survival_test, how='left')
for col in numeric_cols:  # use numeric_cols from training step
    test_df[col].fillna(train_df[col].median(), inplace=True)  # impute with training median
    test_df['cyto_category'] = test_df['CYTOGENETICS'].apply(classify_cytogenetics)
test_df.fillna(0, inplace=True)
test_df.reset_index(inplace=True)

# Apply same clinical feature processing:
test_df['cyto_normal'] = (test_df['cyto_category'] == 'normal').astype(int)
test_df['cyto_complex'] = (test_df['cyto_category'] == 'complex').astype(int)
test_df.drop(columns=['CYTOGENETICS','cyto_category'], inplace=True)

'''
# One-hot encode center in test, aligning with training columns
test_df = pd.get_dummies(test_df, columns=['CENTER'], drop_first=True)
# Add any center dummy columns that were in training but not in test (set them to 0)
for col in [c for c in X_data.columns if c.startswith('CENTER_')]:
    if col not in test_df.columns:
        test_df[col] = 0
'''

# Add any gene columns missing in test (set to 0)
for gene in gene_mutations.columns:
    if gene not in test_df.columns:
        test_df[gene] = 0
        
for col in list(X_data.columns):
    if not col in list(test_df.columns):
        test_df.insert(0, col, np.zeros(len(test_df)))

# Ensure test_df has the same feature columns as X_data
X_test = test_df.set_index('ID')[X_data.columns]  # align columns by selecting in training order

# %%

#model = GradientBoostingSurvivalAnalysis(n_estimators=estimators, learning_rate=0.1, max_depth=3, max_features=None, random_state=0)
#model.fit(X_data, y_data)

# Train the final model on the full training data
model = gb_grid.best_estimator_   # RandomSurvivalForest with best params (already refit on full data by GridSearchCV)
# If needed (in case refit=False in GridSearchCV), we would do: best_rsf.fit(X_data, y_data)

# Predict risk scores for test set
risk_scores = model.predict(X_test)
#risk_scores = 1/risk_scores

# Create output DataFrame
output_df = pd.DataFrame({'ID': X_test.index, 'risk_score': risk_scores})
output_df.set_index('ID', inplace=True)
output_df.to_csv(data_dir + "\\submission_files\\submissions2\\sub0.csv")
print("Saved predictions.csv with columns ID and risk_score")

# %%

# Train the final model on the full training data
# If needed (in case refit=False in GridSearchCV), we would do: best_rsf.fit(X_data, y_data)

times_all = y_data['OS_YEARS']
status_all = y_data['OS_STATUS'].astype(int)
y_lower_all = times_all.copy()
y_upper_all = np.where(status_all==1, times_all, np.inf)
dtrain_all = xgb.DMatrix(X_data, label_lower_bound=y_lower_all, label_upper_bound=y_upper_all)

model = xgb.train(params,
                dtrain_all,
                num_boost_round=10000,
                evals=[(dval, 'validation')],
                early_stopping_rounds=100,
                verbose_eval=0)

# Predict risk scores for test set
dtest = xgb.DMatrix(X_test)
risk_scores = model.predict(dtest)
risk_scores = 1/risk_scores

# Create output DataFrame
output_df = pd.DataFrame({'ID': X_test.index, 'risk_score': risk_scores})
output_df.set_index('ID', inplace=True)
output_df.to_csv(data_dir + "\\submission_files\\submissions2\\sub2.csv")
print("Saved predictions.csv with columns ID and risk_score")
























































































