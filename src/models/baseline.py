import pandas as pd
import sys
import numpy as np
from time import time
import plotly.express as px
sys.path.insert(1, "../")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from src.utils import input_dataset_path
import src.preparation.process_data as p
from importlib import reload
p = reload(p)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


USED_COLUMNS = ["la", "ld", "nf", "nd", "ns", "ent", "nrev", "rtime", "hcmt", "ndev", "age", "nuc", "app", "aexp", "rexp", "arexp", "rrexp", "asexp", "rsexp", "asawr", "rsawr"]
YCOL = "buggy"


def fit_baseline_model(project, n_iter_search):
    df = pd.read_csv(input_dataset_path(project))
    df.shape
    df2 = p.filter_as_in_jit_moving_target(df, project)
    df2.shape

    df['self'] = df['self'].astype(float)
    df.fillna({"nrev": 0, "rtime": 1.1*df.rtime.min(), 'hcmt': -1, 'app': -1, 'rexp': -1, 'self': -1, 'rrexp': -1, 'rsexp': -1}, inplace=True)

    df_train = df.query('strata < 4').sort_values(by='author_date')
    df_test = df.query('strata >= 4').sort_values(by='author_date')

    n_features = len(USED_COLUMNS)
    print(n_features)

    rf = RandomForestClassifier(random_state=0)
    cv = TimeSeriesSplit(5)
    parameters = {'criterion':['gini', 'entropy'],
                  'n_estimators': [10, 100, 500, 800],
                  'max_depth':[2, 5, 7, 15, 20],
                  'min_samples_split': [2, 10, 50, 100],
                  'min_samples_leaf': [1, 10, 50, 100],
                  'max_features':[1, 2,int(n_features ** 0.5), 7, n_features],
                  'max_leaf_nodes':[None, 10, 100, 1000],
                  'min_impurity_decrease':[0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10]}

    n_jobs = -1
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    # run randomized search
    random_search = RandomizedSearchCV(rf, param_distributions=parameters, n_iter=n_iter_search,
                                       n_jobs=n_jobs, cv=cv, scoring=scorer, iid=False)
    start = time()
    random_search.fit(df_train[used_columns], df_train[ycol])
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

# Utility function to report best scores
def _report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


