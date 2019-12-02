from __future__ import annotations
import argparse
import os
import shutil
import sys
from importlib import reload
from time import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint
import src.preparation.process_data as p
from src.utils import input_dataset_path, models_path
p = reload(p)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from joblib import dump


USED_COLUMNS = ["la", "ld", "nf", "nd", "ns", "ent", "nrev", "rtime", "hcmt", "ndev", "age", "nuc", "app", "aexp",
                "rexp", "arexp", "rrexp", "asexp", "rsexp", "asawr", "rsawr"]
YCOL = "buggy"
RANDOM_STATE = 0
NUMBER_OF_CV_SPLITS = 5
PRINT_OUTPUT = sys.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='Name of the experiment', type=str, default="")
    parser.add_argument('--project', help='Name of the project to process', choices=['openstack', 'qt'], required=True)
    parser.add_argument('--estimator', help='Estimator', choices=['RF', 'SVC'], required=True)
    parser.add_argument('--cv_by_time', dest='cv_by_time', default=False, action='store_true')
    parser.add_argument('--n_iter_search', help='Number of iterations for search of parameters', type=int,
                        required=True)
    args = parser.parse_args()

    fit_baseline_model(project=args.project,
                       estimator=args.estimator,
                       CV_by_time=args.cv_by_time,
                       n_iter_search=args.n_iter_search,
                       experiment_name=args.experiment_name)


def fit_baseline_model(project: str, estimator: str, CV_by_time: bool,
                       n_iter_search: int, experiment_name: str) -> None:
    global PRINT_OUTPUT
    if experiment_name is not None and len(experiment_name) > 0:
        shutil.rmtree(models_path(project, experiment_name))
        PRINT_OUTPUT = open(os.path.join(models_path(project, experiment_name), 'log'), 'w')

    print(locals(), file=PRINT_OUTPUT)
    df = pd.read_csv(input_dataset_path(project))
    print('Shape before filtering: {}'.format(df.shape), file=PRINT_OUTPUT)

    df = p.filter_as_in_jit_moving_target(df, project)
    print('Shape after filtering: {}'.format(df.shape), file=PRINT_OUTPUT)

    df['self'] = df['self'].astype(float)
    df.fillna({"nrev": 0, "rtime": 1.1 * df.rtime.min(), 'hcmt': -1, 'app': -1, 'rexp': -1, 'self': -1, 'rrexp': -1,
               'rsexp': -1}, inplace=True)

    df_train = df.query('strata < 4').sort_values(by='author_date')
    df_test = df.query('strata >= 4').sort_values(by='author_date')

    n_features = len(USED_COLUMNS)
    print('Number of features used for training {}'.format(n_features), file=PRINT_OUTPUT)

    # generate list of different variables used for prediction:
    features = _get_features_lists_by_importance(df_train)
    if CV_by_time:
        cv = TimeSeriesSplit(NUMBER_OF_CV_SPLITS)
    else:
        cv = NUMBER_OF_CV_SPLITS

    if estimator == 'RF':
        model = ClassifierWithFeatures(estimator_name=estimator, features=(), random_state=0)
        parameters = {'criterion': ['gini', 'entropy'],
                      'n_estimators': sp_randint(10, 1000),
                      'max_depth': [2, 5, 7, 15, 20, None],
                      'min_samples_split': sp_randint(2, 100),
                      'min_samples_leaf': sp_randint(1, 100),
                      'max_features': sp_randint(1,n_features + 1),
                      'max_leaf_nodes': [None, 10, 100, 1000],
                      'features': features,
                      'min_impurity_decrease': [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10]}
    elif estimator == 'SVC':
        model = ClassifierWithFeatures(estimator_name='SVC', features=(), random_state=0, probability=False)
        parameters = {'kernel': ['poly','linear', 'rbf'],
                      'gamma': ['auto'],
                      'C': [0.1, 1, 10],
                      'degree': sp_randint(2, 5),
                      'features': features
                      }
    else:
        raise NotImplementedError

    n_jobs = -1
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    # run randomized search
    random_search = RandomizedSearchCV(model, param_distributions=parameters, n_iter=n_iter_search, verbose=2,
                                       n_jobs=n_jobs, cv=cv, scoring=scorer, iid=False, random_state=RANDOM_STATE)
    start = time()
    random_search.fit(df_train[USED_COLUMNS], df_train[YCOL])
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search), file=PRINT_OUTPUT)
    _report(random_search.cv_results_)

    best_model = random_search.best_estimator_
    if experiment_name is not None and len(experiment_name) > 0:
        dump(best_model, os.path.join(models_path(project, experiment_name), 'model.joblib'))
    print('ROC AUC on test dataset', file=PRINT_OUTPUT)
    print(roc_auc_score(df_test[YCOL], best_model.predict_proba(df_test[USED_COLUMNS])[:, 1]), file=PRINT_OUTPUT)


# Utility function to report best scores
def _report(results: Dict, n_top=3) -> None:
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i), file=PRINT_OUTPUT)
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]), file=PRINT_OUTPUT)
            print("Parameters: {0}".format(results['params'][candidate]), file=PRINT_OUTPUT)


def _append_and_return(l: List[Any], el: Any) ->Tuple[Any]:
    l.append(el)
    return tuple(l)


def _get_features_lists_by_importance(df_train: pd.DataFrame) -> List[Tuple[str]]:
    rf = RandomForestClassifier(100, random_state=0, max_depth=7)
    rf.fit(df_train[USED_COLUMNS], df_train[YCOL])
    feature_importances = pd.DataFrame(rf.feature_importances_,
                                       index=USED_COLUMNS,
                                       columns=['importance']).sort_values('importance', ascending=False)
    features_by_imp = feature_importances.index.tolist()
    l = []
    increasing_in_length_features_lists = [_append_and_return(l, f) for f in features_by_imp]
    return increasing_in_length_features_lists


class ClassifierWithFeatures(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator_name: str, features: List[str], **kwargs: Dict[str, Any]) -> None:
        self.features = features
        self.estimator_name = estimator_name
        if estimator_name == 'RF':
            self.scaler = None
            self.model = RandomForestClassifier(**kwargs)
        elif estimator_name == 'SVC':
            self.scaler = StandardScaler()
            self.model = SVC(**kwargs)
        else:
            raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        _X = X[list(self.features)]
        if self.scaler is not None:
            _X = self.scaler.fit_transform(_X)
        self.model.fit(_X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        _X = X[list(self.features)]
        if len(self.features) == 1:
            _X = np.asarray(_X).reshape(-1, 1)
        if self.scaler is not None:
            _X = self.scaler.transform(_X)
        if self.estimator_name == 'SVC':
            dec_fun = self.model.decision_function(_X)
            return np.stack([-dec_fun, dec_fun], axis=1)
        else:
            return self.model.predict_proba(_X)

    def get_params(self, deep=True) -> Dict[str, Any]:
        params = self.model.get_params(deep)
        params['features'] = self.features
        params['estimator_name'] = self.estimator_name
        return params

    def set_params(self, **parameters: Dict[str, Any]) -> ClassifierWithFeatures:
        for parameter, value in parameters.items():
            if parameter in ['features', 'estimator_name']:
                setattr(self, parameter, value)
            else:
                if parameter == 'max_features':
                    if 'features' in parameters:
                        features = parameters['features']
                    else:
                        features = self.features
                    value = min(value, len(features))
                setattr(self.model, parameter, value)
        return self


if __name__ == "__main__":
    main()
