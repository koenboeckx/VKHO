from sacred import Experiment
from sacred.observers import MongoObserver

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text

import numpy as np

ex = Experiment('online_retail_tree')
#ex.observers.append(MongoObserver())

@ex.config
def cfg():
    criterion = "entropy"

@ex.automain
def run(criterion):
    iris = load_iris()
    X = iris['data']
    y = iris['target']

    n_split = 120
    X_train, X_test = X[:n_split], X[n_split:]
    y_train, y_test = y[:n_split], y[n_split:]

    decision_tree = DecisionTreeClassifier(criterion=criterion)
    decision_tree.fit(X_train, y_train)
    r = export_text(decision_tree, feature_names=iris["feature_names"])
    print(r)

    y_pred = decision_tree.predict(X_test)
    mse = np.mean((y_pred - y_test)**2)

    ex.log_scalar("mean square error", mse)
