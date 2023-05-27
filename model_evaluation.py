from data_processing import actions

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


def evaluate_model(X_test, y_test, model):
    res = model.predict(X_test)
    print(actions[np.argmax(res[0])])
    print(actions[np.argmax(y_test[0])])

    yhat = model.predict(X_test)

    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    print(multilabel_confusion_matrix(ytrue, yhat))
    print(accuracy_score(ytrue, yhat))