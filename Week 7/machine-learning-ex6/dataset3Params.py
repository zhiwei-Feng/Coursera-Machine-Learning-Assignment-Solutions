import numpy as np
from sklearn import svm


def dataset_3_params(X, y, Xval, yval):
    results = np.eye(64, 3)
    errorRow = 0

    for C_test in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for sigma_test in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
            gamma = 1.0 / (2.0 * sigma_test ** 2)
            model = svm.SVC(C_test, kernel='rbf', gamma=gamma)
            model.fit(X, y)
            predictions = model.predict(Xval)
            predictions_error = np.mean(predictions != yval)

            results[errorRow, :] = np.array([C_test, sigma_test, predictions_error])
            errorRow += 1

    sorted_results = results[results[:, 2].argsort()]
    C = sorted_results[0, 0]
    sigma = sorted_results[0, 1]

    return C, sigma
