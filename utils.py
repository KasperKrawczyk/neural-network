import numpy as np


def make_blobs(
        n: int = 100,
        means: list[list[float]] = [[1,1], [3,3]],
        covariance_matrix: list[list[float]] = [[0.5, 0.0], [0.0, 0.5]]):
    np.random.seed(2633)

    X = np.vstack([np.random.multivariate_normal(i, covariance_matrix, n) for i in means])
    classes = np.repeat([0, 1], n)\
        .reshape(-1, 1)
    data = np.append(X, classes, axis=1)
    return data
