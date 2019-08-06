import numpy as np


def loadDataset():
    X = np.array([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]
    ])
    y = np.array([1.0, 1.0, -1.0, -1.0, 1.0]).reshape(5, 1)
    return X, y


def stump_classify(X, dim, threshold, op='lt'):
    """
    - X: Training data
    - dim: feature index
    - threshold
    - op
    """
    m, _ = X.shape
    y_hat = np.ones((m, 1))

    if op == 'lt':
        y_hat[X[:, dim] < threshold] = -1.0
    else:
        y_hat[X[:, dim] > threshold] = -1.0

    return y_hat


def build_stump(X, y, W):
    """Build a stump

    :X: TODO
    :y: TODO
    :W: TODO
    :returns: TODO
    """
    intervals = 10
    ops = ['lt', 'gt']
    m, n = X.shape

    min_err = np.inf
    best_y = None

    for i in range(n):
        i_min = np.min(X[:, i])
        i_max = np.max(X[:, i])

        span = (i_max - i_min) / intervals

        for j in range(intervals):
            threshold = i_min + j * span
            for op in ops:
                y_hat = stump_classify(X, i, threshold, op)
                err = np.ones((m, 1))
                err[y == y_hat] = 0
                weighted_err = np.dot(W.T, err).flatten()[0]

                # print('error err %3.4f' % weighted_err)
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_y = y_hat.copy()
                    stump = {
                        'dim': i,
                        'op': op,
                        'threshold': threshold
                    }
    return stump, min_err, best_y


class AdaBoost(object):

    """
    Build an AdaBoost classifier
    """

    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []
        return

    def model(self, X, y):
        # Init the weight
        m, n = X.shape
        W = np.ones((m, 1)) / m

        for i in range(self.n_estimators):
            stump, err, y_hat = build_stump(X, y, W)
            err = np.clip(err, 1e-10, err)
            alpha = 0.5 * np.log((1 - err) / err)

            W = W * np.exp(- alpha * y * y_hat)
            W = W / np.sum(W)

            self.estimators.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        """
        Predict
        """
        m, _ = X.shape
        y_hat = np.zeros((m, 1))
        for i in range(self.n_estimators):
            stump = self.estimators[i]
            i_pred = stump_classify(X, stump['dim'], stump['threshold'],
                                    stump['op'])
            y_hat += self.alphas[i] * i_pred

        return np.sign(y_hat)


def main():
    """
    :returns: TODO

    """
    pass
    X, y = loadDataset()
    m, n = X.shape
    boost = AdaBoost(10)
    boost.model(X, y)
    y_hat = boost.predict(X)
    print('Accurcy: ', np.sum(y_hat == y) / m * 100)


if __name__ == "__main__":
    main()
