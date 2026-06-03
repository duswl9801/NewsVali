import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500) # prevent overflow
    return 1 / (1 + np.exp(-z))

class MyLogisticRegression:
    def __init__(self, epochs, lr, threshold, verbose=True):
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.verbose = verbose

        self.weights = None
        self.bias = None
        self.loss_history = []

    def compute_BCE(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1-eps)

        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        return loss

    #TODO: L2 regularization

    def fit(self, X, y, tracker=None):
        y = np.asarray(y).ravel().astype(np.float64) # convert labels to a 1D numeric array.

        n_samples, n_features = X.shape

        # Logistic regression can use zero initialization because there is no hidden-layer symmetry problem.
        # Even if all weights start at zero, each feature can get a different gradient.
        self.weights = np.zeros(n_features)
        self.bias = 0

        for ep in range(self.epochs):
            z = X.dot(self.weights) + self.bias
            y_pred = sigmoid(z)

            loss = self.compute_BCE(y, y_pred)
            self.loss_history.append(loss)

            dw = (1 / n_samples) * X.T.dot(y_pred - y) # gradient of the loss
            db = (1 / n_samples) * np.sum(y_pred - y) # gradient of the loss

            # update weights and bias(training)
            self.weights = self.weights - self.lr * dw
            self.bias -= self.lr * db

            if tracker is not None:
                tracker.write_row({
                    "epoch": ep,
                    "phase": "train",
                    "train_loss": loss
                })

            if self.verbose:
                print(f"Epoch {ep:04d}/{self.epochs} | Train Loss: {loss:.6f}")

    # for test
    def predict_proba(self, X):
        z = X.dot(self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X):
        y_prob = self.predict_proba(X)
        label = (y_prob >= self.threshold).astype(int) # over threshold => 1 | less => 0
        return y_prob, label
