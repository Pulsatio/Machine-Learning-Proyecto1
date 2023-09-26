import numpy as np

class MultiClassSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.classifiers = {}

    def fit(self, X, y):
        unique_classes = np.unique(y)

        for cls in unique_classes:
            binary_y = np.where(y == cls, 1, -1)
            self.classifiers[cls] = self.train_binary_classifier(X, binary_y)

    def train_binary_classifier(self, X, y):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, w) - b) >= 1
                if condition:
                    w -= self.learning_rate * (2 * self.lambda_param * w)
                else:
                    w -= self.learning_rate * (2 * self.lambda_param * w - np.dot(x_i, y[idx]))
                    b -= self.learning_rate * y[idx]

        return (w, b)

    def score(self, X, y):
        predictions = [self.predict(x) for x in X]

        results = [int(y[i] == predictions[i]) for i in range(len(predictions))]
        return sum(results)/len(results)

    def predict(self, X):
        class_scores = {}

        for cls, classifier in self.classifiers.items():
            w, b = classifier
            class_scores[cls] = np.dot(X, w) - b

        return max(class_scores, key=class_scores.get).any()
