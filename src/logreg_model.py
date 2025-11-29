from sklearn.linear_model import LogisticRegression


class LogisticRegressionSummaryModel:
    """
    Wrapper around sklearn's LogisticRegression so it matches the interface
    of the other models used in the summarization pipeline.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        # Binary classifier with class weighting to handle imbalance and
        # probability output enabled for the summarizer.
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )

    def train(self, X_train, y_train):
        """Train the Logistic Regression classifier."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Predict labels for the given feature matrix."""
        return self.model.predict(X)


