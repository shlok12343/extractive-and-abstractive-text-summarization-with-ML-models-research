from sklearn.svm import SVC


class SVMSummaryModel:
    """
    Wrapper around an SVM classifier (SVC) so it matches the interface
    expected by the rest of the pipeline (train, predict, .model with
    predict_proba for the summarizer).
    """

    def __init__(self, C: float = 1.0, random_state: int = 42):
        # Linear SVM with probability estimates enabled and class weighting
        # to help with class imbalance.
        self.model = SVC(
            kernel="linear",
            C=C,
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )

    def train(self, X_train, y_train):
        """Train the SVM classifier."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Predict labels for the given feature matrix."""
        return self.model.predict(X)


