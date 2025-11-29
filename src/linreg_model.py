from sklearn.linear_model import LinearRegression
import numpy as np


class LinearRegressionSummaryModel:
    """
    Wrapper around sklearn's LinearRegression. This is a regression model
    trained on labels 0/1; we threshold its continuous outputs at 0.5 to
    obtain class predictions for evaluation.
    """

    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        """Train the Linear Regression model on 0/1 labels."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict class labels by thresholding the continuous regression outputs
        at 0.5 so we get 0/1 predictions for classification metrics.
        """
        preds = self.model.predict(X)
        return (preds >= 0.5).astype(int)


