from sklearn.naive_bayes import MultinomialNB


class NaiveBayesSummaryModel:
    """
    Wrapper around MultinomialNB so it has a similar interface to SummaryModel.

    - Exposes `.model` holding the underlying sklearn classifier.
    - Provides `.train(X, y)` and `.predict(X)` methods.
    """

    def __init__(self):
        # MultinomialNB works well with count/TF-IDF style features.
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        """Train the Naive Bayes classifier."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Predict labels for the given feature matrix."""
        return self.model.predict(X)


