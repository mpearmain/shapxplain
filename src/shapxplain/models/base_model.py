"""
Base models used for SHAP and LLM explanations.
"""

from sklearn.ensemble import RandomForestClassifier

class ExampleModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
