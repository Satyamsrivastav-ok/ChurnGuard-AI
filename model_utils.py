class ThresholdRandomForest:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.feature_names_in_ = getattr(model, "feature_names_in_", None)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    @property
    def classes_(self):
        return getattr(self.model, "classes_", None)
