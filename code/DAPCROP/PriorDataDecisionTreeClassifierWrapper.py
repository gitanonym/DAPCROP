import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier

class PriorDataDecisionTreeClassifierWrapper(DecisionTreeClassifier):
    """
    A wrapper for DecisionTreeClassifier that accepts prior data for training.
    """
    def __init__(self,
                 model: DecisionTreeClassifier,
                 X_prior: pd.DataFrame,
                 y_prior: pd.Series):
        
        self.__dict__.update(model.__dict__)

        self.X_prior = X_prior
        self.y_prior = y_prior

        self.model = model # Stored for sklearn compatibility

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series, sample_weight: pd.Series = None):
        combined_X = pd.concat([self.X_prior, X], ignore_index=True)
        combined_y = pd.concat([self.y_prior, y], ignore_index=True)
        if sample_weight is not None:
            # Prior data gets weight 1.0
            prior_weights = pd.Series(1.0, index=range(len(self.X_prior)))
            combined_weight = pd.concat([prior_weights, sample_weight.reset_index(drop=True)], ignore_index=True)
        else:
            combined_weight = None
        return super().fit(combined_X, combined_y, sample_weight=combined_weight)