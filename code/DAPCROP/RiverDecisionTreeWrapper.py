import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier

try:
    from river.tree import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier, ExtremelyFastDecisionTreeClassifier
    from river.stream import iter_pandas
except ImportError as e:
    def raise_error(*args, **kwargs):
        raise ImportError("River library is not installed. Please install it to use RiverDecisionTreeWrapper.") from e
    HoeffdingTreeClassifier = raise_error
    HoeffdingAdaptiveTreeClassifier = raise_error
    ExtremelyFastDecisionTreeClassifier = raise_error
    iter_pandas = raise_error

import DAPCROP.Constants as constants

class RiverDecisionTreeWrapper(DecisionTreeClassifier):
    """
    A wrapper for the RiverDecisionTreeClassifiers from the river library to make it compatible with sklearn's DecisionTreeClassifier interface.
    """
    def __init__(self,
                 X_prior: pd.DataFrame = None,
                 y_prior: pd.Series = None,
                 subtree_type: constants.SubTreeType = constants.DEFAULT_SUBTREE_TYPE,
                 **kwargs):
        super().__init__()
        
        assert (X_prior is None) == (y_prior is None), "X_pretrain and y_pretrain must be both provided or None"
        
        if "nominal_attributes" not in kwargs:
            assert X_prior is not None, "X_prior must be provided to infer nominal attributes in case not specified"
            kwargs["nominal_attributes"] = list(filter(lambda column_name: X_prior[column_name].dtype in [object, bool], X_prior.columns))

        self.model = None
        try:
            self.model: HoeffdingTreeClassifier = globals()[subtree_type.name](**kwargs)
        except Exception as e:
            raise ValueError(f"Error while trying to create subtree type: {subtree_type}") from e
        
        self.subtree_type = subtree_type

        self.X_prior, self.y_prior = X_prior, y_prior

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series = None) -> None:
        """
        Fit the ExtremelyFastDecisionTreeClassifier model.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target labels.
            sample_weight (pd.Series, optional): Per-sample weights aligned with X.

        Returns:
            ExtremelyFastDecisionTreeWrapper: The fitted model.
        """
        current_weight = 1
        
        if self.X_prior is not None:
            single_weight = len(self.X_prior) + len(X)
            # balance the weights between prior and current data
            prior_weight, current_weight = (len(X) / single_weight), \
                                            (len(self.X_prior) / single_weight)
            for x_i, y_i in iter_pandas(self.X_prior, self.y_prior):
                self.model.learn_one(x_i, y_i, w=prior_weight)
        
        # Make the model more sensitive
        self.model.grace_period = int(self.model.grace_period * current_weight)
        self.model.delta = 0.05
        self.model.tau = 0.1
        match self.subtree_type:
            case constants.SubTreeType.ExtremelyFastDecisionTreeClassifier:
                self.model.min_samples_reevaluate = 5
            case constants.SubTreeType.HoeffdingAdaptiveTreeClassifier:
                self.model.drift_window_threshold = 10

        if sample_weight is not None:
            for (x_i, y_i), w_i in zip(iter_pandas(X, y), sample_weight):
                self.model.learn_one(x_i, y_i, w=current_weight * w_i)
        else:
            for x_i, y_i in iter_pandas(X, y):
                self.model.learn_one(x_i, y_i, w=current_weight)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the fitted ExtremelyFastDecisionTreeClassifier model.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            pd.Series: The predicted labels.
        """
        predictions = []
        for x_i, _ in iter_pandas(X):
            predictions.append(self.model.predict_one(x_i))
        return np.array(predictions)