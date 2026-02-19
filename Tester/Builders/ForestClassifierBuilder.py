import inspect
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

import Tester.TesterConstants as tester_constants

class StoredSampleWeightsRandomForestClassifier(RandomForestClassifier):
    class StoredSampleWeightsDecisionTreeClassifier(DecisionTreeClassifier):
        def _fit(self, X, y, sample_weight=None, check_input=True, missing_values_in_feature_mask=None):
            self.sample_weight = sample_weight
            return super()._fit(X, y, sample_weight, check_input, missing_values_in_feature_mask)
        
        def fit(self, X, y, sample_weight=None, check_input=True):
            return self._fit(X, y, sample_weight, check_input)
        
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )
        self.estimator = StoredSampleWeightsRandomForestClassifier.StoredSampleWeightsDecisionTreeClassifier()

class StoredSampleWeightsAdaBoostClassifier(AdaBoostClassifier):
    class StoredSampleWeightsDecisionTreeClassifier(DecisionTreeClassifier):
        def _fit(self, X, y, sample_weight=None, check_input=True, missing_values_in_feature_mask=None):
            self.sample_weight = sample_weight
            return super()._fit(X, y, sample_weight, check_input, missing_values_in_feature_mask)
        
        def fit(self, X, y, sample_weight=None, check_input=True):
            return self._fit(X, y, sample_weight, check_input)
        
    def __init__(
        self,
        n_estimators=100,
        *,
        random_state=None
    ):
        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.estimator = StoredSampleWeightsAdaBoostClassifier.StoredSampleWeightsDecisionTreeClassifier()

def build_forest(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model: RandomForestClassifier | AdaBoostClassifier= None
        ) -> RandomForestClassifier | AdaBoostClassifier:
    """
    Build a random forest classifier based on the given data and features.

    Parameters:
        X_train (DataFrame): The training features set.
        y_train (Series): The training labels.
        model (RandomForestClassifier | AdaBoostClassifier, optional): An initialized model to train on (with previous best-chosen hyperparameters).

        If validation data not provided then it is taken from as 0.2 from the training data.

    Returns:
        RandomForestClassifier | AdaBoostClassifier: The forest classifier.
    """
    np.random.seed(tester_constants.constants.RANDOM_STATE)
    
    if model is not None:
        model.fit(X_train, y_train)
        return model
    
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=tester_constants.VALIDATION_SIZE, random_state=tester_constants.constants.RANDOM_STATE)

    # Grid search modification
    modified_X_train, modified_y_train = X_train, y_train
    classes_counts = y_train.value_counts()
    if classes_counts.min() == 1:
        # Duplicate the rows with that one instance
        min_classes = classes_counts[classes_counts == 1].index
        for class_name in min_classes:
            sample_filter = (modified_y_train == class_name)
            modified_X_train = pd.concat([modified_X_train, modified_X_train[sample_filter]], ignore_index=True)
            modified_y_train = pd.concat([modified_y_train, pd.Series([class_name])], ignore_index=True)
    cross_validation_split_count = min(tester_constants.CROSS_VALIDATION_SPLIT_COUNT , modified_y_train.value_counts().min())

    model_class: type[RandomForestClassifier | AdaBoostClassifier]
    model_wrapper_class: type[StoredSampleWeightsRandomForestClassifier | StoredSampleWeightsAdaBoostClassifier]

    match tester_constants.constants.DRIFTING_MODEL:
        case tester_constants.constants.DRIFTING_MODEL.RandomForest:
            model_class = RandomForestClassifier
            model_wrapper_class = StoredSampleWeightsRandomForestClassifier
        case tester_constants.constants.DRIFTING_MODEL.AdaBoost:
            model_class = AdaBoostClassifier
            model_wrapper_class = StoredSampleWeightsAdaBoostClassifier
        case _:
            raise ValueError(f"Unsupported drifting model: {tester_constants.constants.DRIFTING_MODEL}")
    
    forest_classifier = model_class(random_state=tester_constants.constants.RANDOM_STATE)
    # Find best parameters using grid search cross validation (on training data)
    grid_search_classifier = GridSearchCV(estimator=forest_classifier, 
                                     param_grid={param_name: param_values for param_name, param_values in tester_constants.FOREST_PARAM_GRID.items() \
                                                 if param_name in inspect.signature(model_class).parameters}, 
                                     cv=cross_validation_split_count)
    grid_search_classifier.fit(modified_X_train, modified_y_train)
    
    model = model_class(**grid_search_classifier.best_params_, 
                                   random_state=tester_constants.constants.RANDOM_STATE,
                                   )
    model.fit(X_train, y_train)
    model.best_accuracy = accuracy_score(y_validation, model.predict(X_validation))
    
    estimator: model_wrapper_class.StoredSampleWeightsDecisionTreeClassifier
    for estimator in model.estimators_:
        estimator.feature_names_in_ = model.feature_names_in_
        if tester_constants.constants.SYNTHESIZE_BY_BOOTSTRAP:
            estimator.sample_weight = pd.concat([
                pd.Series(estimator.sample_weight, index=X_train.index),
                pd.Series(0, index=X_validation.index)
            ]).sort_index().values
        estimator.best_accuracy = accuracy_score(y_validation, estimator.predict(X_validation))

    return model