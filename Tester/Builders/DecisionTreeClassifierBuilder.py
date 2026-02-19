import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

import Tester.TesterConstants as tester_constants

def build_tree(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model: DecisionTreeClassifier = None
        ) -> DecisionTreeClassifier:
    """
    Build a decision tree classifier based on the given data and features.

    Parameters:
        X_train (DataFrame): The training features set.
        y_train (Series): The training labels.
        model (DecisionTreeClassifier, optional): An initialized model to train on (with previous best-chosen hyperparameters).

        If validation data not provided then it is taken from as 0.2 from the training data.

    Returns:
        DecisionTreeClassifier: The decision tree classifier.
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

    decision_tree_classifier = DecisionTreeClassifier(random_state=tester_constants.constants.RANDOM_STATE)
    # Find best parameters using grid search cross validation (on training data)
    grid_search_classifier = GridSearchCV(estimator=decision_tree_classifier, 
                                     param_grid=tester_constants.TREE_PARAM_GRID, 
                                     cv=cross_validation_split_count)
    grid_search_classifier.fit(modified_X_train, modified_y_train)
    decision_tree_classifier = DecisionTreeClassifier(**grid_search_classifier.best_params_,
                                                      random_state=tester_constants.constants.RANDOM_STATE
                                                      )
    pruning_path = decision_tree_classifier.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = set(pruning_path.ccp_alphas)
    best_decision_tree, best_accuracy = None, -1
    for ccp_alpha in ccp_alphas:
        if ccp_alpha < 0:
            continue
        current_decision_tree = DecisionTreeClassifier(**grid_search_classifier.best_params_,
                                                       ccp_alpha=ccp_alpha,
                                                       random_state=tester_constants.constants.RANDOM_STATE
                                                       )
        current_decision_tree.fit(X_train, y_train)
        current_predictions = current_decision_tree.predict(X_validation)
        current_accuracy = accuracy_score(y_validation, current_predictions)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_decision_tree = current_decision_tree
    
    best_decision_tree.best_accuracy = best_accuracy

    return best_decision_tree