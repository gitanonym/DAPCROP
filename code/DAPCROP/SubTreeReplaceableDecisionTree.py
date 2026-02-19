from copy import deepcopy

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

import DAPCROP.Constants as constants

from .ModelMapping.ATreeBasedMappedModel import ATreeBasedMappedModel
from .ModelMapping.TreeNodeComponent import TreeNodeComponent

from .RiverDecisionTreeWrapper import RiverDecisionTreeWrapper as RiverDecisionTree
from .PriorDataDecisionTreeClassifierWrapper import PriorDataDecisionTreeClassifierWrapper as PriorDataDecisionTreeClassifier

class SubTreeReplaceableDecisionTree(DecisionTreeClassifier):
    """
    A Decision Tree Classifier that allows replacing subtrees.
    """
    def __init__(self,
                 original_tree: DecisionTreeClassifier,
                 nodes_to_replace: list[TreeNodeComponent],
                 dependency_handling_type: constants.SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES = constants.DEFAULT_SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPE,
                 use_prior_knowledge: constants.PRIOR_KNOWLEDGE_USAGE_TYPES = constants.DEFAULT_USE_OF_PRIOR_KNOWLEDGE,
                 subtree_type: constants.SubTreeType = constants.DEFAULT_SUBTREE_TYPE,
                 X_prior: pd.DataFrame = None,
                 y_prior: pd.Series = None
                 ):
        # Initialize parent with same parameters as original tree
        super().__init__(
            criterion=original_tree.criterion,
            splitter=original_tree.splitter,
            max_depth=original_tree.max_depth,
            min_samples_split=original_tree.min_samples_split,
            min_samples_leaf=original_tree.min_samples_leaf,
            min_weight_fraction_leaf=original_tree.min_weight_fraction_leaf,
            max_features=original_tree.max_features,
            random_state=original_tree.random_state,
            max_leaf_nodes=original_tree.max_leaf_nodes,
            min_impurity_decrease=original_tree.min_impurity_decrease,
            class_weight=getattr(original_tree, 'class_weight', None),
            ccp_alpha=getattr(original_tree, 'ccp_alpha', 0.0),
            monotonic_cst=getattr(original_tree, 'monotonic_cst', None)
        )
        self.base_sklearn_tree_model: DecisionTreeClassifier = deepcopy(original_tree)
        
        self.replacement_candidates: list[TreeNodeComponent] = nodes_to_replace
        self.replaced_subtrees: dict[TreeNodeComponent, DecisionTreeClassifier] = {}

        assert isinstance(dependency_handling_type, constants.SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES), f"expecting dependency_handling_type to be SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES but got {type(dependency_handling_type)}"
        self.dependency_handling_type = dependency_handling_type

        assert isinstance(use_prior_knowledge, constants.PRIOR_KNOWLEDGE_USAGE_TYPES), f"expecting use_prior_knowledge to be PRIOR_KNOWLEDGE_USAGE_TYPES but got {type(use_prior_knowledge)}"
        self.use_prior_knowledge = use_prior_knowledge

        assert isinstance(subtree_type, constants.SubTreeType), f"expecting subtree_type to be SubTreeType but got {type(subtree_type)}"
        self.subtree_type = subtree_type
        
        self.X_prior = X_prior
        self.y_prior = y_prior
    
    def get_candidate_conflicts_indices(self,
                                        current_candidate_index: int,
                                        current_candidate_node: TreeNodeComponent) -> list[int]:
        all_candidate_conflict_indices = []
        for lower_candidate_index in range(current_candidate_index + 1, len(self.replacement_candidates)):
            if current_candidate_node.is_ancestor_of(self.replacement_candidates[lower_candidate_index]):
                all_candidate_conflict_indices.append(lower_candidate_index)
        return all_candidate_conflict_indices

    def get_unrelated_child(self,
                            node: TreeNodeComponent,
                            successor_node: TreeNodeComponent
                            ) -> TreeNodeComponent:
        """
        Get a child of the given node that is not on the same path as the successor node.
        """
        assert node.is_ancestor_of(successor_node), "The given node is not an ancestor of the successor node."
        if node.left_child.is_ancestor_of(successor_node):
            return node.right_child
        return node.left_child
    
    def resolve_candidate_conflicts(self, 
                                    current_candidate_index: int) -> int:
        """
        Resolve conflicts for the current candidate node.
        
        current_candidate_index (int): The index of the current candidate node.
        Returns:
            int: The next candidate index to look for conflicts at.
        """        
        current_candidate_node = self.replacement_candidates[current_candidate_index]
        
        all_candidate_conflict_indices = self.get_candidate_conflicts_indices(current_candidate_index, current_candidate_node)
        
        if not all_candidate_conflict_indices:
            # No conflicts, move to the next candidate
            return current_candidate_index + 1
        
        match self.dependency_handling_type:
            case constants.SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES.Take_Top:
                # Remove all successors
                for candidate_conflict_index in all_candidate_conflict_indices:
                    del self.replacement_candidates[candidate_conflict_index]
                # No change required to the current candidate, move to the next one
                return current_candidate_index + 1
            case constants.SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES.Replace_Ancestors:        
                if len(self.replacement_candidates) >= 2 and self.replacement_candidates[all_candidate_conflict_indices[0]].get_sibling() == self.replacement_candidates[all_candidate_conflict_indices[1]]:
                    # Both children need to be replaced, remove the current node (the parent)
                    del self.replacement_candidates[current_candidate_index]
                else:
                    # No more than one direct-child needs to be replaced.
                    highest_successor_node = self.replacement_candidates[all_candidate_conflict_indices[0]]
                    unrelated_child = self.get_unrelated_child(current_candidate_node, highest_successor_node)
                    self.replacement_candidates[current_candidate_index] = unrelated_child
                # changes were made, re-check the current index
                return current_candidate_index
            case _:
                raise NotImplementedError("The specified dependency handling type is not implemented yet for this fixer")

    def resolve_candidates_conflicts(self):
        """
        Resolve cases where there are multiple nodes on the same path.
        """
        self.replacement_candidates.sort(key=lambda node: node.depth)
        current_candidate_index = 0
        while current_candidate_index < len(self.replacement_candidates):
            current_candidate_index = self.resolve_candidate_conflicts(current_candidate_index)

    def create_replaceable_subtree(self, 
                                   node_to_replace: TreeNodeComponent,
                                   X_prior: pd.DataFrame = None,
                                   y_prior: pd.Series = None) -> DecisionTreeClassifier:
        match self.use_prior_knowledge:
            case constants.PRIOR_KNOWLEDGE_USAGE_TYPES.Ignore:
                return deepcopy(self.base_sklearn_tree_model)
            case constants.PRIOR_KNOWLEDGE_USAGE_TYPES.Use:
                X_prior, y_prior = node_to_replace.get_data_reached_node(X_prior, y_prior, allow_empty=False)
            case constants.PRIOR_KNOWLEDGE_USAGE_TYPES.Synthesize:
                X_prior, y_prior = node_to_replace.synthesize_data_reached_node()

        if self.subtree_type == constants.SubTreeType.Original:
            return PriorDataDecisionTreeClassifier(deepcopy(self.base_sklearn_tree_model), X_prior=X_prior, y_prior=y_prior)
        
        tree_kwargs = {
            "split_criterion": self.base_sklearn_tree_model.criterion.replace("entropy", "info_gain"),
            "grace_period": 50,
            "delta": 0.01,
            "tau": 0.05
        }
        if self.subtree_type == constants.SubTreeType.HoeffdingAdaptiveTreeClassifier:
            tree_kwargs["seed"] = constants.RANDOM_STATE
            tree_kwargs["drift_window_threshold"] = 30

        return RiverDecisionTree(X_prior=X_prior, y_prior=y_prior, subtree_type=self.subtree_type, **tree_kwargs)
    
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            sample_weight: pd.Series = None) -> 'SubTreeReplaceableDecisionTree':
        """
        Fit the decision tree to the data.

        Parameters:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target labels.
            sample_weight (pd.Series, optional): Per-sample weights aligned with X.
            
        Returns:
            SubTreeReplaceableDecisionTree: Returns self for method chaining.
        """
        self.resolve_candidates_conflicts()
        
        for node_to_replace in self.replacement_candidates:
            X_reached, y_reached = node_to_replace.get_data_reached_node(X, y, allow_empty=False)
            w_reached = sample_weight.loc[X_reached.index] if sample_weight is not None else None
            self.replaced_subtrees[node_to_replace] = self.create_replaceable_subtree(node_to_replace, self.X_prior, self.y_prior)
            self.replaced_subtrees[node_to_replace].fit(X_reached, y_reached, sample_weight=w_reached)

        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.classes_ = np.unique(list(self.replacement_candidates[0].get_node_class_count().keys()))
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = 1
        
        self.tree_ = self.base_sklearn_tree_model.tree_
        self.max_features_ = getattr(self.base_sklearn_tree_model, 'max_features_', None)
        
        return self

    def predict(self, X: pd.DataFrame | np.ndarray, check_input=False) -> np.ndarray:
        """
        Predict the class labels for the given input data.

        Parameters:
            X (pd.DataFrame | np.ndarray): The input features.
            check_input (bool): Whether to check the input features. MATCHING THE BASE BUT NOT USING IT.

        Returns:
            np.ndarray: The predicted class labels.
        """
        # Use predict_proba and take argmax to get class predictions
        # This eliminates code duplication and follows sklearn's pattern
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def predict_proba(self, X: pd.DataFrame | np.ndarray, check_input=False) -> np.ndarray:
        """
        Predict class probabilities for the given input data.

        Parameters:
            X (pd.DataFrame | np.ndarray): The input features.
            check_input (bool): Whether to check the input features. MATCHING THE BASE BUT NOT USING IT.

        Returns:
            np.ndarray: The predicted class probabilities.
        """
        check_is_fitted(self)

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=list(self.feature_names_in_))
        
        assert np.array_equal(self.feature_names_in_, X.columns), "Feature names don't match training data"
        
        decision_paths = self.base_sklearn_tree_model.decision_path(X)
        
        def get_prediction_tree(test_index: int) -> DecisionTreeClassifier:
            for replaced_node in self.replaced_subtrees:
                if decision_paths[test_index, replaced_node.get_index()]:
                    return self.replaced_subtrees[replaced_node]
            return self.base_sklearn_tree_model
        
        probabilities = []
        for i in range(len(X)):
            prediction_tree = get_prediction_tree(i)
            
            proba = prediction_tree.predict_proba(X.iloc[[i]])
            # Flatten the probability if it's a 2D array with one row
            if isinstance(proba, np.ndarray) and proba.shape[0] == 1:
                proba = proba[0]
            probabilities.append(proba)
        
        return np.array(probabilities)
    
    def predict_log_proba(self, X: pd.DataFrame | np.ndarray, check_input=False) -> np.ndarray:
        """
        Predict class log-probabilities for the given input data.

        Parameters:
            X (pd.DataFrame | np.ndarray): The input features.
            check_input (bool): Whether to check the input features. MATCHING THE BASE BUT NOT USING IT.

        Returns:
            np.ndarray: The predicted class log-probabilities.
        """
        return np.log(self.predict_proba(X))
        