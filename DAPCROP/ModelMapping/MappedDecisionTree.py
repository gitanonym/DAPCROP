import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from DAPCROP import Constants as constants
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree._tree import TREE_LEAF

from .ATreeBasedMappedModel import ATreeBasedMappedModel
from .TreeNodeComponent import TreeNodeComponent

class MappedDecisionTree(ATreeBasedMappedModel):
    model: DecisionTreeClassifier

    root: TreeNodeComponent # The root node of the tree
    
    prune: bool # Whether to prune the tree upon mapping
    
    def __init__(self, 
                 model: DecisionTreeClassifier,
                 feature_types: dict[str, str],
                 X: pd.DataFrame = None,
                 y: pd.Series = None,
                 prune: bool = True
    ):
        """
        Initialize the MappedDecisionTree.
        
        Parameters:
            model (DecisionTreeClassifier): The sklearn decision tree.
            prune (bool): Whether to prune the tree.
            X (DataFrame): The data.
            y (Series): The target column.
        """
        self.prune = prune
        if constants.SYNTHESIZE_BY_BOOTSTRAP and hasattr(model, "sample_weight"):
            # Use only the trained indices
            assert X is not None and y is not None, "X and y must be provided to fit the model with sample weights"
            sample_weights = model.sample_weight
            # Create deterministic sampling based on integer sample weights
            train_indices = []
            for idx, weight in enumerate(sample_weights):
                # Add each index as many times as its weight (assuming integer weights)
                train_indices.extend([idx] * int(weight))
            X = X.iloc[train_indices].reset_index(drop=True)
            y = y.iloc[train_indices].reset_index(drop=True)
        super().__init__(model, X, y, feature_types)

    def update_model_statistics(self, X: pd.DataFrame, y: pd.Series):
        """
        Update the model statistics based on the data.

        Parameters:
        X (DataFrame): The data.
        y (Series): The target column.
        """
        super().update_model_statistics(X, y)
        
        self.root.update_condition()
        
        if X is not None:
            self.root.update_node_data_attributes(X, y, self.data_feature_types)

    def map_model(self):
        self.sk_features = self.model.tree_.feature
        self.sk_thresholds = self.model.tree_.threshold
        self.sk_children_left = self.model.tree_.children_left
        self.sk_children_right = self.model.tree_.children_right
        self.sk_values = self.model.tree_.value
        sk_class_names = self.model.classes_
        
        self.components_map = {}
        nodes_to_check = [TreeNodeComponent(0)]

        while len(nodes_to_check):
            current_node = nodes_to_check.pop(0)
            current_node.update_condition()
            current_index = current_node.get_index()
            self.components_map[current_index] = current_node
            left_child_index = self.sk_children_left[current_index]
            right_child_index = self.sk_children_right[current_index]

            if left_child_index == right_child_index:  # Leaf
                current_node_value = self.sk_values[current_index]
                class_name = np.argmax(current_node_value)
                class_name = sk_class_names[class_name]
                current_node.class_name = class_name
                continue
            
            current_node.threshold = self.sk_thresholds[current_index]
            feature_index = self.sk_features[current_index]
            current_node.feature = list(self.data_feature_types.keys())[feature_index]
            current_node.feature_type = self.data_feature_types[current_node.feature]
            right_child_index = self.sk_children_right[current_index]
            for child_index in (left_child_index, right_child_index):
                child_node = TreeNodeComponent(child_index, parent=current_node)
                nodes_to_check.append(child_node)
            current_node.update_children(*(nodes_to_check[-2: ]))

        self.root = self[0]
        
        if self.prune:
            self.prune_tree()
        
        super().map_model()
    
    def prune_sibling_leaves(self,
                             leaf1: TreeNodeComponent,
                             leaf2: TreeNodeComponent
     ) -> TreeNodeComponent:
        """
        Prune the sibling leaves.

        Parameters:
            leaf1 (TreeNodeComponent): The first leaf.
            leaf2 (TreeNodeComponent): The second leaf.

        Returns:
            TreeNodeComponent: The parent of the leaves.
        """
        self.components_map.pop(leaf1.get_index())
        self.components_map.pop(leaf2.get_index())
        parent = leaf1.parent
        parent.update_children(None, None)
        current_class = leaf1.class_name
        parent.feature, parent.feature_type, parent.threshold, parent.class_name = None, None, None, current_class
        # Adjust the tree
        parent_index = parent.get_index()
        self.model.tree_.children_left[parent_index] = TREE_LEAF
        self.model.tree_.children_right[parent_index] = TREE_LEAF
        self.model.tree_.feature[parent_index] = -2
        return parent
    
    def prune_leaf(self,
                   leaf_node: TreeNodeComponent
     ) -> TreeNodeComponent:
        """
        Prune a leaf.
        The prune is done by removing the leaf and replacing the parent with it's sibling.
    
        Parameters:
            leaf (TreeNodeComponent): The leaf to prune.
        """
        parent, sibling = leaf_node.parent, leaf_node.get_sibling()
        self.components_map.pop(leaf_node.get_index())
        self.components_map.pop(sibling.get_index())
        # Replace all parent's attributes with siblings'
        parent.update_children(sibling.left_child, sibling.right_child)
        parent.feature, parent.feature_type, parent.threshold, parent.class_name = sibling.feature, sibling.feature_type, sibling.threshold, sibling.class_name
        # Update the sklearn tree
        parent_index = parent.get_index()
        self.sk_children_left[parent_index] = sibling.left_child.get_index() if not sibling.is_terminal() else TREE_LEAF
        self.sk_children_right[parent_index] = sibling.right_child.get_index() if not sibling.is_terminal() else TREE_LEAF
        self.sk_features[parent_index] = self.sk_features[sibling.get_index()]
        self.sk_thresholds[parent_index] = sibling.threshold
        self.sk_values[parent_index] = self.sk_values[sibling.get_index()]
        if parent.is_terminal():
            return parent
    
    def prune_tree(self) -> None:
        leaf_nodes = [node for node in self.components_map.values() if node.is_terminal()]
        while len(leaf_nodes):
            current_leaf = leaf_nodes.pop(0)
            if current_leaf.get_index() not in self.components_map: # Already pruned
                continue
            sibling = current_leaf.get_sibling()
            if sibling is None: # Root
                continue
            if sibling.is_terminal() and current_leaf.class_name == sibling.class_name: # Sibling is a leaf with the same class
                # leaf_nodes = [leaf_node for leaf_node in leaf_nodes if leaf_node.get_index() != sibling.get_index()] # Remove sibling from the list
                if sibling in leaf_nodes:
                    leaf_nodes.remove(sibling)
                leaf_nodes += [self.prune_sibling_leaves(current_leaf, sibling)]
            elif hasattr(current_leaf, "reached_samples_count") and not current_leaf.reached_samples_count: # Redundant leaf
                new_leaf = self.prune_leaf(current_leaf)
                if new_leaf:
                    leaf_nodes += [new_leaf]            

    def get_node_indicator(self, X: pd.DataFrame) -> csr_matrix:
        # Source: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#decision-path
        node_indicator = self.model.tree_.decision_path(X.to_numpy(dtype="float32"))
        return node_indicator
    
    def get_model_representation(self) -> str:
        return export_text(self.model, feature_names=list(self.data_feature_types.keys()))
