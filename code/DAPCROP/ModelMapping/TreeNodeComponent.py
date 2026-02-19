from typing import Callable
from collections import Counter
from functools import reduce

import pandas as pd

from DAPCROP import Constants as constants
from DAPCROP.DataSynthesizer import DataSynthesizer

class TreeNodeComponent:
    # Indexing
    component_index: int # The model's index of the node. (in the sklearn tree)
    
    # Tree structure
    parent: 'TreeNodeComponent' # The parent node
    left_child: 'TreeNodeComponent' # The left child node
    right_child: 'TreeNodeComponent' # The right child node
    depth: int 
    
    # Node data properties
    feature: str # The feature
    feature_type: constants.FeatureType # The feature type
    feature_average_value: float # The average value of the feature (for the reached data)
    threshold: float # The threshold
    class_name: str # The class name
    
    # Data statistics
    reached_samples_count: int # The number of samples that reached the node
    confidence: float # The confidence of the node (correct classifications / reached samples)
    data_synthesizer: DataSynthesizer # Data synthesizer for the node

    # Path
    conditions_path: list[Callable[[pd.DataFrame], pd.Series]] # The conditions path from the root to the node

    def __init__(self, 
                 component_index: int,
                 parent: 'TreeNodeComponent' = None
    ):
        """
        Initialize the TreeNodeComponent.

        Parameters:
            component_index (int): The index of the node (in the sklearn tree).
            parent (TreeNodeComponent): The parent node.
            left_child (TreeNodeComponent): The left child node.
            right_child (TreeNodeComponent): The right child node.
            feature (str): The feature.
            threshold (float): The threshold.
            class_name (str): The class name.
            spectra_index (int): The index of the node (in the spectra matrix).
        """
        self.component_index: int = component_index
        self.parent = parent
        self.update_children(None, None)
        self.depth = 0 if parent is None else parent.depth + 1

        self.feature = None
        self.feature_type = None
        self.feature_average_value = None
        self.threshold = None
        self.class_name = None

    def get_index(self) -> int:
        """
        Get the index of the node.

        Parameters:
            index_type (constants.NodeIndexType): The index type.
        Returns:
            int: The index of the node.
        """
        return self.component_index
    
    def update_children(self, 
                        left_child: 'TreeNodeComponent', 
                        right_child: 'TreeNodeComponent'
    ) -> None:
        """
        Update the children of the node.

        Parameters:
            left_child (TreeNodeComponent): The left child node.
            right_child (TreeNodeComponent): The right child node.
        """
        # A node can be either a terminal node (two children) or a non-terminal node (a leaf - no children)
        assert (left_child is None) == (right_child is None)
        self.left_child = left_child
        self.right_child = right_child

    def is_terminal(self) -> bool:
        """
        Check if the node is a terminal node.
        
        Returns:
            bool: True if the node is a terminal node, False otherwise.
        """
        return self.left_child is None
    
    def is_internal(self) -> bool:
        """
        Check if the node is an internal node.
        
        Returns:
            bool: True if the node is an internal node, False otherwise.
        """
        return not self.is_terminal()
    
    def is_left_child(self) -> bool:
        """
        Check if the node is a left child.
        
        Returns:
            bool: True if the node is a left child, False otherwise.
        """
        assert self.parent is not None, "A root is not a child of any node"
        return self.parent.left_child == self
    
    def is_right_child(self) -> bool:
        """
        Check if the node is a right child.
        
        Returns:
            bool: True if the node is a right child, False otherwise.
        """
        return not self.is_left_child()
    
    def get_sibling(self) -> 'TreeNodeComponent':
        """
        Get the sibling of the node.
        
        Returns:
            TreeNodeComponent: The sibling of the node.
        """
        if self.parent is None:
            return None
        return self.parent.left_child if self.is_right_child() else self.parent.right_child
    
    def is_successor_of(self,
                        other: 'TreeNodeComponent'
    ) -> bool:
        """
        Check if the node is a successor of another node.

        Parameters:
            other (TreeNodeComponent): The other node.
        Returns:
            bool: True if the node is a successor of the other node, False otherwise.
        """
        assert isinstance(other, TreeNodeComponent), "other must be a TreeNodeComponent"
        if self == other:
            return True
        if self.is_terminal():
            return False
        return self.left_child.is_successor_of(other) or self.right_child.is_successor_of(other)
    
    def is_ancestor_of(self,
                       other: 'TreeNodeComponent'
    ) -> bool:
        """
        Check if the node is a ancestor of another node.

        Parameters:
            other (TreeNodeComponent): The other node.
        Returns:
            bool: True if the node is a ancestor of the other node, False otherwise.
        """
        return other.is_successor_of(self)
    
    def are_connected(self,
                      other: 'TreeNodeComponent'
    ) -> bool:
        """
        Check if the node is connected to another node.

        Parameters:
            other (TreeNodeComponent): The other node.
        Returns:
            bool: True if the node is connected to the other node, False otherwise.
        """
        return self.is_successor_of(other) or self.is_ancestor_of(other)
    
    def get_all_leaves(self) -> list['TreeNodeComponent']:
        """
        Get all leaves under the node.

        Returns:
            list[TreeNodeComponent]: The leaves under the node.
        """
        if self.is_terminal():
            return [self]
        return self.left_child.get_all_leaves() + self.right_child.get_all_leaves()
    
    def current_condition(self, X: pd.DataFrame) -> pd.Series:
        column: pd.Series = X[self.parent.feature]
        threshold: float = self.parent.threshold
        return column <= threshold if self.is_left_child() else column > threshold
    
    def update_condition(self,
                         recursive: bool = True
    ) -> None:
        """
        Update the conditions path of the node.
        
        The conditions path is the a filter function that filters the data that reached the node.

        Parameters:
            recursive (bool): Whether to update the conditions path of the children nodes.
            """
        if self.parent is None:
            self.conditions_path = []
        else:
            self.conditions_path = self.parent.conditions_path + [self.current_condition]
        if self.is_terminal():
            return
        if recursive:
            self.left_child.update_condition(recursive=recursive)
            self.right_child.update_condition(recursive=recursive)
    
    def get_data_reached_node(self,
                              X: pd.DataFrame,
                              y: pd.Series = None,
                              allow_empty: bool = True
        ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
        """
        Filter the data that reached the node.

        Parameters:
            X (DataFrame): The data.
            y (Series): The target column.

        Returns:
            pd.DataFrame | tuple[pd.DataFrame, pd.Series]: The data that reached the node.
        """
        condition: Callable[[pd.DataFrame], pd.Series]

        for condition in self.conditions_path:
            indices_filter = condition(X)
            next_X, next_y = X[indices_filter], None
            if y is not None:
                next_y = y[indices_filter]
                assert len(next_X) == len(next_y), "unmatched X and y lengths after filtering data reached node"
            if not indices_filter.any():
                if allow_empty:
                    X, y = next_X, next_y
                break
        return X if y is None else (X, y)
    
    def update_node_data_attributes(self, 
                                    X: pd.DataFrame,
                                    y: pd.Series,
                                    feature_types: dict[str, str] = None
        ) -> None:
        """
        Update the average feature value of the node.
        The average is calculated on the data that reached the node.

        Parameters:
            X (pd.DataFrame): The data.
            y (pd.Series): The target column
        """
        # Get the data that reached the node
        X, y = self.get_data_reached_node(X, y)
        self.reached_samples_count = len(X)
        if self.feature_type == constants.FeatureType.Numeric:
            self.feature_average_value = X[self.feature].mean()
        if not self.is_terminal():
            # update the descendant stats
            self.left_child.update_node_data_attributes(X, y, feature_types)
            self.right_child.update_node_data_attributes(X, y, feature_types)
        if y is not None:
            # count correct classifications
            if self.is_terminal():
                self.correct_classifications_count = (y == self.class_name).sum()
                self.misclassifications_count = (y != self.class_name).sum()
                if feature_types is not None:
                    self.data_synthesizer = DataSynthesizer(X, y, feature_types)
            else:
                self.correct_classifications_count = self.left_child.correct_classifications_count + self.right_child.correct_classifications_count
                self.misclassifications_count = self.left_child.misclassifications_count + self.right_child.misclassifications_count
            self.confidence = self.correct_classifications_count / self.reached_samples_count if self.reached_samples_count > 0 else 0
    
    def synthesize_data_reached_node(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Synthesize data that reached the node.
        """
        X, y = pd.DataFrame(), pd.Series()
        for leaf in self.get_all_leaves():
            synthesized_X, synthesized_y = leaf.data_synthesizer.synthesize()
            X = pd.concat([X, synthesized_X], ignore_index=True)
            y = pd.concat([y, synthesized_y], ignore_index=True)
        
        return X, y
    
    def get_node_class_count(self) -> dict[int, int]:
        return reduce(lambda x, y: x + Counter(y.data_synthesizer.class_counts), 
                      self.get_all_leaves(), 
                      Counter()
                      )
    
    def __eq__(self, other) -> bool:
        if isinstance(other, TreeNodeComponent):
            return self.get_index() == other.get_index()
        if isinstance(other, int):
            return self.get_index() == other
        return False
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    def __repr__(self) -> str:
        """
        Get the string representation of the node.
        format: Node(c:<component_index>; s:<spectra_index>)
        """
        return f"Node({self.get_index()})"