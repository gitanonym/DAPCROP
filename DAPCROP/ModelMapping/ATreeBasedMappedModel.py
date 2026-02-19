from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.base import ClassifierMixin

from .TreeNodeComponent import TreeNodeComponent

class ATreeBasedMappedModel(ABC):
    
    model: ClassifierMixin

    X: pd.DataFrame
    y: pd.Series
    data_feature_types: dict[str, str]
    model_used_features: set[str]
    
    components_map: dict[int, TreeNodeComponent]

    @abstractmethod
    def map_model(self):
        """
        Map the underlying model to its components; Updates the components_map attribute.
        """
        # sort the components by their indices
        self.components_map = dict(sorted(self.components_map.items()))

    @abstractmethod
    def get_node_indicator(self, X: pd.DataFrame) -> csr_matrix:
        """
        Get the node indicator matrix for the data.

        Parameters:
        X (DataFrame): The data.

        Returns:
        csr_matrix: The node indicator matrix.
        """
        pass
            
    @abstractmethod
    def get_model_representation(self) -> str:
        """
        Get the tree representation of the mapped model.

        Returns:
        str: The tree representation.
        """
        pass
    
    def __init__(self,
                 model: ClassifierMixin,
                 X: pd.DataFrame,
                 y: pd.Series,
                 feature_types: dict[str, str]
    ):
        """
        Initialize the mapped model.

        Parameters:
        model: The underlying model.
        X (DataFrame): The data.
        y (Series): The target column.
        feature_types (dict): The feature types.
        """
        self.model = model
        
        self.X = X
        self.y = y
        self.data_feature_types = feature_types
        
        self.map_model()
        
        self.update_model_statistics(X, y)

    def update_model_statistics(self, X: pd.DataFrame, y: pd.Series):
        """
        Update the model statistics based on the data.

        Parameters:
        X (DataFrame): The data.
        y (Series): The target column.
        """
        self.model_used_features = set(map(lambda component: component.feature, filter(TreeNodeComponent.is_internal, self)))

    def get_indices(self) -> Iterable[int]:
        """
        Get the indices of the mapped components.

        Returns:
        Iterable[int]: The indices of the mapped components.
        """
        return self.components_map.keys()
    
    def __getitem__(self, component_index: int) -> TreeNodeComponent:
        """
        Get the mapped component by its index.

        Parameters:
        key (tuple): contains:
            index (int): The component index.
            index_type (constants.NodeIndexType): The index type.

        Returns:
        TreeNodeComponent: The mapped component.
        """
        return self.components_map[component_index]

    def __iter__(self) -> Iterable[TreeNodeComponent]:
        """ Iterate over the components. """
        return iter(self.components_map.values())

    def __len__(self) -> int:
        """ Get the number of components in the model. """
        return len(self.components_map)
    
    def __repr__(self) -> str:
        """ Get the string representation of the model. """
        return self.get_model_representation()
    