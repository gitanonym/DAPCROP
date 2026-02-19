from abc import ABC, abstractmethod
import pandas as pd
from copy import deepcopy

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from DAPCROP.ModelMapping.ATreeBasedMappedModel import ATreeBasedMappedModel
from DAPCROP.ModelMapping.TreeNodeComponent import TreeNodeComponent
from DAPCROP.ModelMapping.MappedDecisionTree import MappedDecisionTree

from DAPCROP.Diagnosers import *

class AFixer(ABC):
    alias = None
    def __init__(self, 
                 mapped_model: ATreeBasedMappedModel,
                 X: pd.DataFrame,
                 y: pd.Series,
                 faulty_nodes_indices: list[int],
                 X_prior: pd.DataFrame = None,
                 y_prior: pd.Series = None,
                 sklearn_model: ClassifierMixin = None,
                 sample_weight: pd.Series = None
    ):
        """
        Initialize the Fixer.
        
        Parameters:
        mapped_model (ATreeBasedMappedModel): The mapped model.
        X (DataFrame): The data.
        y (Series): The target column.
        faulty_nodes_indices (list[int]): The indices of the faulty nodes.
        sklearn_model (ClassifierMixin, optional): The sklearn model. Defaults to None (Taken from mapped_model).
        sample_weight (Series, optional): Per-sample weights aligned with X. Defaults to None (uniform weights).
        """
        assert self.alias is not None, "Alias must be set to a fixer class"

        self.mapped_model = deepcopy(mapped_model)
        self.feature_types = mapped_model.data_feature_types
        self.X = X
        self.y = y
        self.faulty_nodes_indices = faulty_nodes_indices
        self.fixed_model: ClassifierMixin = None

        self.X_prior = X_prior
        self.y_prior = y_prior
        self.sample_weight = sample_weight
        
        self.sklearn_model = sklearn_model if sklearn_model else mapped_model.model
        
    def _filter_data_reached_fault(self,
                                  faulty_node_index: int
        ) -> pd.DataFrame:
        """
        Filter the data that reached the faulty nodes.

        Parameters:
            faulty_nodes_count (int): The number of faulty nodes.

        Returns:
            DataFrame: The data that reached the faulty nodes.
        """
        faulty_node = self.mapped_model[faulty_node_index]
        return faulty_node.get_data_reached_node(self.X, self.y, allow_empty=False)
    
    @abstractmethod
    def fix_model(self) -> ClassifierMixin:
        """
        Fix the model.

        Returns:
            ClassifierMixin: The fixed model.
        """
        assert self.fixed_model, "The model wasn't fixed yet"

        return self.fixed_model
    
class ATreeFixer(AFixer):
    """
    Abstract class for tree-based fixers.
    """
    def __init__(self, 
                 mapped_model: MappedDecisionTree,
                 X: pd.DataFrame,
                 y: pd.Series,
                 faulty_nodes_indices: list[int],
                 X_prior: pd.DataFrame = None,
                 y_prior: pd.Series = None,
                 sklearn_model: DecisionTreeClassifier = None,
                 sample_weight: pd.Series = None
    ):
        super().__init__(mapped_model, X, y, faulty_nodes_indices, X_prior, y_prior, sklearn_model, sample_weight=sample_weight)
        self.faulty_nodes_indices.sort(key=lambda idx: mapped_model[idx].depth)