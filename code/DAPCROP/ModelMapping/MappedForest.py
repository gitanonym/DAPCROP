from copy import deepcopy

import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from .ATreeBasedMappedModel import ATreeBasedMappedModel
from .TreeNodeComponent import TreeNodeComponent
from .MappedDecisionTree import MappedDecisionTree

class MappedForest(ATreeBasedMappedModel):
    mapped_estimators: list[MappedDecisionTree]
    component_estimator_map: dict[int, tuple[int, int]] # component index to (estimator index, sklearn index in estimator)
    inverse_component_estimator_map: dict[tuple[int, int], int] # (estimator index, sklearn index in estimator) to component index

    def __init__(self, 
                 model: RandomForestClassifier | AdaBoostClassifier,
                 feature_types: dict[str, str],
                 X: pd.DataFrame = None,
                 y: pd.Series = None
    ):
        """
        Initialize the MappedForest.
        
        Parameters:
            model (RandomForestClassifier | AdaBoostClassifier): The sklearn ensemble model.
            X (DataFrame): The data.
            y (Series): The target column.
        """
        self.mapped_estimators: list[MappedDecisionTree] = [
            MappedDecisionTree(model=decision_tree,
                               feature_types=feature_types,
                               X=X,
                               y=y,
                               prune=False
                               )
            for decision_tree in model.estimators_
        ]
        
        super().__init__(model, X, y, feature_types)


    def map_model(self, 
    ) -> dict[int, 'MappedDecisionTree.DecisionTreeNode']:
        self.components_map = {}
        self.component_estimator_map = {}
        self.inverse_component_estimator_map = {}
        current_component_index = 0
        mapped_estimator: MappedDecisionTree
        component: TreeNodeComponent
        for estimator_index, mapped_estimator in enumerate(self.mapped_estimators):
            for component in mapped_estimator:
                previous_component_index = component.get_index()
                component_local_identity = (estimator_index, previous_component_index)
                self.component_estimator_map[current_component_index] = component_local_identity
                component = deepcopy(component)
                component.component_index = current_component_index
                self.components_map[current_component_index] = component
                self.inverse_component_estimator_map[component_local_identity] = current_component_index
                current_component_index += 1
                
    def get_node_indicator(self, X: pd.DataFrame) -> csr_matrix:
        return hstack([mapped_estimator.get_node_indicator(X) for mapped_estimator in self.mapped_estimators],
                      format='csr')

    def get_model_representation(self) -> str:
        for i, tree in enumerate(self.mapped_estimators):
            tree_repr = f"Tree {i}:\n{tree.__repr__()}\n\n"
        return tree_repr