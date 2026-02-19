from collections import defaultdict
import pandas as pd
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Callable
import os

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import DAPCROP.Constants as constants
from DAPCROP.ModelMapping.MappedDecisionTree import MappedDecisionTree
from DAPCROP.ModelMapping.MappedForest import MappedForest

from DAPCROP.Diagnosers import *

from .AFixer import AFixer, ATreeFixer

def _fix_estimator_worker(estimator_index: int,
                          mapped_estimator: MappedDecisionTree,
                          base_fixer_creator: Callable[[MappedDecisionTree, int], ATreeFixer]
                          ) -> DecisionTreeClassifier:
    """
    Worker function to fix a single estimator in parallel.
    
    Args:
        estimator_index: Index of the estimator in the forest
        mapped_estimator: The mapped decision tree estimator
        base_fixer_creator: Callable that creates a base fixer instance
    
    Returns:
        DecisionTreeClassifier: The fixed decision tree estimator
    """
    fixer: ATreeFixer = base_fixer_creator(mapped_estimator, estimator_index)
    return fixer.fix_model()

class ForestFixerWrapper(AFixer):
    alias = "forest_fixer_wrapper"

    mapped_model: MappedForest

    def __init__(self,
                 base_fixer_class: type[ATreeFixer],
                 mapped_model: MappedForest,
                 X: pd.DataFrame,
                 y: pd.Series,
                 faulty_nodes_indices: list[int],
                 X_prior: pd.DataFrame = None,
                 y_prior: pd.Series = None,
                 sklearn_model: RandomForestClassifier | AdaBoostClassifier = None,
                 **kwargs: dict
    ):
        super().__init__(mapped_model, X, y, faulty_nodes_indices, X_prior, y_prior, sklearn_model)
        self.base_fixer_class: type[ATreeFixer] = base_fixer_class
        self.base_fixer_parameters = kwargs
        self.faulty_estimators: dict[int, list[int]] = self.get_faulty_estimators()

    def get_faulty_estimators(self) -> dict[int, list[int]]:
        """
        Get the faulty estimators and their faulty nodes.
        
        Returns:
            dict[int, list[int]]: A mapping from estimator index to list of faulty node indices in that estimator.
        """
        faulty_estimators: dict[int, list[int]] = defaultdict(list)
        for faulty_node_index in self.faulty_nodes_indices:
            estimator_index, sklearn_node_index = self.mapped_model.component_estimator_map[faulty_node_index]
            faulty_estimators[estimator_index].append(sklearn_node_index)
        assert list(faulty_estimators.keys()) == sorted(faulty_estimators.keys())
        return faulty_estimators

    def base_fixer_creator(self, mapped_estimator: MappedDecisionTree, faulty_estimator_index: int) -> ATreeFixer:
        """
        Create a base fixer instance for the given mapped estimator.
        
        Args:
            mapped_estimator: The mapped decision tree estimator
            faulty_estimator_index: Index of the faulty estimator
            
        Returns:
            ATreeFixer: The configured fixer instance
        """
        return self.base_fixer_class(mapped_estimator, self.X, self.y, self.faulty_estimators[faulty_estimator_index], self.X_prior, self.y_prior, **self.base_fixer_parameters)
    
    def fix_model(self) -> RandomForestClassifier:
        """
        Fix the model using parallel processing.

        Returns:
            RandomForestClassifier | AdaBoostClassifier: The fixed ensemble model.
        """
        self.fixed_model: RandomForestClassifier | AdaBoostClassifier = deepcopy(self.sklearn_model)
        
        fixing_tasks = [(estimator_index, (estimator_index, mapped_estimator, self.base_fixer_creator))
                        for estimator_index, mapped_estimator in enumerate(self.mapped_model.mapped_estimators)
                        if estimator_index in self.faulty_estimators]
        
        fixed_estimators: list[tuple[int, DecisionTreeClassifier]]
        if constants.DISTRIBUTE_FIXING_COMPUTATION:
            with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, len(self.faulty_estimators))) as executor:
                futures = [(estimator_index, executor.submit(_fix_estimator_worker, *task)) for (estimator_index, task) in fixing_tasks]
                fixed_estimators = [(estimator_index, future.result()) for (estimator_index, future) in futures]
        else:
            fixed_estimators = [(estimator_index, _fix_estimator_worker(*task)) for (estimator_index, task) in fixing_tasks]
        
        for fixed_estimator_index, fixed_estimator in fixed_estimators:
            self.fixed_model.estimators_[fixed_estimator_index] = fixed_estimator

        return self.fixed_model