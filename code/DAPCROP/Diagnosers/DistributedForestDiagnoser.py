import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import Callable
import os

from sklearn.metrics import accuracy_score

from DAPCROP import Constants as constants

from DAPCROP.ModelMapping.MappedForest import MappedForest
from DAPCROP.ModelMapping.MappedDecisionTree import MappedDecisionTree

from .ADiagnoser import *
from .Oracle import Oracle
from .SFLDT import SFLDT

def _diagnose_estimator_worker(estimator_index: int,
                               mapped_estimator: MappedDecisionTree,
                               base_diagnoser_creator: Callable[[MappedDecisionTree], ADiagnoser],
                               convert_indices_to_global_function: Callable[[int, list[int]], list[int]]
                               ) -> list[tuple[list[int], float]]:
    """
    Worker function to diagnose a single estimator in parallel.
    
    This function handles the UnaffectedModelError internally and converts
    local indices to global indices before returning.
    
    Args:
        estimator_index: Index of the estimator in the forest
        mapped_estimator: The mapped decision tree estimator
        base_diagnoser_creator: Callable that creates a base diagnoser instance
        convert_indices_to_global_function: Callable that converts local indices to global indices
    
    Returns:
        List of tuples containing (global_diagnosis, rank) or empty list if unaffected
    """

    estimator_diagnoses = []
    try:
        diagnoser = base_diagnoser_creator(mapped_estimator)
        estimator_diagnoses = diagnoser.get_diagnoses(retrieve_ranks=True)
    except SFLDT.UnaffectedModelException:
        pass
    finally:
        return [
            (convert_indices_to_global_function(estimator_index, estimator_diagnosis), estimator_diagnosis_rank)
            for estimator_diagnosis, estimator_diagnosis_rank in estimator_diagnoses
        ]

class DistributedForestDiagnoser(ADiagnoser):
    """
    The distributed forest diagnoser with parallel processing support.
    
    Using a base diagnoser and run it for each tree separately in parallel threads.
    At the end, aggregate the results from all trees.
    
    The parallelization uses ProcessPoolExecutor to run diagnosis on multiple trees
    simultaneously, with each worker handling exceptions internally and returning
    global indices directly.
    """
    mapped_model: MappedForest

    def __init__(self,
                 base_diagnoser_class: type[ADiagnoser],
                 mapped_model: MappedForest,
                 X: pd.DataFrame,
                 y: pd.Series,
                 **kwargs: object
    ):
        """
        Initialize the DistributedForestDiagnoser with parallel processing support.
        
        Parameters:
            base_diagnoser_class (type[ADiagnoser]): The base diagnoser class to use for each tree.
            mapped_model (MappedForest): The mapped random forest model.
            X (pd.DataFrame): The input data.
            y (pd.Series): The target data.
            max_workers (Optional[int]): Maximum number of worker threads for parallel execution.
                If None, defaults to min(cpu_count, number_of_estimators).
            **kwargs: Additional parameters to pass to the base diagnoser.
        """
        super().__init__(mapped_model, X, y)
        self.base_estimator_class = base_diagnoser_class
        self.base_diagnoser_parameters = kwargs

    def base_diagnoser_creator(self, mapped_estimator: MappedDecisionTree) -> ADiagnoser:
        return self.base_estimator_class(mapped_estimator, self.X_after, self.y_after, **self.base_diagnoser_parameters)

    def convert_indices_to_global(self,
                                  estimator_index: int,
                                  local_diagnosis: list[int]
    ) -> list[int]:
        """
        Convert local diagnosis indices to global indices.

        Parameters:
            local_diagnosis (list[int]): The local diagnosis indices.

        Returns:
            list[int]: The global diagnosis indices.
        """
        inverse_map_keys = map(lambda local_index: (estimator_index, local_index), local_diagnosis)
        return list(map(self.mapped_model.inverse_component_estimator_map.get, inverse_map_keys))

    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
     ) -> list[list[int]] | list[tuple[list[int], float]]:
        """
        Get the diagnosis.
        Achieved by running the base diagnoser on each tree in parallel and aggregate the results.
        """
        self.diagnoses: list[tuple[list[int], float]] = []

        # diagnose only buggy estimators
        is_estimator_buggy: Callable[[MappedDecisionTree], bool] = lambda mapped_estimator: mapped_estimator.model.best_accuracy - accuracy_score(self.y_after, mapped_estimator.model.predict(self.X_after)) >= constants.MINIMUM_DRIFT_ACCURACY_DROP
        buggy_estimators = filter((lambda estimator_index_mapped_pair: is_estimator_buggy(estimator_index_mapped_pair[1])), enumerate(self.mapped_model.mapped_estimators)) # needs to include the estimator index to maintain the global conversion
        diagnosis_tasks = [(estimator_index, mapped_estimator, self.base_diagnoser_creator, self.convert_indices_to_global)
                           for estimator_index, mapped_estimator in buggy_estimators
                           ]
        
        all_estimator_diagnoses: list[list[tuple[list[int], float]]]
        if constants.DISTRIBUTE_DIAGNOSES_COMPUTATION and self.base_estimator_class is not Oracle:
            with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, len(self.mapped_model.mapped_estimators))) as executor:
                futures = [executor.submit(_diagnose_estimator_worker, *task) for task in diagnosis_tasks]
                all_estimator_diagnoses = [future.result() for future in futures]
        else:
            all_estimator_diagnoses = [_diagnose_estimator_worker(*task) for task in diagnosis_tasks]
        
        for estimator_diagnoses in all_estimator_diagnoses:
            for diagnosis_index, (global_diagnosis, estimator_diagnosis_rank) in enumerate(estimator_diagnoses):
                if diagnosis_index == len(self.diagnoses):
                    self.diagnoses.append(([], 0.0))                
                current_diagnosis, current_rank = self.diagnoses[diagnosis_index]
                current_diagnosis.extend(global_diagnosis)
                self.diagnoses[diagnosis_index] = (current_diagnosis, current_rank + estimator_diagnosis_rank)
            
        return super().get_diagnoses(retrieve_ranks)