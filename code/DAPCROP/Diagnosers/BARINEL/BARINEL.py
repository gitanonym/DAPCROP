import numpy as np
import pandas as pd

from DAPCROP import Constants as constants
from DAPCROP.ModelMapping.ATreeBasedMappedModel import ATreeBasedMappedModel

from DAPCROP.Diagnosers.SFLDT import SFLDT

from .utils import *

def get_barinel_diagnoses(spectra: np.ndarray,
                          error_vector: np.ndarray,
                          components_prior_probabilities: np.ndarray = None,
                          error_threshold: float = 1,
                          candidates_spectra: np.ndarray = None
 ) -> list[tuple[list[int], float]]:
    """
    Perform the Barinel diagnosis algorithm on the given spectrum.

    Parameters:
    spectra (ndarray): The spectrum.
    error_vector (ndarray): The error vector.
    components_prior_probabilities (ndarray): The components prior probabilities.
    error_threshold (float): The threshold for the error vector.
    candidates_spectra (ndarray): The candidates spectra. If not provided, the spectra will be used.

    Returns:
    list[tuple[list[int], float]]: The diagnoses with their corresponding ranks.
    """
    if (error_vector >= error_threshold).sum() == 0:
        if not constants.SFLDT_ALLOW_UNAFFECTED_MODELS:
            raise SFLDT.UnaffectedModelException(f"No path with error above the threshold {error_threshold} (average: {error_vector.mean()}). The largest error is {max(error_vector)}")
        return []
    assert candidates_spectra is None or spectra.shape == candidates_spectra.shape, f"The candidates spectra must have the same shape as the spectra: {spectra.shape} != {candidates_spectra.shape}"
    
    spectra = np.concatenate((spectra.T, error_vector[:, None]), axis=1)
    
    if candidates_spectra is None:
        candidates_spectra = spectra
    else:
        candidates_spectra = np.concatenate((candidates_spectra.T, error_vector[:, None]), axis=1)
    
    diagnoses = get_candidate_diagnoses(candidates_spectra.tolist(), error_threshold=error_threshold)
    assert diagnoses, "No candidate diagnoses found"
    
    diagnoses = list(map(np.array, diagnoses))
    diagnoses = rank_diagnoses(spectra, diagnoses, components_prior_probabilities)
    diagnoses = [(diagnosis[0].tolist(), diagnosis[1]) for diagnosis in diagnoses]

    return diagnoses

class BARINEL(SFLDT):
    """
    The BARINEL diagnoser.

    This diagnoser uses the BARINEL algorithm to diagnose the drift.
    It is a subclass of the SFLDT diagnoser. It uses the spectra matrix to calculate the diagnoses.
    It returns multiple fault diagnoses.
    """

    def __init__(self, 
                 mapped_model: ATreeBasedMappedModel,
                 X: pd.DataFrame,
                 y: pd.Series,
                 **kwargs: object
    ):
        """
        Initialize the BARINEL diagnoser.
        
        Parameters:
        mapped_model (ATreeBasedMappedModel): The mapped model.
        X (DataFrame): The data.
        y (Series): The target column.
        kwargs (object): All the SFLDT parameters
        """
        self.components_prior_probabilities = None
        self.threshold = 1
        super().__init__(mapped_model, X, y, **kwargs)

    def update_threshold(self
    ) -> None:
        """
        Update the error threshold that determine which test is considered fail in the candidation process.
        """
        error_average, error_std = self.error_vector.mean(), self.error_vector.std()
        self.threshold = error_average + constants.BARINEL_THRESHOLD_ABOVE_STD_RATE * error_std
        self.threshold = min(self.threshold, max(self.error_vector)) # decrease to catch at least one error

    def update_error_vector_to_fuzzy(self
    ) -> None:
        """
        After using the parent class (SFLDT)'s update_fuzzy_error method, update the threshold accordingly.
        """
        super().update_error_vector_to_fuzzy()
        self.update_threshold()

    def add_target_to_feature_components(self,
                                         target_name: str = "target"
    ) -> None:
        if constants.BARINEL_ADD_TARGET_TO_FEATURE_COMPONENTS:
            super().add_target_to_feature_components(target_name)
        pass

    def combine_stat_diagnoses(self
     ) -> None:
        """
        Combine stat diagnoses with the SFLDT diagnoses.
        the combination is done by using the stat ranks of the components as the prior probabilities of the components.
        """
        stat_diagnoses = self.load_stat_diagnoses()
        stat_diagnoses.sort(key=lambda diagnosis: self.convert_node_index_to_spectra_index(diagnosis[0][0])) # sort by the components order (to match the spectra indices)
        nodes_stat_rank_vector = np.array([diagnosis[1] for diagnosis in stat_diagnoses])
        if self.group_feature_nodes:
            # get for each feature the average of the stat ranks of the components
            self.components_prior_probabilities = np.array([np.mean(nodes_stat_rank_vector[self.feature_index_to_node_indices_dict[feature_index]]) for feature_index in range(self.components_count)])
        else:
            self.components_prior_probabilities = nodes_stat_rank_vector

    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
                      retrieve_spectra_indices: bool = False
     ) -> list[list[int]] | list[tuple[list[int], float]]:
        """
        Get the diagnosis of the nodes.
        The diagnosis consists the nodes ordered.

        Parameters:
        retrieve_ranks (bool): Whether to return the diagnosis ranks.
        retrieve_spectra_indices (bool): Whether to return the spectra indices or the node indices.
        components_prior_probabilities (ndarray): The components prior probabilities.
        threshold (float): The threshold for the error vector

        Returns:
        list[int] | list[tuple[int, float]]: The diagnosis. If retrieve_ranks is True, the diagnosis will be a list of tuples,
          where the first element contains the indices of the faulty nodes and the second is the similarity rank.
        """
        if self.diagnoses is None:
            if self.stat:
                self.combine_stat_diagnoses()
            self.diagnoses = get_barinel_diagnoses(spectra=self.spectra, error_vector=self.error_vector, components_prior_probabilities=self.components_prior_probabilities, error_threshold=self.threshold)
        return super().get_diagnoses(retrieve_ranks, retrieve_spectra_indices)
        
