import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr as pearson_correlation
from shap import TreeExplainer

from DAPCROP import Constants as constants
from DAPCROP.ModelMapping.TreeNodeComponent import TreeNodeComponent

from .ADiagnoser import *
from .STAT import STAT

class SFLDT(ADiagnoser):
    class UnaffectedModelException(Exception):
        """Exception raised when the model is unaffected by the drift."""
        pass

    def __init__(self, 
                 mapped_model: ATreeBasedMappedModel,
                 X: pd.DataFrame,
                 y: pd.Series,
                 combine_stat: bool = constants.DEFAULT_COMBINE_STAT,
                 group_feature_nodes: bool = constants.DEFAULT_GROUP_FEATURE_NODES,
                 aggregate_tests: bool = constants.DEFAULT_AGGREGATE_TESTS_BY_PATHS,
                 combine_prior_confidence: bool = constants.DEFAULT_COMBINE_PRIOR_CONFIDENCE,
                 use_shap_contribution: bool = constants.DEFAULT_USE_SHAP_CONTRIBUTION,
                 combine_components_depth: bool = constants.DEFAULT_COMBINE_COMPONENTS_DEPTH
    ):
        """
        Initialize the SFLDT diagnoser.
        
        Parameters:
        mapped_model (ATreeBasedMappedModel): The mapped model.
        X (DataFrame): The data.
        y (Series): The target column.
        combine_stat (bool): Whether to combine the diagnoses with the STAT diagnoser.
        use_shap_contribution (bool): Whether to use SHAP contributions for the participation.
        aggregate_tests (bool): Whether to aggregate tests based on the classification paths.
        group_feature_nodes (bool): Whether to use feature components.
        combine_prior_confidence (bool): Whether to combine the confidence of the tests in the error vector calculation.
        combine_components_depth (bool): whether to include the components depth in the components' participations.
        """
        super().__init__(mapped_model, X, y)
        
        self.components_count = len(mapped_model)
        self.tests_count = len(X)
        
        self.spectra = np.zeros((self.components_count, self.tests_count))
        self.error_vector = np.zeros(self.tests_count)
        
        # Components
        self.group_feature_nodes = group_feature_nodes
        self.use_shap_contribution = use_shap_contribution
        self.combine_components_depth = combine_components_depth
        self.is_participation_fuzzy = self.use_shap_contribution or combine_components_depth
        # Tests
        self.aggregate_tests = aggregate_tests
        self.combine_prior_confidence = combine_prior_confidence
        self.is_error_fuzzy = self.combine_prior_confidence or self.aggregate_tests
        self.path_tests_indices = defaultdict(list)

        self.spectra_map: dict[int, TreeNodeComponent]
        self.inverse_spectra_map: dict[TreeNodeComponent, int]
        self.get_component_spectra_map()
        
        try:
            self.fill_spectra_and_error_vector(X, y)
        except self.UnaffectedModelException as e:
            if not constants.SFLDT_ALLOW_UNAFFECTED_MODELS:
                raise e
        self.stat = STAT(mapped_model, X, y) if combine_stat else None

    def get_component_spectra_map(self):
        self.spectra_map, self.inverse_spectra_map = {}, {}
        for spectra_index, component in enumerate(self.mapped_model):
            self.spectra_map[spectra_index] = component
            self.inverse_spectra_map[component] = spectra_index
    
    def convert_node_index_to_spectra_index(self,
                                            index: int
    ) -> int:
        return self.inverse_spectra_map[self.mapped_model[index]]
    
    def convert_spectra_index_to_node_index(self,
                                            index: int
    ) -> int:
        return self.spectra_map[index].get_index()

    def shrink_spectra_based_on_paths(self,
                                      to_shrink_spectra: np.ndarray,
                                      paths_example_test_indices: list[int]
    ) -> np.ndarray:
        """
            Shrink the spectra based on the new tests.

            Parameters:
            to_shrink_spectra (ndarray): The spectra to shrink.
            paths_example_test_indices (list[int]): Because all tests belong to the same path have the same "participation vector", holding one test index from each one.

            Returns:
                The shrunk spectra.
        """
        to_shrink_spectra = to_shrink_spectra[:, paths_example_test_indices]
        assert to_shrink_spectra.shape == (self.components_count, self.tests_count)
        return to_shrink_spectra

    def aggregate_tests_by_paths(self
    ) -> None:
        """
        Merge the tests by their classification paths.
        Each test will be correspond to classification path in the tree (that has any nodes passed through).
        The error value will be the average error (the misclassification) of the classification path.
        """
        assert self.tests_count > len(self.path_tests_indices) or self.tests_count == sum(map(len, self.path_tests_indices.values())), f"Tests count should be changed after aggregation, but it is still the same: previous ({self.tests_count}) == new ({self.error_vector.shape[0]})"
        self.tests_count = len(self.path_tests_indices)
        paths_error_vector = np.zeros(self.tests_count)
        paths_first_test_indices = []
        for path_index, path_test_indices in enumerate(self.path_tests_indices.values()):
            paths_first_test_indices.append(path_test_indices[0])
            paths_error_vector[path_index] = self.error_vector[path_test_indices].mean()
        
        self.error_vector = paths_error_vector / paths_error_vector.sum()
        
        self.spectra = self.shrink_spectra_based_on_paths(self.spectra, paths_first_test_indices)

    def update_error_vector_to_fuzzy(self
    ) -> None:
        """
        Update all needed attributes to support the fuzzy error vector
        """
        if self.aggregate_tests:
            self.aggregate_tests_by_paths()

    def add_target_to_feature_components(self,
                                         target_name: str = "target"
    ) -> None:
        if any((self.is_participation_fuzzy, self.is_error_fuzzy)):
            return
        target_feature_index = len(self.feature_indices_dict)
        self.feature_indices_dict[target_name] = target_feature_index
        target_nodes = [node_spectra_index for node_spectra_index, node in self.spectra_map.items() if node.is_terminal()]
        self.feature_index_to_node_indices_dict[target_feature_index] = target_nodes

    def update_spectra_to_feature_components(self
    ) -> None:
        """
        Update the spectra matrix to be based on the features.
        Each feature will be represented by a single spectra index.
        The participation will be based on the average participation of all the nodes that represent the feature (which participated in the relevant test).
        """
        # Create feature to feature_index mapping (based on the order in the data)
        self.feature_indices_dict = {feature: feature_index
                                     for (feature_index, feature) in enumerate(
                                         filter(lambda feature: feature in self.mapped_model.model_used_features, self.X_after.columns)
                                         )}
        # Create feature_index to the corresponding nodes mapping
        self.feature_index_to_node_indices_dict = defaultdict(list)
        for node_spectra_index, node in self.spectra_map.items():
            if node.is_terminal():
                continue
            self.feature_index_to_node_indices_dict[self.feature_indices_dict[node.feature]].append(node_spectra_index)
        # add target
        self.add_target_to_feature_components()
        self.components_count = len(self.feature_indices_dict)
        features_spectra = np.zeros((self.components_count, self.tests_count))
        for feature_index, feature_nodes_spectra_indices in self.feature_index_to_node_indices_dict.items():
            current_feature_spectra = self.spectra[feature_nodes_spectra_indices, :]
            participations_mask = current_feature_spectra > 0
            participations_sums = (current_feature_spectra * participations_mask).sum(axis=0)
            participations_counts = participations_mask.sum(axis=0)
            features_spectra[feature_index] = np.divide(participations_sums, participations_counts, out=np.zeros_like(participations_sums, dtype=float), where=participations_counts != 0)
        self.spectra = np.nan_to_num(features_spectra)

    def combine_shap_contributions(self) -> None:
        """
        Update the participation matrix to use the shap contributions. For now only relevant for feature components.
        The participations are calculated as the weighted average of the SHAP values of the features. The weights are based on the predicted probabilities of the samples.
        """
        explainer = TreeExplainer(self.mapped_model.model)
        
        samples_predicted_probabilities = self.mapped_model.model.predict_proba(self.X_after)   # shape: (|tests|, |classes|)
        samples_predicted_probabilities = (samples_predicted_probabilities == samples_predicted_probabilities.max(axis=1, keepdims=True)).astype(int)
        samples_positive_shap_values = np.maximum(explainer.shap_values(self.X_after), 0)  # shape: (|tests|, |FEATURES!|, |classes)
        tree_features_locations = [column_index for column_index, feature in enumerate(self.X_after.columns) if feature in self.mapped_model.model_used_features]
        samples_positive_shap_values = samples_positive_shap_values[:, tree_features_locations, :]  # shape: (|tests|, |COMPONENTS!|, |classes)
        weighted_shap_values = samples_positive_shap_values * samples_predicted_probabilities[:, None, :]  # shape: (|tests|, |features=components|, |classes)
        
        fuzzy_spectra = weighted_shap_values.sum(axis=2).T
        assert fuzzy_spectra.shape == self.spectra.shape, f"The new fuzzy spectra's shape {fuzzy_spectra.shape} does not match the original spectra's shape {self.spectra.shape}"
        
        participations_sum = fuzzy_spectra.sum()
        if participations_sum in (0, fuzzy_spectra.size):
            # Constant value for all spectra - cannot use fuzzy participation
            self.is_participation_fuzzy = False
            return
  
        fuzzy_spectra /= participations_sum
        fuzzy_spectra[fuzzy_spectra == 0] = constants.EPSILON

        # Including the shap contributions only for the paths participations
        self.spectra = fuzzy_spectra * self.spectra
    
    def update_spectra_to_fuzzy(self
    ) -> None:
        """
        Update all needed attributes to support the fuzzy participation spectra
        """
        if self.use_shap_contribution:
            assert self.group_feature_nodes, "Cannot use SHAP contributions without feature components"
            self.combine_shap_contributions()

    def fill_spectra_and_error_vector(self, 
                                      X: pd.DataFrame, 
                                      y: pd.Series
    ) -> None:
        """
        Fill the spectra matrix and the error vector.

        Parameters:
        X (DataFrame): The data.
        y (Series): The target column.
        """
        # Counting how many terminals are there in each test.
        # This should be equal for all tests, thus will be using test 0 for example
        terminals_count = 0
        node_indicator = self.mapped_model.get_node_indicator(X)
        discrete_error_vector = (self.mapped_model.model.predict(X) != y.values).astype(int)
        for test_index in range(self.tests_count):
            participated_component_indices = node_indicator.indices[
                node_indicator.indptr[test_index] : node_indicator.indptr[test_index + 1]
            ]
            path_length = len(participated_component_indices)
            assert path_length >= 2, f"Test test_index: ({participated_component_indices}, total {path_length}) has no participated nodes in the tree, but should have at least 2 (root and terminal node)."
            for node in map(self.mapped_model.__getitem__, participated_component_indices):
                self.spectra[self.inverse_spectra_map[node], test_index] = (node.depth + 1) / path_length if self.combine_components_depth else 1
                if node.is_terminal():
                    terminals_count += not test_index
                    if self.combine_prior_confidence:
                        self.error_vector[test_index] += node.confidence if discrete_error_vector[test_index] else (1 - node.confidence)
            test_participation_vector = self.spectra[:, test_index]
            self.path_tests_indices[tuple(test_participation_vector)].append(test_index)

        if self.combine_prior_confidence:
            self.error_vector /= terminals_count
        else:
            self.error_vector = discrete_error_vector
        
        if not np.any(self.error_vector > 0):
            raise SFLDT.UnaffectedModelException("No errors found in the error vector, cannot perform diagnosis.")
        
        if self.group_feature_nodes:
            self.update_spectra_to_feature_components()
        if self.is_participation_fuzzy:
            self.update_spectra_to_fuzzy()
        if self.is_error_fuzzy:
            self.update_error_vector_to_fuzzy()
        
        if self.is_participation_fuzzy:
            assert np.all((0 <= self.spectra) & (self.spectra <= 1)), f"Participation spectra suppose to be fuzzy, while some of the components participation values are not in the range [0, 1] (Min: {np.min(self.spectra)}, Max: {np.max(self.spectra)})."
        else:
            assert np.isin(self.spectra, [0, 1]).all(), f"The spectra isn't binary while suppose to be (values: {np.unique(self.spectra)})"

        if self.is_error_fuzzy:
            assert np.all((0 <= self.error_vector) & (self.error_vector <= 1)), f"Error vector suppose to be fuzzy, while some of the tests results not in the range [0, 1] (Min: {np.min(self.error_vector)}, Max: {np.max(self.error_vector)})."
        else:
            assert np.isin(self.error_vector, [0, 1]).all(), f"The error vector isn't binary while suppose to be (values: {np.unique(self.error_vector)})"
        
    def convert_features_diagnosis_to_nodes_diagnosis(self,
                                                      features_diagnosis: list[int]
     ) -> list[int]:
        """
        Convert the features diagnosis to nodes diagnosis.
        The function will return the spectra indices of the nodes that their feature are part of the features diagnosis.
        Parameters:
            features_diagnosis (list[int]): The features diagnosis.
        Returns:
            list[int]: The diagnosis with node spectra indices.
        """
        nodes_diagnosis = []
        for feature_index in features_diagnosis:
            nodes_diagnosis.extend(self.feature_index_to_node_indices_dict[feature_index])
        return nodes_diagnosis
    
    def convert_diagnoses_indices(self,
                                  retrieve_spectra_indices: bool = False
    ) -> None:
        """
        Convert the diagnoses indices from spectra indices to node indices.
        If the feature components are used, the function will first convert the indices to node indices.
        Parameters:
            retrieve_spectra_indices (bool): Whether to return the spectra indices or the node indices.
        """
        if self.group_feature_nodes:
            self.diagnoses = [(self.convert_features_diagnosis_to_nodes_diagnosis(diagnosis), rank) for diagnosis, rank in self.diagnoses]
        if retrieve_spectra_indices:
            return
        return_indices_diagnoses = []
        for diagnosis, rank in self.diagnoses:
            diagnosis = [self.convert_spectra_index_to_node_index(spectra_index) for spectra_index in diagnosis]
            return_indices_diagnoses.append((diagnosis, rank))
        self.diagnoses = return_indices_diagnoses

# Similarity functions

    def get_relevant_similarity_function(self):
        """
        Get the relevant similarity function based of the type of the vectors (participation and error).:
        if both are binary - use faith similarity.
        if both are continuous - use correlation.
        if one is binary and the other continuous - use BCE similarity.
        Parameters:
        
        Returns:
            The relevant similarity function
        """

        def get_faith_similarity(participation_vector: np.ndarray, error_vector: np.ndarray) -> float:
            """
            Get the faith similarity of the component to the error vector.

            The similarity is calculated by
                (error_participation +  0.5 * accurate_nonparticipation) /
                (error_participation + accurate_participation + error_nonparticipation + accurate_nonparticipation)
            Parameters:
                participation_vector (ndarray): The participation vector, where high value (1) represent high participation in the sample classification.
                error_vector (ndarray): The error vector, where high value (1) represent that the sample classified incorrectly.

            Returns:
                float: The faith similarity between the vectors.
            """
            n11 = participation_vector @ error_vector
            n10 = participation_vector @ (1 - error_vector)
            n01 = (1 - participation_vector) @ error_vector
            n00 = (1 - participation_vector) @ (1 - error_vector)
            
            return (n11 +  0.5 * n00) / (n11 + n10 + n01 + n00)

        def get_cosine_similarity(participation_vector: np.ndarray, error_vector: np.ndarray) -> float:
            """
            Get the cosine similarity of the component to the error vector.
            Parameters:
                participation_vector (ndarray): The participation vector, where high value (1) represent high participation in the sample classification.
                error_vector (ndarray): The error vector, where high value (1) represent that the sample classified incorrectly.

            Returns:
                float: The cosine similarity between the vectors.
            """
            participation_vector, error_vector = participation_vector[None, :], error_vector[None, :]
            return cosine_similarity(participation_vector, error_vector)[0][0]

        def get_correlation(participation_vector: np.ndarray, error_vector: np.ndarray) -> float:
            """
            Get the correlation of the component to the error vector.
            Parameters:
                participation_vector (ndarray): The participation vector, where high value (1) represent high participation in the sample classification.
                error_vector (ndarray): The error vector, where high value (1) represent that the sample classified incorrectly.

            Returns:
                float: The correlation similarity between the vectors.
            """
            return pearson_correlation(participation_vector, error_vector)[0]

        def get_BCE_similarity(participation_vector: np.ndarray, error_vector: np.ndarray) -> float:
            """
            Get binary-cross-entropy similarity between the two vectors.
            for this similarity one of the vectors should be binary.
            the similarity is calculated as e^(-BCE) so high value means strong relationship.
            Parameters:
                participation_vector (ndarray): The participation vector, where high value (1) represent high participation in the sample classification.
                error_vector (ndarray): The error vector, where high value (1) represent that the sample classified incorrectly.

            Returns:
                float: The binary-cross-entropy similarity between the vectors.
            """
            def get_binary_continuous_vectors(participation_vector: np.ndarray,
                                            error_vector: np.ndarray
            ) -> tuple[np.ndarray]:
                """
                determine which vector is binary and which is continuous.
                Parameters:
                    participation_vector (ndarray): The participation vector, where high value (1) represent high participation in the sample classification.
                    error_vector (ndarray): The error vector, where high value (1) represent that the sample classified incorrectly.
                
                Returns:
                    ndarray: the binary vector.
                    ndarray: the continuous vector.
                """
                if self.is_error_fuzzy:
                    return participation_vector, error_vector
                return error_vector, participation_vector
            
            binary_vector, continuous_vector = get_binary_continuous_vectors(participation_vector, error_vector)
            continuous_vector = np.clip(continuous_vector, constants.EPSILON, 1 - constants.EPSILON)
            bce_loss = -np.mean(binary_vector * np.log(continuous_vector) + (1 - binary_vector) * np.log(1 - continuous_vector))
            return np.exp(-bce_loss)
    
    
        # Get the relevant function
        are_continuous = self.is_participation_fuzzy, self.is_error_fuzzy
        is_constant_vector = lambda vector: np.all(np.isclose(vector, vector[0]))
        if all(are_continuous): # both continuous
            # Check whether correlation test can be performed. For correlation test we need at least 2 samples and that no vector will be constant (either participation or error)
            if self.tests_count < 2:    # not enough samples for correlation measure
                return get_cosine_similarity
            is_any_participation_vector_constant, is_error_vector_constant = any(map(is_constant_vector, self.spectra)), is_constant_vector(self.error_vector)
            if any((is_any_participation_vector_constant, is_error_vector_constant)):
                # constant participation, cannot calculate correlation, using cosine similarity instead
                return get_cosine_similarity
            return get_correlation
        if any(are_continuous): # one binary one continuous
            return get_BCE_similarity
        # both binary
        return get_faith_similarity
    
    def load_stat_diagnoses(self
     ) -> list[tuple[int, float]]:
        """
        Load the diagnoses from the STAT diagnoser.
        The diagnoses will be used to combine with the SFLDT diagnoses.
        """
        assert self.stat, "STAT diagnoser is not initialized"
        return self.stat.get_diagnoses(retrieve_ranks=True)
    
    def combine_stat_diagnoses(self
     ) -> None:
        """
        Combine stat diagnoses with the SFLDT diagnoses.
        the combination is done by multiplying the current diagnosis rank with the average STAT rank of all the corresponding diagnosis nodes.

        The process is as follows - for each diagnosis (can contains either single or multiple nodes):
        1. Convert the spectra indices for the corresponding node indices.
        2. Get the corresponding STAT ranks for the nodes.
        3. Calculate the average STAT rank for the nodes.
        4. Multiply the SFLDT rank with the calculated average STAT rank.
        """
        stat_diagnoses_dict = {node_index[0]: rank for node_index, rank in self.load_stat_diagnoses()}
        convert_spectra_to_node_indices_function = lambda spectra_indices: map(self.convert_spectra_index_to_node_index, spectra_indices)
        get_nodes_stat_ranks_function = lambda spectra_indices: map(stat_diagnoses_dict.get, convert_spectra_to_node_indices_function(spectra_indices))
        get_average_stat_rank_function = lambda spectra_indices: max(0.5, sum(get_nodes_stat_ranks_function(spectra_indices)) / len(spectra_indices))
        get_nodes_from_features = lambda spectra_indices: self.convert_features_diagnosis_to_nodes_diagnosis(spectra_indices) if self.group_feature_nodes else spectra_indices
        self.diagnoses = [(diagnosis, sfldt_rank * get_average_stat_rank_function(get_nodes_from_features(diagnosis))) for diagnosis, sfldt_rank in self.diagnoses]
    
    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
                      retrieve_spectra_indices: bool = False
     ) -> list[int] | list[tuple[int, float]]:
        """
        Get the diagnoses of the nodes.
        The diagnoses consists the nodes ordered by their similarity to the error vector (DESC).

        Parameters:
        retrieve_spectra_indices (bool): Whether to return the spectra indices or the node indices.
        retrieve_ranks (bool): Whether to return the diagnoses rank.

        Returns:
        list[int] | list[tuple[int, float]]: The diagnoses. If retrieve_ranks is True, the diagnoses will be a list of tuples,
          where the first element is the index and the second is the similarity rank.
        """
        if self.diagnoses is None:
            similarity_measure_function = self.get_relevant_similarity_function()
            self.diagnoses = [([spectra_index], similarity_measure_function(self.spectra[spectra_index], self.error_vector)) for spectra_index in range(self.components_count)]
            if self.stat:
                self.combine_stat_diagnoses()
        self.convert_diagnoses_indices(retrieve_spectra_indices)
        return super().get_diagnoses(retrieve_ranks)