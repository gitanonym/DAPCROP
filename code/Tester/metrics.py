import warnings
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import accuracy_score

from DAPCROP import *

import Tester.TesterConstants as tester_constants

def get_accuracy(model, X, y):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        y_predicted = model.predict(X)
        return accuracy_score(y, y_predicted)

def get_top_k(mapped_model: ATreeBasedMappedModel,
              diagnoses: list[list[int]],
              faulty_features: set[str],
              K: int = 1
              ) -> float:
    
    def get_single_correctly_identified_ratio(detected_faulty_features: list[str]) -> float:
        """
        Calculate the correctly identified ratio of the diagnoser.
        Calculated as
                        | nodes with faulty feature |
                    _________________________________
                                | nodes |
        Parameters:
        detected_faulty_features (list[str]): The detected faulty features by the diagnoser.
        true_faulty_features (set[str]): The true faulty features.
        
        Returns:
        float: The correctly identified ratio.
        """
        return sum(map(faulty_features.__contains__, detected_faulty_features)) / len(detected_faulty_features)
    
    def get_current_correctly_identified_ratio(diagnosis: list[int]) -> float:
        diagnosis_divided_to_trees: list[list[int]] = [diagnosis]
        if isinstance(mapped_model, MappedForest):
            group_by_tree_map = defaultdict(list)
            for node_index in diagnosis:
                estimator_index, _ = mapped_model.component_estimator_map[node_index]
                group_by_tree_map[estimator_index].append(node_index)
            diagnosis_divided_to_trees = list(group_by_tree_map.values())
        
        correctly_identified_ratio_sums: float = 0
        for tree_diagnosis in diagnosis_divided_to_trees:
            tree_diagnosis_detected_faulty_features = list(map(lambda node_index: mapped_model[node_index].feature, tree_diagnosis))
            correctly_identified_ratio_sums += get_single_correctly_identified_ratio(tree_diagnosis_detected_faulty_features)
        
        return correctly_identified_ratio_sums / len(diagnosis_divided_to_trees)
        
    
    top_k = 0
    for k in range(min(K, len(diagnoses))):
        current_diagnosis = diagnoses[k]
        top_k += get_current_correctly_identified_ratio(current_diagnosis)
        if top_k >= 1:
            break

    return min(1.0, top_k)

def _get_wasted_effort(mapped_model: ATreeBasedMappedModel,
                       diagnoses: list[list[int]],
                       faulty_features_nodes: dict[str, list[int]]
 ) -> float:
    """
    Calculate the wasted effort of the diagnoser.
    In here the wasted effort is calculated for fixes of all the nodes that include a faulty feature.

    Parameters:
    mapped_model (ATreeBasedMappedModel): The mapped model.
    diagnoses (list[list[int]]): The diagnoses of the nodes.
    faulty_features_nodes (dict[str, list[int]]): Dict of the drifted features and their corresponding faulty nodes indices.
    require_full_fix (bool): If True, the diagnoser is required to fix all faulty nodes of a feature

    Returns:
    float: The wasted effort (can be float in case we fix randomly with float expected value).
    """
    faulty_features_healthy_nodes = defaultdict(list)
    
    for node in mapped_model:
        node_index, node_feature = node.get_index(), node.feature
        if node_feature in faulty_features_nodes and node_index not in faulty_features_nodes[node_feature]:
            faulty_features_healthy_nodes[node_feature].append(node_index)

    undetected_faulty_features_nodes = deepcopy(faulty_features_nodes)
    wasted_effort_nodes = set()
    are_all_faults_detected = lambda: not any(undetected_faulty_features_nodes.values())
    for diagnosis in diagnoses:
        for diagnosed_faulty_node in map(mapped_model.__getitem__, diagnosis):
            diagnosed_faulty_feature = diagnosed_faulty_node.feature
            if diagnosed_faulty_feature not in undetected_faulty_features_nodes or diagnosed_faulty_node.get_index() in faulty_features_healthy_nodes[diagnosed_faulty_feature]:
                # a wasted effort
                wasted_effort_nodes.add(diagnosed_faulty_node)
            else:
                # relevant fix
                if tester_constants.WASTED_EFFORT_REQUIRE_FULL_FIX:
                    if diagnosed_faulty_node.get_index() in undetected_faulty_features_nodes[diagnosed_faulty_feature]:
                        undetected_faulty_features_nodes[diagnosed_faulty_feature].remove(diagnosed_faulty_node.get_index())
                else:
                    undetected_faulty_features_nodes[diagnosed_faulty_feature] = []
        if are_all_faults_detected():
            break

    current_wasted_effort = len(wasted_effort_nodes)
    healthy_nodes_count = len(mapped_model) - sum(map(len, faulty_features_nodes.values()))

    wasted_effort: int
    if are_all_faults_detected():
        wasted_effort = current_wasted_effort
    else:
        # Didn't detect all the faulty features, handle all the missing nodes
        current_undetected_faults_count = sum(map(len, undetected_faulty_features_nodes.values()))

        assert healthy_nodes_count >= len(wasted_effort_nodes), "Wasted effort nodes count is greater than the healthy nodes count (suppose to be subset of)"
        current_undetected_wasted_effort = healthy_nodes_count - current_wasted_effort
            
        handling_missing_action = {
            "all": healthy_nodes_count,
            "none": current_wasted_effort,  # (+ 0)
            "random": current_wasted_effort + (
                current_undetected_wasted_effort * (current_undetected_faults_count if tester_constants.WASTED_EFFORT_REQUIRE_FULL_FIX else 1)
                ) / (current_undetected_faults_count + 1)
        }
                
        wasted_effort = handling_missing_action[tester_constants.WASTED_EFFORT_MISSING_ACTION]
    
    if tester_constants.NORMALIZE_WASTED_EFFORT:
        # normalize by total number of components
        wasted_effort = wasted_effort / healthy_nodes_count * 100
    
    return wasted_effort

def get_wasted_effort(mapped_model: ATreeBasedMappedModel,
                      diagnoses: list[list[int]],
                      faulty_features_nodes: dict[str, list[int]]
 ) -> float:
    if isinstance(mapped_model, MappedDecisionTree):
        return _get_wasted_effort(mapped_model, diagnoses, faulty_features_nodes)
    
    # split the diagnoses by estimators
    estimators_diagnoses: dict[int, list[list[int]]] = defaultdict(list)
    for diagnosis_index, diagnosis in enumerate(diagnoses):
        for node_index in diagnosis:
            estimator_index, component_sklearn_index = mapped_model.component_estimator_map[node_index]
            if len(estimators_diagnoses[estimator_index]) <= diagnosis_index:
                estimators_diagnoses[estimator_index].append([])
            estimators_diagnoses[estimator_index][diagnosis_index].append(component_sklearn_index)

    # split the faulty features nodes by estimators
    estimators_faulty_features_nodes: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for faulty_feature, faulty_nodes in faulty_features_nodes.items():
        for node_index in faulty_nodes:
            estimator_index, component_sklearn_index = mapped_model.component_estimator_map[node_index]
            estimators_faulty_features_nodes[estimator_index][faulty_feature].append(component_sklearn_index)
    
    # calculate wasted effort per estimator
    summed_wasted_effort: float = 0
    for estimator_index, mapped_estimator in enumerate(mapped_model.mapped_estimators):
        estimator_diagnoses, estimator_faulty_features_nodes = estimators_diagnoses[estimator_index], estimators_faulty_features_nodes[estimator_index]
        match (bool(estimator_diagnoses), bool(estimator_faulty_features_nodes)): 
            case (True, True) | (False, True):
                # there are faulty nodes - will be fixed based of the diagnoses
                summed_wasted_effort += _get_wasted_effort(
                    mapped_estimator,
                    estimator_diagnoses,
                    estimator_faulty_features_nodes
                )
            case (False, False):
                # no faulty nodes and none were fixed - no waste
                summed_wasted_effort += 0
            case (True, False):
                # no faulty nodes - all nodes are healthy and will be fixed
                if tester_constants.NORMALIZE_WASTED_EFFORT:
                    summed_wasted_effort += 100
                else:
                    summed_wasted_effort += len(mapped_estimator)
    
    # average the wasted effort per estimator
    return summed_wasted_effort / len(mapped_model.mapped_estimators)