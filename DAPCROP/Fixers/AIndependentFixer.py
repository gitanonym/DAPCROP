import pandas as pd
import numpy as np
from copy import deepcopy

from sklearn.tree import DecisionTreeClassifier

from DAPCROP import Constants as constants

from .AFixer import *

class AIndependentFixer(ATreeFixer):
    def _fix_terminal_faulty_node(self,
                                 faulty_node_index: int,
                                 X_reached_faulty_node: pd.DataFrame,
                                 y_reached_faulty_node: pd.Series,
                                 sample_weight: pd.Series = None
     ) -> None:
        """
        Fix a terminal faulty node.
        The fix is done by changing the class of the node to the most common class in the data that reached the node (after the drift).
        If the data that reached the node is empty, the fix is done by switching between the top and second top classes.
        
        Parameters:
            faulty_node_index (int): The index of the faulty node.
            X_reached_faulty_node (DataFrame): The data that reached the faulty node.
            sample_weight (Series, optional): Per-sample weights aligned with the reached data.
        """
        values = self.sklearn_model.tree_.value[faulty_node_index]

        if len(y_reached_faulty_node):
            # Get the most common class in the data that reached the faulty node
            if sample_weight is not None:
                most_common_class_index = sample_weight.groupby(y_reached_faulty_node).sum().idxmax()
            else:
                most_common_class_index = y_reached_faulty_node.value_counts().idxmax()

            # Make the most common class the class with the max count in the node
            max_value_count = np.max(values)
            values[0][most_common_class_index] = max_value_count + 1
        else:
            # Switch between top and second top classes
            max_value_index = values.argmax()
            max_value = values[0][max_value_index]
            values[0][max_value_index] = 0
            second_max_value_index = values.argmax()
            values[0][max_value_index] = max_value
            values[0][second_max_value_index] = max_value
        
        # print(f"{self.diagnoser_output_name}: Faulty node {faulty_node_index} (Terminal) class changed from {max(old_values[0])} to {max(values[0])}")
        self.fixed_model.tree_.value[faulty_node_index] = values
        self.mapped_model[faulty_node_index].class_name = self.fixed_model.classes_[values.argmax()]


    def _fix_numeric_faulty_node(self, 
                                  faulty_node_index: int,
                                  X_reached_faulty_node: pd.DataFrame,
                                  sample_weight: pd.Series = None
     ) -> None:
        """
        Fix a numeric faulty node.
        The fix is done by replacing the threshold of the node

        Parameters:
            faulty_node_index (int): The index of the faulty node.
            X_reached_faulty_node (DataFrame): The data that reached the faulty node.
            sample_weight (Series, optional): Per-sample weights aligned with the reached data.
        """
        faulty_node = self.mapped_model[faulty_node_index]
        node_feature_average_before_drift = faulty_node.feature_average_value
        if node_feature_average_before_drift is None:
            raise NotImplementedError("The average feature value before the drift is not available")
        if sample_weight is not None:
            node_feature_average_after_drift = np.average(
                X_reached_faulty_node[faulty_node.feature], weights=sample_weight.values
            )
        else:
            node_feature_average_after_drift = X_reached_faulty_node[faulty_node.feature].mean()
        node_feature_average_difference = node_feature_average_after_drift - node_feature_average_before_drift
        new_threshold = faulty_node.threshold + node_feature_average_difference
        # print(f"{self.diagnoser_output_name}: Faulty node {faulty_node_index} (Numeric) threshold changed from {faulty_node.threshold:.2f} to {new_threshold:.2f}")
        self.mapped_model[faulty_node_index].threshold = new_threshold
        self.fixed_model.tree_.threshold[faulty_node_index] = new_threshold

    def _fix_categorical_faulty_node(self,
                                    faulty_node_index: int,
     ) -> None:
          """
          Fix a categorical faulty node.
          The fix is done by flipping the switch of the condition in it.
    
          Parameters:
                faulty_node_index (int): The index of the faulty node.
                X_reached_faulty_node (DataFrame): The data that reached the faulty node.
          """
          faulty_node = self.mapped_model[faulty_node_index]
          left_child, right_child = faulty_node.left_child, faulty_node.right_child
          left_child_index, right_child_index = left_child.get_index(), right_child.get_index()
          self.fixed_model.tree_.children_left[faulty_node_index] = right_child_index
          self.fixed_model.tree_.children_right[faulty_node_index] = left_child_index
          self.mapped_model[faulty_node_index].left_child, self.mapped_model[faulty_node_index].right_child = right_child, left_child
        #   print(f"{self.diagnoser_output_name}: Faulty node {faulty_node_index} (Categorical) condition flipped")
          
    def fix_faulty_node(self,
                        faulty_node_index: int,
                        X_reached_faulty_node: pd.DataFrame,
                        y_reached_faulty_node: pd.Series
     ) -> None:
        """
        Fix a faulty node.

        Parameters:
            faulty_node_index (int): The index of the faulty node.
            X_reached_faulty_node (DataFrame): The data that reached the faulty node.
        """
        self.fixed_model: DecisionTreeClassifier = deepcopy(self.sklearn_model)
        
        # Extract weights for the samples that reached this node
        sample_weight = None
        if self.sample_weight is not None:
            sample_weight = self.sample_weight.loc[X_reached_faulty_node.index]
        
        faulty_node = self.mapped_model[faulty_node_index]
        if faulty_node.is_terminal():
            self._fix_terminal_faulty_node(faulty_node_index, X_reached_faulty_node, y_reached_faulty_node, sample_weight)
            return
        faulty_node_feature_type: constants.FeatureType = faulty_node.feature_type
        if faulty_node_feature_type is None:
            # Determine the type from the after drift dataset
            faulty_node_feature_type: constants.FeatureType = self.feature_types[faulty_node.feature]
        match faulty_node_feature_type:
            case constants.FeatureType.Numeric:
                self._fix_numeric_faulty_node(faulty_node_index, X_reached_faulty_node, sample_weight)
            case constants.FeatureType.Binary | constants.FeatureType.Categorical:
                self._fix_categorical_faulty_node(faulty_node_index)
            case _:
                raise ValueError(f"Unsupported feature type: {faulty_node_feature_type} for faulty node {faulty_node_index}")