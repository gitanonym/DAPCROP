import os
from scipy.io.arff import loadarff

import pandas as pd
import math

from typing import Generator, Callable

import Tester.TesterConstants as tester_constants

from .DriftSimulation import single_feature_concept_drift_generator, multiple_features_concept_drift_generator

class Dataset:
    partitions = ["before", "after", "repair", "test"]

    def __init__(self, 
                 source: str | pd.DataFrame
    ):
        """
        source (str | DataFrame): If str, the path to the dataset file; If DataFrame, the dataset itself.
        """
        # Get data
        if type(source) == str:    # Path to the file
            self.name, source_format = os.path.splitext(os.path.basename(source))
            if source_format in (".csv", ".data", ".txt"):
                source = pd.read_csv(source)
            elif source_format == ".arff":
                data, _ = loadarff(source)
                source = pd.DataFrame(data)
        assert isinstance(source, pd.DataFrame)

        self.target_name = source.columns[-1]
        self.data, y = self.split_features_targets(source)

        self.feature_types = {}
        one_hot_encoded_dict = {}
        for column_name in self.data.columns:
            column_type = self.data[column_name].dtype
            if column_type not in [object, bool]:   # Numeric
                self.data[column_name] = self.data[column_name].fillna(self.data[column_name].mean())    # Fill NaN values
                self.feature_types[column_name] = tester_constants.constants.FeatureType.Numeric
                continue
            # Categorical or Binary
            self.data[column_name] = self.data[column_name].fillna(self.data[column_name].mode().iloc[0])    # Fill NaN values
            if len(self.data[column_name].unique()) <= 2: # Consider as binary
                column_type = bool
            if column_type == bool or not tester_constants.ONE_HOT_ENCODING_CATEGORICAL:
                self.data[column_name] = pd.Categorical(self.data[column_name])
                self.data[column_name] = self.data[column_name].cat.codes
                self.feature_types[column_name] = tester_constants.constants.FeatureType.Binary if column_type == bool else tester_constants.constants.FeatureType.Categorical
                continue
            # One hot encoding with multiple values
            one_hot_encoded_dict.update({f"{column_name}_{value}": tester_constants.constants.FeatureType.Binary for value in self.data[column_name].unique()})
            self.data = pd.get_dummies(self.data, columns=[column_name])
        self.feature_types.update(one_hot_encoded_dict)

        self.data[self.target_name] = pd.Categorical(y.fillna(y.mode().iloc[0]))
        self.data[self.target_name] = self.data[self.target_name].cat.codes

        if tester_constants.DRIFT_SYNTHESIZING_VERSION == 1:  # shuffle data - same shuffle always
            self.data = self.data.sample(frac=1, random_state=tester_constants.constants.RANDOM_STATE).reset_index(drop=True)

        self.data.attrs["name"] = self.name

        n_samples = len(self.data)
        self.before_proportion, self.repair_proportion, self.test_proportion = tester_constants.PROPORTIONS_TUPLE
        self.before_concept_size = math.floor(self.before_proportion*n_samples)
        self.total_repair_size = math.floor(self.repair_proportion*n_samples)
        self.test_size = math.floor(self.test_proportion*n_samples)
        self.after_concept_size = self.total_repair_size + self.test_size

        assert all([0 < current_size for current_size in (self.before_concept_size, self.total_repair_size, self.test_size)])
        assert (self.before_concept_size + self.total_repair_size + self.test_size) <= n_samples

        self.update_repair_window_size(tester_constants.DEFAULT_REPAIR_WINDOW_PROPORTION)

    def __len__(self) -> int:
        return len(self.data)

    def update_repair_window_size(self, 
                          new_repair_window_size: int | float
     ) -> None:
        assert 0 < new_repair_window_size <= 1

        self.repair_window_proportion = new_repair_window_size
        self.repair_window_size = math.floor(self.total_repair_size * self.repair_window_proportion)

    def split_features_targets(self, 
                               data: pd.DataFrame
     ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split the data to X and y
        
        Parameters:
            data (DataFrame): The data to split
        
        Returns:
            tuple[DataFrame, Series]: The X and y
        """
        X = data.drop(columns=[self.target_name]).reset_index(drop=True)
        y = data[self.target_name].reset_index(drop=True)
        return X, y

    def get_before_concept_data(self) -> tuple[pd.DataFrame, pd.Series]:
        before_concept_data = self.data.iloc[:self.before_concept_size]
        return self.split_features_targets(before_concept_data)
    
    def get_after_concept_data(self) -> tuple[pd.DataFrame, pd.Series]:
        after_concept_data = self.data.iloc[self.before_concept_size:]
        return self.split_features_targets(after_concept_data)
    
    def get_repair_data(self) -> tuple[pd.DataFrame, pd.Series]:
        X_after, y_after = self.get_after_concept_data()
        return X_after.iloc[:self.total_repair_size], y_after.iloc[:self.total_repair_size]
    
    def get_test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        X_after, y_after = self.get_after_concept_data()
        return X_after.iloc[-self.test_size:], y_after.iloc[-self.test_size:]

    def _drift_data_generator(self,
                   data: pd.DataFrame,
                   drift_features: str | list[str],
                   severity_levels: tuple = tester_constants.DEFAULT_GENERATED_SEVERITY_LEVELS
     ) -> Generator[pd.DataFrame, None, None]:
        """
        Create a drift in the data
        
        Parameters:
            data (DataFrame): The data to drift
            drift_features (str or list): single feature or list of features to drift
            severity_levels (tuple[int]): The severity levels the column should be drifted to. Default is all.
                
        Returns:
            DataFrame: The drifted data
        """
        if type(drift_features) == str:
            assert drift_features in data.columns, f"Feature {drift_features} not in the dataset"
            feature_type = self.feature_types[drift_features]
            return single_feature_concept_drift_generator(data, drift_features, feature_type, severity_levels)
        assert all([feature in data for feature in drift_features]), "Not all features in the dataset"
        # Get subset of the dictionary
        drift_features_dict = {feature: self.feature_types[feature] for feature in drift_features}
        return multiple_features_concept_drift_generator(data, drift_features_dict, severity_levels)

    def drift_generator(self,
                        drift_features: str | set[str],
                        partition: str = "after",
                        severity_levels: tuple = tester_constants.DEFAULT_GENERATED_SEVERITY_LEVELS
     ) -> Generator[tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], int, str], None, None]:
        """
        Drift generator for a specific partition
        
        Parameters:
            drift_features (str or list): single feature or set of features to drift
            partition (str): The partition to drift.
            severity_levels (tuple[int]): The severity levels the column should be drifted to. Default is all.

        Returns:
            Generator[tuple[tuple[DataFrame, Series], str], None, None]: 
                A generator of all possible drifts in the feature and the description of the drift in the given partition name.
                Each drift represented by the (drifted dataset, original y) and the description of the drift and the drifted features.
        """
        assert partition in Dataset.partitions, "Invalid partition name"
        partition_function_mapping = {
            "before": self.get_before_concept_data,
            "after": self.get_after_concept_data,
            "repair": self.get_repair_data,
            "test": self.get_test_data
        }
        original_X, y = partition_function_mapping[partition]()
        for drifted_X, drift_severity_level, drift_description in self._drift_data_generator(original_X, drift_features, severity_levels):
            if partition == "before":
                yield (drifted_X, y), drift_severity_level, f"BEFORE_{drift_description}"
                continue
            drift_description += f"_window[{self.repair_proportion * self.repair_window_proportion * 100:.1f}%]"
            # split to repair and test
            X_repair, y_repair = drifted_X.iloc[:self.total_repair_size], y.iloc[:self.total_repair_size]
            X_repair_window, y_repair_window = drifted_X.iloc[:self.repair_window_size], y.iloc[:self.repair_window_size]
            X_test, y_test = drifted_X.iloc[self.test_size:], y.iloc[self.test_size:]
            yield (X_repair, y_repair), (X_repair_window, y_repair_window), (X_test, y_test), drift_severity_level, drift_description

    def get_sorted_data_by_features(self,
                              features: set[str]
     ) -> None:
        """
        Sort the data by the given features.
        
        Parameters:
            features (list[str]): The features to sort by.
        """
        if len(features) == 2:
            raise NotImplementedError("Sorting by more than one feature is not implemented yet")
        
        feature = next(iter(features))
        
        assert self.feature_types[feature] == tester_constants.constants.FeatureType.Numeric, "Sorting by non-numeric feature is not supported"
        
        # sort by the feature
        return self.data.sort_values(by=feature, ascending=True).reset_index(drop=True)

    def drift_generator_v2(self,
                           drift_features: set[str]
        ) -> Generator[tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], str], None, None]:
        """
        Drift generator for a specific partition - version 2
        
        The drift is done by sorting the data by the given features and splitting the data to before and after concept.
        The post-drift will be on the top or bottom of the sorted data.
        The before concept and after concept will be shuffled before re-partitioning to repair and test.

        Parameters:
            drift_features (list[str]): The features to drift.
        Returns:
            A generator with the following data:
                tuple[DataFrame, Series]: The before concept data (X, y).
                tuple[DataFrame, Series]: The repair data (X, y).
                tuple[DataFrame, Series]: The test data (X, y).
                str: The description of the drift.
        """
        sorted_data = self.get_sorted_data_by_features(drift_features)

        drift_type_mapping: dict[str, Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]] = {
            "top": lambda data: (data.iloc[: self.before_concept_size], data.iloc[-self.after_concept_size: ]), # after is the highest values
            "bottom": lambda data: (data.iloc[-self.before_concept_size: ], data.iloc[: self.after_concept_size]) # after is the lowest values
        }

        for drift_type, drift_function in drift_type_mapping.items():
            drift_description = f"{drift_type} {'-'.join(sorted(drift_features))}"
            before_concept_data, after_concept_data = drift_function(sorted_data)
            assert len(before_concept_data) == self.before_concept_size and len(after_concept_data) == self.after_concept_size, "Drifted data size is not correct"
            
            before_concept_data = before_concept_data.sample(frac=1, random_state=tester_constants.constants.RANDOM_STATE).reset_index(drop=True)
            after_concept_data = after_concept_data.sample(frac=1, random_state=tester_constants.constants.RANDOM_STATE).reset_index(drop=True)

            X_before, y_before = self.split_features_targets(before_concept_data)
            X_repair, y_repair = self.split_features_targets(after_concept_data.iloc[ :self.total_repair_size])
            X_repair_window, y_repair_window = self.split_features_targets(after_concept_data.iloc[ :self.repair_window_size])
            X_test, y_test = self.split_features_targets(after_concept_data.iloc[-self.test_size: ])

            yield (X_before, y_before), (X_repair, y_repair), (X_repair_window, y_repair_window), (X_test, y_test), drift_description