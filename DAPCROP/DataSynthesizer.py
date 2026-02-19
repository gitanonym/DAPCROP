import pandas as pd
import numpy as np

from DAPCROP import Constants as constants

class DataSynthesizer:
    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 data_feature_types: dict):
        self.count = len(X)

        self.class_counts = y.value_counts().to_dict()
        self.class_distribution = {class_name: class_count / self.count for class_name, class_count in self.class_counts.items()}

        self.features_distribution = {}
        for feature in X.columns:
            assert isinstance(data_feature_types[feature], constants.FeatureType)
            match data_feature_types[feature]:
                case constants.FeatureType.Binary | constants.FeatureType.Categorical:
                    feature_counts = X[feature].value_counts().to_dict()
                    self.features_distribution[feature] = {category: feature_counts[category] / self.count for category in feature_counts.keys()}
                case constants.FeatureType.Numeric:
                    self.features_distribution[feature] = {
                        'mean': X[feature].mean(),
                        'std': X[feature].std(),
                        'min': X[feature].min(),
                        'max': X[feature].max()
                        }
                case _:
                    raise ValueError(f"Unsupported feature type: {data_feature_types[feature]} for feature: {feature}")

        # Caching
        self.synthesized_X, self.synthesized_y = None, None

    def synthesize(self) -> tuple[pd.DataFrame, pd.Series]:
        if self.synthesized_X is None or self.synthesized_y is None:
            self.synthesized_X = pd.DataFrame()
            for feature, distribution in self.features_distribution.items():
                if isinstance(distribution, dict) and 'mean' in distribution:
                    self.synthesized_X[feature] = pd.Series(
                        np.random.normal(loc=distribution['mean'], scale=distribution['std'], size=self.count)
                    ).clip(lower=distribution['min'], upper=distribution['max'])
                else:
                    self.synthesized_X[feature] = pd.Series(
                        np.random.choice(list(distribution.keys()), size=self.count, p=list(distribution.values()))
                    )

            self.synthesized_y = pd.Series(
                np.random.choice(
                    list(self.class_distribution.keys()),
                    size=self.count,
                    p=list(self.class_distribution.values())
                )
            )

        return self.synthesized_X, self.synthesized_y