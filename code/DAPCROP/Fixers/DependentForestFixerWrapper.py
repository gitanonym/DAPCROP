import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from DAPCROP.ModelMapping.MappedDecisionTree import MappedDecisionTree

from .ForestFixerWrapper import ForestFixerWrapper


class DependentForestFixerWrapper(ForestFixerWrapper):
    """
    A fixer wrapper for ensemble classifiers that fixes trees iteratively
    with distance-based weight reduction.
    
    Unlike ForestFixer fixing (which can be parallelized), fixing is sequential
    because each fix depends on the previous ones.  For each sample we track
    the most recent fixed tree in which it reached a faulty node.  When fixing
    a tree, a sample's weight is ``1 - decay_rate ** d`` where *d* is the
    number of steps since its last faulty-node appearance.  Samples that
    appeared recently get the lowest weight; the penalty fades as *d* grows.
    Samples that never appeared in any faulty node keep full weight (1.0).
    Weights are passed to the base fixer so that it can take them into
    account when fixing each tree.
    """
    alias = "dependent_forest_fixer_wrapper"

    def __init__(self, *args, decay_rate: float = 0.5, **kwargs):
        """
        Args:
            *args: Positional arguments forwarded to ForestFixerWrapper.
            decay_rate: Controls how quickly the penalty fades with distance.
                        Must be in (0, 1).  Lower values mean the penalty
                        disappears faster.  Default: 0.5.
            **kwargs: Extra keyword arguments forwarded to ForestFixerWrapper
                      (and ultimately to the base fixer).
        """
        super().__init__(*args, **kwargs)
        if not 0 < decay_rate < 1:
            raise ValueError(f"decay_rate must be in (0, 1), got {decay_rate}")
        self.decay_rate = decay_rate

    def _get_samples_reaching_faulty_nodes(self,
                                           mapped_estimator: MappedDecisionTree,
                                           faulty_node_indices: list[int],
                                           X: pd.DataFrame
    ) -> pd.Index:
        """
        Get the indices of samples that reach any faulty node in the given estimator.
        
        Args:
            mapped_estimator: The mapped decision tree estimator
            faulty_node_indices: List of faulty node indices in this estimator
            X: The data to check
            
        Returns:
            pd.Index: Indices of samples that reached any faulty node
        """
        samples_reaching_fault = set()
        for node_index in faulty_node_indices:
            faulty_node = mapped_estimator[node_index]
            reached_X = faulty_node.get_data_reached_node(X, allow_empty=True)
            samples_reaching_fault.update(reached_X.index)
        return pd.Index(list(samples_reaching_fault))

    def fix_model(self) -> AdaBoostClassifier | RandomForestClassifier:
        """
        Fix the model using iterative processing with distance-based weighting.
        
        For each sample we track the last step at which it reached a faulty
        node.  Before each tree fix the sample weight is computed as
        ``1 - decay_rate ** distance``, where *distance* is the number of
        steps since the last faulty appearance.  Samples that never appeared
        in a faulty node receive weight 1.  All samples are passed to the
        base fixer together with these weights.

        Returns:
            AdaBoostClassifier | RandomForestClassifier: The fixed ensemble model.
        """
        self.fixed_model: AdaBoostClassifier | RandomForestClassifier = deepcopy(self.sklearn_model)
        
        faulty_estimator_indices = sorted(self.faulty_estimators.keys())
        
        # For each sample, the step index at which it last reached a faulty node.
        # NaN means the sample has never reached a faulty node.
        last_faulty_step = pd.Series(np.nan, index=self.X.index)
        
        for step, estimator_index in enumerate(faulty_estimator_indices):
            mapped_estimator = self.mapped_model.mapped_estimators[estimator_index]
            
            # Compute per-sample weights based on distance to last faulty appearance
            weights = pd.Series(1.0, index=self.X.index)
            appeared = last_faulty_step.notna()
            if appeared.any():
                distance = step - last_faulty_step[appeared]
                weights[appeared] = 1 - self.decay_rate ** distance
                
            fixer = self.base_fixer_class(
                mapped_estimator,
                self.X,
                self.y,
                self.faulty_estimators[estimator_index],
                self.X_prior,
                self.y_prior,
                sample_weight=weights,
                **self.base_fixer_parameters
            )
            self.fixed_model.estimators_[estimator_index] = fixer.fix_model()
            
            # Record which samples reached faulty nodes in this tree
            reaching_samples = self._get_samples_reaching_faulty_nodes(
                mapped_estimator,
                self.faulty_estimators[estimator_index],
                self.X
            )
            last_faulty_step.loc[reaching_samples] = step

        return self.fixed_model
