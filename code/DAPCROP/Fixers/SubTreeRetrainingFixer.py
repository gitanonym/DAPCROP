from sklearn.tree import DecisionTreeClassifier

import DAPCROP.Constants as constants
from DAPCROP.SubTreeReplaceableDecisionTree import SubTreeReplaceableDecisionTree

from .AFixer import *

class SubTreeRetrainingFixer(ATreeFixer):
    alias = "subtree_retrain"

    def __init__(self,
                 *args,
                 dependency_handling_type: str | constants.SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES = constants.DEFAULT_SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPE,
                 use_prior_knowledge: str | constants.PRIOR_KNOWLEDGE_USAGE_TYPES = constants.DEFAULT_USE_OF_PRIOR_KNOWLEDGE,
                 subtree_type: str | constants.SubTreeType = constants.DEFAULT_SUBTREE_TYPE,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # check type of handling type
        if isinstance(dependency_handling_type, str):
            dependency_handling_type: constants.SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES = \
                constants.SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES[dependency_handling_type]
        self.dependency_handling_type: constants.SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES = dependency_handling_type
        
        if isinstance(use_prior_knowledge, str):
            use_prior_knowledge: constants.PRIOR_KNOWLEDGE_USAGE_TYPES = \
                constants.PRIOR_KNOWLEDGE_USAGE_TYPES[use_prior_knowledge]
        self.use_prior_knowledge: constants.PRIOR_KNOWLEDGE_USAGE_TYPES = use_prior_knowledge

        if isinstance(subtree_type, str):
            subtree_type: constants.SubTreeType = constants.SubTreeType[subtree_type]
        self.subtree_type: constants.SubTreeType = subtree_type

    def fix_model(self) -> tuple[DecisionTreeClassifier, list[int]]:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        # print(f"Fixing faulty nodes: {self.faulty_nodes_indices}")
        self.fixed_model = SubTreeReplaceableDecisionTree(self.sklearn_model,
                                                         list(map(self.mapped_model.__getitem__, self.faulty_nodes_indices)),
                                                         dependency_handling_type=self.dependency_handling_type,
                                                         use_prior_knowledge=self.use_prior_knowledge,
                                                         subtree_type=self.subtree_type,
                                                         X_prior=self.X_prior,
                                                         y_prior=self.y_prior)
        
        self.fixed_model.fit(self.X, self.y, sample_weight=self.sample_weight)
        
        return super().fix_model()