from sklearn.tree import DecisionTreeClassifier

from DAPCROP.ModelMapping.MappedDecisionTree import MappedDecisionTree

from .AIndependentFixer import AIndependentFixer

class TopNodeFixer(AIndependentFixer):
    alias = "single_top_tweak"
    def fix_model(self) -> tuple[DecisionTreeClassifier, list[int]]:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        top_faulty_node_index = min(self.faulty_nodes_indices, key=lambda node_index: self.mapped_model[node_index].depth)
        X_reached_top_faulty_node, y_reached_faulty_node = self._filter_data_reached_fault(top_faulty_node_index)
        self.fix_faulty_node(top_faulty_node_index, X_reached_top_faulty_node, y_reached_faulty_node)
        return super().fix_model()