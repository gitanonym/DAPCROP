from sklearn.tree import DecisionTreeClassifier

from DAPCROP.ModelMapping.MappedDecisionTree import MappedDecisionTree

from .AIndependentFixer import AIndependentFixer

class AllNodesFixer(AIndependentFixer):
    alias = "all_nodes_tweak"

    def fix_model(self) -> tuple[DecisionTreeClassifier, list[int]]:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        # print(f"Fixing faulty nodes: {self.faulty_nodes}")
        for faulty_node_index in self.faulty_nodes_indices:
            X_reached_faulty_node, y_reached_faulty_node = self._filter_data_reached_fault(faulty_node_index)
            self.fix_faulty_node(faulty_node_index, X_reached_faulty_node, y_reached_faulty_node)
        return super().fix_model()