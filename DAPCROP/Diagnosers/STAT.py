from copy import deepcopy

from .ADiagnoser import *
from DAPCROP.ModelMapping.TreeNodeComponent import TreeNodeComponent

class STAT(ADiagnoser):
    def __init__(self, 
                 mapped_model: ATreeBasedMappedModel,
                 X: pd.DataFrame,
                 y: pd.Series
    ):
        super().__init__(mapped_model, X, y)
    
    @staticmethod
    def get_violation_ratio(node: TreeNodeComponent
     ) -> float:
        """
        Get the violation ratio of the node.

        Parameters:
        node (TreeNodeComponent): The node.

        Returns:
        float: The violation ratio of the node.
        """
        if node.is_terminal():
            return 0
        samples_reached_node_count = node.reached_samples_count
        if samples_reached_node_count == 0:
            return 0
        violated_samples_count = node.right_child.reached_samples_count
        return violated_samples_count / samples_reached_node_count

    def get_before_violation(self,
                             node_index: int
     ) -> float:
        """
        Get the violation of the node before the drift.
        
        Parameters:
        node_index (int): The node index.
        
        Returns:
        float: The violation of the node before the drift.
        """
        node = self.mapped_model[node_index]
        return STAT.get_violation_ratio(node)
    
    def get_after_violation(self,
                            node_index: int
     ) -> float:
        """
        Get the violation of the node after the drift.
        
        Parameters:
        node_index (int): The node index.
        
        Returns:
        float: The violation of the node after the drift.
        """
        node = self.mapped_model[node_index]
        # copy the node and calculate the violation for it
        node_after = deepcopy(node)
        node_after.update_node_data_attributes(self.X_after, self.y_after)
        return STAT.get_violation_ratio(node_after)
    
    def get_node_violation_difference(self,
                                      node_index: int,
     ) -> float:
        """
        Get the violation difference of the node.
        
        Parameters:
        node_index (int): The node index.
        
        Returns:
        float: The violation difference of the node.
        """
        before_violation = self.get_before_violation(node_index)
        after_violation = self.get_after_violation(node_index)
        return abs(before_violation - after_violation)

    def get_diagnoses(self,
                      retrieve_ranks: bool = False
     ) -> list[int] | list[tuple[int, float]]:
        """
        Get the diagnoses of the drift.


        Parameters:
        retrieve_ranks (bool): Whether to return the ranks of the nodes.
        
        Returns:
        list[int] | list[tuple[int, float]]: The diagnosis. If retrieve_ranks is True, the diagnosis will be a list of tuples,
          where the first element is the index and the second is the violation ratio.
        """
        if self.diagnoses is None:
            self.diagnoses = [([node_index], self.get_node_violation_difference(node_index)) for node_index in self.mapped_model.get_indices()]
            self.sort_diagnoses()
        return super().get_diagnoses(retrieve_ranks)