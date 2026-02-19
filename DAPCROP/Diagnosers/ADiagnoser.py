import pandas as pd
from abc import ABC, abstractmethod

from DAPCROP.ModelMapping.ATreeBasedMappedModel import ATreeBasedMappedModel
from DAPCROP.ModelMapping.TreeNodeComponent import TreeNodeComponent

class ADiagnoser(ABC):
    def __init__(self, 
                 mapped_model: ATreeBasedMappedModel,
                 X: pd.DataFrame,
                 y: pd.Series
    ):
        """
        Initialize the STAT diagnoser.
        
        Parameters:
        mapped_model (ATreeBasedMappedModel): The mapped model.
        X (DataFrame): The data.
        y (Series): The target column.
        """
        self.mapped_model = mapped_model
        self.X_after = X
        self.y_after = y
        self.diagnoses = None

    def get_diagnoses_without_ranks(self,
                                    diagnoses: list[int] | list[tuple[int, float]] | list[list[int]] | list[tuple[list[int], float]]
    ):
        return [diagnosis for diagnosis, _ in diagnoses]
    
    def sort_diagnoses(self):
        self.diagnoses.sort(key=lambda diagnosis: diagnosis[1], reverse=True)
        
    @abstractmethod
    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
     ) -> list[list[int]] | list[tuple[list[int], float]]:
        """
        Get the diagnosis of the nodes.
        Each diagnosis consist nodes.
        The diagnoses ordered by their rank.

        Parameters:
        retrieve_ranks (bool): Whether to return the diagnosis ranks.

        Returns:
        list[list[int]] | list[tuple[list[int], float]]: The diagnosis (can be single or multiple). If retrieve_ranks is True, the diagnosis will be a list of tuples,
          where the first element is the diagnosis and the second is the rank.
        """
        self.sort_diagnoses()
        return self.diagnoses if retrieve_ranks else self.get_diagnoses_without_ranks(self.diagnoses)