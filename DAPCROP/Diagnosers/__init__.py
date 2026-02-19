import os
import importlib
import inspect

from .ADiagnoser import ADiagnoser
from .Oracle import Oracle
from .DistributedForestDiagnoser import DistributedForestDiagnoser

from DAPCROP import Constants as constants

DIAGNOSER_CLASSES_DICT = {}

def _load_diagnoser_classes():
    """
    Load all classes in this package and subpackages that inherit from ADiagnoser.
    """
    base_dir = os.path.dirname(__file__)
    base_pkg = __name__

    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith(".py") and filename != "__init__.py":
                # Build module path
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, base_dir)
                module_name = rel_path[:-3].replace(os.sep, '.')
                full_module_name = f"{base_pkg}.{module_name}"

                module = importlib.import_module(full_module_name)
                for _, cls in inspect.getmembers(
                    module,
                    lambda c: inspect.isclass(c) and issubclass(c, (ADiagnoser, DistributedForestDiagnoser))
                ):
                    DIAGNOSER_CLASSES_DICT[cls.__name__] = cls

_load_diagnoser_classes()

def get_diagnoser(diagnoser: str | type[Oracle]
 ) -> ADiagnoser:
    """
    Get a diagnoser class by its name.
    
    Parameters:
    diagnoser (str | type[Oracle]): The name of the diagnoser class Or the Oracle class
    
    Returns:
    ADiagnoser: The diagnoser class.
    
    Raises:
    ValueError: If the diagnoser class is not found.
    """
    DiagnoserClass: type[ADiagnoser]
    if diagnoser is Oracle:
        DiagnoserClass = Oracle
    else:
        assert diagnoser in DIAGNOSER_CLASSES_DICT, f"Diagnoser {diagnoser} is not supported"
        DiagnoserClass = DIAGNOSER_CLASSES_DICT[diagnoser]
    
    match constants.DRIFTING_MODEL_TYPE:
        case constants.DriftingModelType.Tree:
            return DiagnoserClass
        case constants.DriftingModelType.Forest:
            if not constants.DISTRIBUTE_DIAGNOSES_COMPUTATION:
                return DiagnoserClass
            return lambda *args, **kwargs: DistributedForestDiagnoser(
                DiagnoserClass,
                *args, **kwargs
                )

__all__ = ["ADiagnoser", "get_diagnoser", "Oracle"]