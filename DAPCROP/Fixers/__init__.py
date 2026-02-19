import os
import importlib
import inspect

from .AFixer import AFixer
from .ForestFixerWrapper import ForestFixerWrapper
from .DependentForestFixerWrapper import DependentForestFixerWrapper

from DAPCROP import Constants as constants

FIXER_CLASSES_DICT = {}

def _load_fixer_classes():
    """
    Load all classes in this package and subpackages that inherit from AFixer.
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
                    lambda c: inspect.isclass(c) and issubclass(c, (AFixer, ForestFixerWrapper)) and c is not AFixer
                ):
                    FIXER_CLASSES_DICT[cls.__name__] = cls

_load_fixer_classes()

def get_fixer(fixer_name: str
 ):
    """
    Get a fixer class by its name.
    
    Parameters:
    fixer_name (str): The name of the fixer class.
    
    Returns:
    AFixer: The fixer class.
    
    Raises:
    ValueError: If the fixer class is not found.
    """
    assert fixer_name in FIXER_CLASSES_DICT, f"Fixer {fixer_name} is not supported"
        
    FixerClass = FIXER_CLASSES_DICT[fixer_name]
    match constants.DRIFTING_MODEL_TYPE:
        case constants.DriftingModelType.Tree:
            return FixerClass
        case constants.DriftingModelType.Forest:
            FixerWrapperClass: type[ForestFixerWrapper]
            match constants.DRIFTING_MODEL:
                case constants.DriftingModel.RandomForest:
                    FixerWrapperClass = ForestFixerWrapper
                case constants.DriftingModel.AdaBoost:
                    FixerWrapperClass = DependentForestFixerWrapper
            return lambda *args, **kwargs: FixerWrapperClass(
                FixerClass,
                *args, **kwargs
                )

__all__ = ["AFixer", "get_fixer"]