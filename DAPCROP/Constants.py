from enum import Enum, auto

from sys import float_info
EPSILON = float_info.epsilon

class FeatureType(Enum):
    Numeric = auto()
    Categorical = auto()
    Binary = auto()
    Mixed = auto()

class NodeIndexType(Enum):
    COMPONENT_INDEX = auto()
    SPECTRA_INDEX = auto()

class DriftingModel(Enum):
    DecisionTree = auto()
    RandomForest = auto()
    AdaBoost = auto()

DRIFTING_MODEL = DriftingModel.AdaBoost

class DriftingModelType(Enum):
    Tree = auto()
    Forest = auto()

DRIFTING_MODEL_TYPE: DriftingModelType
match DRIFTING_MODEL:
    case DriftingModel.DecisionTree:
        DRIFTING_MODEL_TYPE = DriftingModelType.Tree
    case DriftingModel.RandomForest | DriftingModel.AdaBoost:
        DRIFTING_MODEL_TYPE = DriftingModelType.Forest

DISTRIBUTE_DIAGNOSES_COMPUTATION = True

# Random state
RANDOM_STATE = 7

BARINEL_COMPONENT_PRIOR_PROBABILITY = 1 / 1000

# BARINEL ranking v1
BARINEL_GRADIENT_STEP = 0.5

# BARINEL ranking v2
INCLUDE_FUZZY_PARTICIPATION_WHEN_OPTIMIZING = True

# Choose the BARINEL ranking algorithm
# V1: discrete error ranking algorithm taken from DDIFMAS
# V2: new ranking algorithm, supporting fuzzy error and participation matrix, and custom prior probabilities
BARINEL_RANKING_ALGORITHM_VERSION = 2


DEFAULT_COMBINE_STAT = True # Combine STAT Diagnoses
DEFAULT_USE_SHAP_CONTRIBUTION = False # Try to use SHAP contributions for the participation
DEFAULT_AGGREGATE_TESTS_BY_PATHS = False # Aggregate tests by paths in the decision tree
DEFAULT_GROUP_FEATURE_NODES = False # Group feature nodes to a single component
DEFAULT_COMBINE_PRIOR_CONFIDENCE = False # Use the prior confidence in Error vector calculation
DEFAULT_COMBINE_COMPONENTS_DEPTH = False # Include the components depth in the components' participations

BARINEL_THRESHOLD_ABOVE_STD_RATE = 0.5 # setting the error threshold (from which tests considered as failed) for mean + std * BARINEL_THRESHOLD_ABOVE_STD_RATE

BARINEL_ADD_TARGET_TO_FEATURE_COMPONENTS = True # Whether to add the target as component if we're merging nodes based on the feature

class SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES(Enum): #
    Take_Top = auto() # Do not handle dependencies
    Take_Bottom = auto() # Replace also all dependent subtrees
    Replace_Ancestors = auto() # Replace also all ancestor subtrees

DEFAULT_SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPE = SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES.Replace_Ancestors

if DEFAULT_SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPE == SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES.Take_Bottom:
    raise NotImplementedError("TAKE_BOTTOM handling type is not implemented yet.")

class PRIOR_KNOWLEDGE_USAGE_TYPES(Enum): #
    Use = auto() # Use prior knowledge when fitting
    Ignore = auto() # Ignore prior knowledge when fitting
    Synthesize = auto() # Synthesize prior knowledge data when fitting

DEFAULT_USE_OF_PRIOR_KNOWLEDGE = PRIOR_KNOWLEDGE_USAGE_TYPES.Synthesize

class SubTreeType(Enum):
    Original = auto()
    ExtremelyFastDecisionTreeClassifier = auto()
    HoeffdingAdaptiveTreeClassifier = auto()

DEFAULT_SUBTREE_TYPE = SubTreeType.HoeffdingAdaptiveTreeClassifier

# Distributing Computation
DISTRIBUTE_DIAGNOSES_COMPUTATION = True
DISTRIBUTE_FIXING_COMPUTATION = False

MINIMUM_DRIFT_ACCURACY_DROP = 0.1

SYNTHESIZE_BY_BOOTSTRAP = False

SFLDT_ALLOW_UNAFFECTED_MODELS = True
