from DAPCROP import Constants as constants

from .ranking_v1 import (
    get_candidate_diagnoses as _get_candidate_diagnoses_v1,
    rank_diagnoses as _rank_diagnoses_v1
)


from .ranking_v2 import (
    rank_diagnoses as _rank_diagnoses_v2
)

# Wrapper Functions:
get_candidate_diagnoses = _get_candidate_diagnoses_v1

assert constants.BARINEL_RANKING_ALGORITHM_VERSION in range(1, 3)
def rank_diagnoses(spectrum, diagnoses, components_prior_probabilities=None):
    """
    Rank the diagnoses.

    Parameters:
        spectrum (ndarray): The spectrum.
        diagnoses (ndarray): The diagnoses.
        fuzzy_error_vector (ndarray): The fuzzy error vector.

    Returns:
        ndarray: The ranked diagnoses.
    """
    if constants.BARINEL_RANKING_ALGORITHM_VERSION == 1:
        return _rank_diagnoses_v1(spectrum, diagnoses, constants.BARINEL_GRADIENT_STEP)
    return _rank_diagnoses_v2(spectrum, diagnoses, components_prior_probabilities)


__all__ = ["get_candidate_diagnoses", "rank_diagnoses"]