import numpy as np
from math import prod
from scipy.optimize import minimize

from DAPCROP import Constants as constants

def get_total_likelihood(diagnosis: np.ndarray,
                         healthiness_probabilities: np.ndarray,
                         spectrum: np.ndarray,
                         fuzzy_error_vector: np.ndarray,
                         use_components_participation: bool = constants.INCLUDE_FUZZY_PARTICIPATION_WHEN_OPTIMIZING
 ) -> float:
    """
    Get the likelihood of the diagnosis.

    Parameters:
    diagnosis (ndarray): The diagnosis.
    healthiness_probabilities (ndarray): The healthiness probabilities.
    spectrum (ndarray): The spectrum.
    fuzzy_error_vector (ndarray): The fuzzy error vector.
    use_components_participation (bool): Whether to include the components participations in the likelihood calculation.

    Returns:
    float: The likelihood of the diagnosis.
    """
    def get_single_test_likelihood(participated_components: np.ndarray,
                                   participation_vector: np.ndarray,
                                   fuzzy_error: float
        ) -> float:
        """"
        Get the likelihood of the single test.
        """
        transaction_goodness = healthiness_probabilities[participated_components].prod()
        if use_components_participation:
            transaction_goodness *= participation_vector[participated_components].prod()
        return fuzzy_error * (1 - transaction_goodness) + (1 - fuzzy_error) * transaction_goodness
    get_diagnosis_participated_components = lambda test_participation_vector: diagnosis[test_participation_vector[diagnosis] > 0]
    tests_diagnosis_components = map(get_diagnosis_participated_components, spectrum)
    tests_likelihoods = map(get_single_test_likelihood, tests_diagnosis_components, spectrum, fuzzy_error_vector)
    return prod(tests_likelihoods)

def rank_diagnosis(diagnosis: np.ndarray,
                   spectrum: np.ndarray,
                   fuzzy_error_vector: np.ndarray,
                   components_prior_probabilities: np.ndarray
 ) -> float:
    """
    Rank the diagnosis.

    Parameters:
    diagnosis (ndarray): The diagnosis.
    spectrum (ndarray): The spectrum.
    fuzzy_error_vector (ndarray): The fuzzy error vector.

    Returns:
        float: The rank of the diagnosis.
    """
    components_count = spectrum.shape[1]
    components_prior_probabilities = components_prior_probabilities.copy()
    vectorized_flip_probability = np.vectorize(lambda spectrum_index, probability: probability if spectrum_index in diagnosis else 1 - probability)
    components_prior_probabilities = vectorized_flip_probability(np.arange(components_count), components_prior_probabilities)
    prior_probability = components_prior_probabilities.prod()
    healthiness_probabilities = np.full(components_count, 0.5)
    healthiness_bounds = [(0, 1) for _ in range(components_count)]
    likelihood_objective_function = lambda healthiness_probabilities: -get_total_likelihood(diagnosis, healthiness_probabilities, spectrum, fuzzy_error_vector)  # Maximize the likelihood
    mle_model = minimize(likelihood_objective_function, healthiness_probabilities, bounds=healthiness_bounds, options={"maxiter": 1000})    # FIXME: Error in cluster
    optimized_healthiness_probabilities = mle_model.x
    assert ((0 <= optimized_healthiness_probabilities) & (optimized_healthiness_probabilities <= 1)).all(), f"Not all optimized healthiness probabilities are within [0, 1] (min: {optimized_healthiness_probabilities.min()}, max: {optimized_healthiness_probabilities.max()})"
    return prior_probability * get_total_likelihood(diagnosis, optimized_healthiness_probabilities, spectrum, fuzzy_error_vector, use_components_participation=True)

def rank_diagnoses(spectrum: np.ndarray,
                   diagnoses: list[np.ndarray],
                   components_prior_probabilities: np.ndarray = None
 ) -> list[tuple[np.ndarray, float]]:
    """
    Rank the diagnoses.

    Parameters:
        spectrum (ndarray): The spectrum.
        diagnoses (list[ndarray]): The diagnoses.
        fuzzy_error_vector (ndarray): The fuzzy error vector.

    Returns:
        list[tuple[ndarray, float]]: The ranked diagnoses.
    """
    spectrum, fuzzy_error_vector = spectrum[:, :-1], spectrum[:, -1]
    if components_prior_probabilities is None:
        components_prior_probabilities = np.full(spectrum.shape[1], constants.BARINEL_COMPONENT_PRIOR_PROBABILITY)
    return [(diagnosis, rank_diagnosis(diagnosis, spectrum, fuzzy_error_vector, components_prior_probabilities)) for diagnosis in diagnoses]