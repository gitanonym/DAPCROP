import copy
import sympy

from . import functions

def estimation_and_derivative_functions(diagnosis, spectrum):
    # declare variables
    h = []
    for hj in range(len(spectrum[0][:-1])):
        h.append(sympy.symbols(f'h{hj}'))
    ef, DF = functions.estimation_and_derivative_functions(h, spectrum, diagnosis)
    return ef, DF, h

def eval_grad_R0(diagnosis, H, DF):
    Gradients = {}
    for a in diagnosis:
        gradient_value = functions.substitute_and_eval(H, DF[a])
        Gradients[f'h{a}'] = float(gradient_value)
    return Gradients

def rank_diagnoses(spectrum, diagnoses, step):
    """
    ranks the diagnoses. for each diagnosis, the diagnoser
    computes a corresponding estimation function
    and then maximizes it
    Parameters:
        spectrum (ndarray): The spectrum.
        diagnoses (ndarray): The diagnoses.
        step (float): the gradient step
    Returns:
        list[list[int]]: ranked diagnosis list.
    """
    ranked_diagnoses = []
    for diagnosis in diagnoses:
        # initialize H values of the agents involved in the diagnosis to 0.5
        # initialize an epsilon value for the stop condition: |P_n(d, H, M) - P_{n-1}(d, H, M)| < epsilon
        # initialize P_{n-1}(d, H, M) to zero
        # create symbolic local estimation function (E)
        # create symbolic local derivative function (D)
        # while true
        #   calculate P_n(d, H, M)
        #   if condition is is reached, abort
        #   calculate gradients
        #   update H

        # initialize H values of the agents involved in the diagnosis to 0.5
        H = {}
        for a in diagnosis:
            H[f'h{a}'] = 0.5

        # initialize an epsilon value for the stop condition: |P_n(d, H, LS) - P_{n-1}(d, H, LS)| < epsilon
        epsilon = 0.0005

        # initialize P_{n-1}(d, H, LS) to zero
        P_arr = [[0.0, {}]]

        # create symbolic local estimation function (E)
        # create symbolic local derivative function (D)
        ef, DF, h = estimation_and_derivative_functions(diagnosis, spectrum)

        # while true
        for i in range(1000):
            # calculate P_n(d, H, S)
            P = functions.substitute_and_eval(H, ef)
            P = float(P)
            P_arr.append([P, copy.deepcopy(H)])
            # if condition is reached, abort
            likelihood = P_arr[-1][0]
            if abs(P_arr[-1][0] - P_arr[-2][0]) < epsilon:
                H = P_arr[-1][1]
                break
            if P_arr[-1][0] > 1.0:
                likelihood = P_arr[-2][0]
                H = P_arr[-2][1]
                break
            # calculate gradients
            Gradients = eval_grad_R0(diagnosis, H, DF)
            # update H
            number_of_agents = len(spectrum[0][:-1])
            H, _ = update_h(H, Gradients, step, number_of_agents)

        ranked_diagnoses.append([diagnosis, likelihood, H])

    # normalize the diagnosis probabilities
    normalized_diagnoses = functions.normalize_diagnoses(ranked_diagnoses)
    return normalized_diagnoses

def update_h(H, Gradients, step, number_of_agents):
    information_sent_update_h = 0
    for key in Gradients.keys():
        information_sent_update_h += number_of_agents - 1
        if H[key] + step * Gradients[key] > 1.0:
            H[key] = 1.0
        elif H[key] + step * Gradients[key] < 0.0:
            H[key] = 0.0
        else:
            H[key] = H[key] + step * Gradients[key]
    return H, information_sent_update_h