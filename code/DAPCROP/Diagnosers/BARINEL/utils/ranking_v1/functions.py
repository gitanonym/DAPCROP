import copy
import sympy

def conflict_directed_search(conflicts):
    if len(conflicts) == 0:
        return []
    diagnoses = []
    new_diagnoses = [[conflicts[0][i]] for i in range(len(conflicts[0]))]
    for conflict in conflicts[1:]:
        diagnoses = new_diagnoses
        new_diagnoses = []
        while len(diagnoses) != 0:
            diagnosis = diagnoses.pop(0)
            intsec = list(set(diagnosis) & set(conflict))
            if len(intsec) == 0:
                new_diags = [diagnosis + [c] for c in conflict]

                def filter_supersets(new_diag):
                    for d in diagnoses + new_diagnoses:
                        if set(d) <= set(new_diag):
                            return False
                    return True

                filtered_new_diags = list(filter(filter_supersets, new_diags))
                new_diagnoses += filtered_new_diags
            else:
                new_diagnoses.append(diagnosis)
    diagnoses = new_diagnoses
    return diagnoses

def estimation_and_derivative_functions(h, spectrum, diagnosis):
    # estimation function
    ef = sympy.Integer(1)
    for row in spectrum:
        row_comp = sympy.Integer(1)
        for fa in diagnosis:
            if row[fa] != 0:
                row_comp = row_comp * h[fa]
        if row[-1] == 1:
            row_comp = 1 - row_comp
        ef = ef * row_comp

    # derivative functions
    DF = []
    for hi, hvar in enumerate(h):
        df = sympy.diff(ef, hvar)
        DF.append(df)
    return ef, DF

def substitute_and_eval(H, lf):
    free_syms = lf.free_symbols
    substitution = {}
    for fs in free_syms:
        substitution[fs] = H[fs.name]
    P = lf.subs(substitution)
    return P

def normalize_diagnoses(ranked_diagnoses):
    normalized_diagnoses = []
    probabilities_sum = 0.0
    for diagnosis in ranked_diagnoses:
        normalized_diagnoses.append(copy.deepcopy(diagnosis))
        probabilities_sum += diagnosis[1]
    for diagnosis in normalized_diagnoses:
        diagnosis[1] = diagnosis[1] / probabilities_sum
    return normalized_diagnoses