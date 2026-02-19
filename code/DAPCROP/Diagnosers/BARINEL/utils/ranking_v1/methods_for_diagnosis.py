from . import functions

def get_candidate_diagnoses(spectrum: list[list[int]],
                            error_threshold:float = 1
 ) -> list[list[int]]:
    """
    the traditional hitting set algorithm
    
    Parameters:
        spectrum (list[list[int]]): The spectrum. rows are tests, where the last item in each row is the test error.
        error_threshold (float): The error threshold. Default is 1.

    Returns:
        list[list[int]]: The candidates.
    """
    # calculate conflicts
    conflicts = []
    for i, row in enumerate(spectrum):
        if row[-1] >= error_threshold:
            conf = [j for j, a in enumerate(row[:-1]) if a != 0]
            conflicts.append(conf)

    # compute diagnoses
    diagnoses = functions.conflict_directed_search(conflicts)

    # sort diagnoses
    for d in diagnoses:
        d.sort()
    return diagnoses