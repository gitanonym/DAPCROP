if __name__ == "__main__":
    print("Initializing")

    import sys
    from argparse import ArgumentParser
    import os
    from csv import DictReader

    import pandas as pd
    from scipy import __version__ as scipy_version
    print(f"python version: {sys.version}")
    print(f"scipy version: {scipy_version}")

    from warnings import filterwarnings, simplefilter as warnings_simplefilter
    warnings_simplefilter(action='ignore', category=FutureWarning)
    filterwarnings("ignore", message="The number of unique classes is greater than 50%")

    from Tester import *

    ##### Parser Arguments #####
    parser = ArgumentParser(description="Run all tests")
    parser.add_argument("-o", "--output", type=str, help=f"Output file name prefix, default is the result_{tester_constants.DEFAULT_RESULTS_FILENAME_PREFIX}_<TIMESTAMP>", default=tester_constants.DEFAULT_RESULTS_FILENAME_EXTENDED_PREFIX)
    parser.add_argument("-e", "--exception", action="store_true", help=f"skip exceptions, default is {tester_constants.SKIP_EXCEPTIONS}. If true will write the errors to errors file", default=tester_constants.SKIP_EXCEPTIONS)
    parser.add_argument("-n", "--names", type=str, nargs="+", help="Specific datasets to run, default is all", default=())
    parser.add_argument("-p", "--prefixes", type=str, nargs="+", help="prefixes to datasets to run, default is all. Relevant only if names argument is not provided", default=())
    parser.add_argument("-w", "--repair_window", type=float, nargs="+", help="Repair window sizes, default is all", default=tester_constants.REPAIR_WINDOW_TEST_SIZES)
    parser.add_argument("-d", "--drift_size", type=int, help=f"size of the drift, default is between {tester_constants.MIN_DRIFT_SIZE} and {tester_constants.MAX_DRIFT_SIZE}", default=-1)
    parser.add_argument("-c", "--count", type=int, help="Number of tests to run, default is running all", default=-1)
    parser.add_argument("-t", "--test", type=str, help="Test dataset to run if you want to run a specific test")
    parser.add_argument("-v", "--version", type=int, help=f"Version of the drift synthesizing to use (1=steric or 2=edge values), default is {tester_constants.DRIFT_SYNTHESIZING_VERSION}", default=0)

    args = parser.parse_args()

    print(f"Running tests with {len(tester_constants.diagnosers_output_names)} diagnosers: {tester_constants.diagnosers_output_names}")
    print(f"Running tests with {len(tester_constants.fixers_output_names)} fixers: {tester_constants.fixers_output_names}")

    if args.test:
        print(f"Running single test for {args.test}")
        single_test.sanity_run(file_name=args.test + ".csv", diagnosers_data=tester_constants.diagnosers_data, fixers_data=tester_constants.fixers_data)
        sys.exit(0)

    if args.version:
        assert args.version in (1, 2), f"Unsupported drift synthesizing version: {args.version}. Supported versions are 1 and 2."
        tester_constants.DRIFT_SYNTHESIZING_VERSION = args.version

    specific_datasets_string = ""
    specific_datasets, specific_prefixes = args.names, args.prefixes
    if specific_datasets:
        specific_datasets = tuple(map(str.lower, specific_datasets))
        specific_prefixes = ()
        specific_datasets_string = '-'.join(specific_datasets)
    elif specific_prefixes:
        specific_prefixes = tuple(map(str.lower, specific_prefixes))
        specific_datasets_string = "prefix_" + '-'.join(specific_prefixes)
    if specific_datasets_string:
        print(f"Running tests for {specific_datasets_string}")

    datasets_count = args.count
    if datasets_count > 0:
        print(f"Running tests for the first {datasets_count} datasets")

    skip_exceptions = args.exception
    if skip_exceptions:
        print("Skip exceptions is set to True. Exceptions will be written to errors file.")
    else:
        print("Skip exceptions is set to False. Exceptions will be raised.")

    repair_window_test_sizes, repair_windows_string = tester_constants.REPAIR_WINDOW_TEST_SIZES, ""
    if args.repair_window != tester_constants.REPAIR_WINDOW_TEST_SIZES:
        repair_window_test_sizes = args.repair_window
        repair_windows_string = "-".join(map(str, repair_window_test_sizes))
    repair_window_test_sizes = [round(size, 1) for size in repair_window_test_sizes]
    print(f"Running tests with repair window sizes: {repair_windows_string or 'all'}")

    min_drift_size, max_drift_size, drift_size_string = tester_constants.MIN_DRIFT_SIZE, tester_constants.MAX_DRIFT_SIZE, ""
    if args.drift_size > 0:
        drift_size_string = str(args.drift_size)
        min_drift_size = max_drift_size = args.drift_size
    print(f"Running tests with drift size: {drift_size_string} ({min_drift_size} -> {max_drift_size})")


    ##### Create Results DataFrame #####
    raw_results = pd.DataFrame(columns=tester_constants.RAW_RESULTS_COLUMN_NAMES)
    errors = pd.DataFrame(columns=["name", "error"])


    ###### Run Tests #####
    single_test_function = run_single_test if tester_constants.DRIFT_SYNTHESIZING_VERSION == 1 else run_single_test_v2

    with open(tester_constants.DATASET_DESCRIPTION_FILE_PATH, "r") as descriptions_file:
        descriptions_reader = DictReader(descriptions_file)
        for dataset_description in descriptions_reader:
            if not datasets_count:
                break

            dataset_name = dataset_description["name"]
            
            if specific_datasets and dataset_name.lower() not in specific_datasets:
                continue
            if specific_prefixes and not dataset_name.lower().startswith(specific_prefixes):
                continue
            
            datasets_count -= 1

            print(f"Running tests for {dataset_name}")
            for test_result in single_test_function(tester_constants.DATASETS_DIRECTORY_FULL_PATH, dataset_name, repair_window_test_sizes=repair_window_test_sizes, min_drift_size=min_drift_size, max_drift_size=max_drift_size, diagnosers_data=tester_constants.diagnosers_data, fixers_data=tester_constants.fixers_data):
                if isinstance(test_result, Exception):
                    if not skip_exceptions:
                        raise test_result
                    errors = errors._append({"name": dataset_name, "error": test_result}, ignore_index=True)
                    continue
                raw_results = raw_results._append(test_result, ignore_index=True)


    ##### Prepare Output Files #####
    temp_files_suffix = ""
    if drift_size_string:
        temp_files_suffix += f"_drift_size_{drift_size_string}"
    if repair_windows_string:
        temp_files_suffix += f"_repair_window_{repair_windows_string}"
    if specific_datasets_string:
        temp_files_suffix += f"_{specific_datasets_string}"

    output_files_suffix, output_path = args.output, tester_constants.OUTPUT_DIRECTORY_FULL_PATH
    output_files_suffix += f"_v{tester_constants.DRIFT_SYNTHESIZING_VERSION}"
    if temp_files_suffix:
        output_path = tester_constants.TEMP_OUTPUT_DIRECTORY_FULL_PATH
        if not output_files_suffix.startswith(tester_constants.DEFAULT_RESULTS_FILENAME_PREFIX):
            output_path = f"{output_path}_{output_files_suffix}"
        output_files_suffix = temp_files_suffix
    results_file_name, errors_file_name = f"{tester_constants.RESULTS_FILE_NAME_PREFIX}_{output_files_suffix}.csv", f"{tester_constants.ERRORS_FILE_NAME_PREFIX}_{output_files_suffix}.csv"
    if raw_results.empty and tester_constants.STORE_EMPTY_RESULTS:
        results_file_name = f"{tester_constants.EMPTY_RESULTS_FILE_NAME_PREFIX}_{results_file_name}"

    results_file_path = os.path.join(output_path, results_file_name)
    errors_file_path = os.path.join(output_path, errors_file_name)


    ##### Save Results and Errors #####
    os.makedirs(output_path, exist_ok=True)

    if not raw_results.empty or tester_constants.STORE_EMPTY_RESULTS:
        raw_results.to_csv(results_file_path, index=False)
    elif os.path.exists(results_file_path):
        os.remove(results_file_path)

    if not errors.empty:
        errors.to_csv(errors_file_path, index=False)
    elif os.path.exists(errors_file_path):
        os.remove(errors_file_path)

    print("All tests are done! Wasted-Effort; Correctly-Identified; Accuracy Increase:")
    for baseline_output_name in tester_constants.BASELINES_OUTPUT_NAMES:
        print(f"{baseline_output_name}: xxxx; xxxx; {raw_results[f'{baseline_output_name} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}'].mean():.2f}%")
    print()
    for diagnoser_output_name in tester_constants.diagnosers_output_names:
        diagnoser_diagnosis_results = f"{diagnoser_output_name}: xxxx; xxxx;"
        if diagnoser_output_name != Oracle.__name__:
            diagnoser_diagnosis_results = f"{diagnoser_output_name}: {raw_results[f'{diagnoser_output_name} {tester_constants.WASTED_EFFORT_NAME_SUFFIX}'].mean():.2f}; {raw_results[f'{diagnoser_output_name} {tester_constants.CORRECTLY_IDENTIFIED_NAME_SUFFIX_TEMPLATE.format(k=1)}'].mean():.2f}%;"
        for fixer_output_name in tester_constants.fixers_output_names:
            print(f"{diagnoser_diagnosis_results} {fixer_output_name} {raw_results[f'{diagnoser_output_name}-{fixer_output_name} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}'].mean():.2f}%")

    print()
    print(f"Results saved to {results_file_path}")