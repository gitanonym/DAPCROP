# Explainable Adaptation to Concept Drift in Decision Tree Ensembles

Decision tree ensembles (also known as Forests) are widely utilized in machine learning for their high predictive accuracy and robustness. In environments where the underlying data distribution evolves—referred to as concept drift—the resulting misalignment causes model performance to degrade significantly. While the standard response involves retraining a new ensemble from scratch, treating the model as a nontransparent system limits explainability and prevents a rigorous, explanation-oriented analysis of the latent processes responsible for the observed distributional shift. In this paper, we propose a novel two-step framework for managing concept drift in decision tree ensemble models, applicable to both bagging and boosting architectures where the base estimators are decision trees. Our approach first applies model-based diagnosis, a well-established methodology in Knowledge Representation (KR), to identify and explain which features are responsible for the drift, followed by a targeted repair phase. We introduce two diagnosis algorithms and two repair mechanisms designed to adapt the existing estimators within the ensemble to the evolving data distribution. Evaluation using Random Forest (bagging) and AdaBoost (boosting) demonstrates that our diagnosis-and-repair framework outperforms traditional retraining strategies based on either post-drift data alone or the entire dataset.

## This Repository is listed as:
- DAPCROP - The algorithm, with baselines in it.
- Tester - All the files relevant to evaluate the algorithm, including configurations for each baseline diagnoser that ran.
- Data - The datasets with their descriptions.
- appendix (duplicated as results) - The results for the AdaBoost and RandomForest runs.

## Reproduce:
To run the algorithms, all needed is:
1. Install the required packages using [the requirements file](https://github.com/gitanonym/DAPCROP/blob/main/requirements.txt).
2. Choose your desired model [in the constatns file by choosing an enum value](https://github.com/gitanonym/DAPCROP/blob/main/code/DAPCROP/Constants.py#L21).

That's it. The results will be saved in the results folder.


[Appendix of all the datasets and corresponding results can be located here]([https://github.com/my-anonymous-git/spider_fuzzy_diagnosis/blob/main/data/all_datasets.csv](https://github.com/gitanonym/DAPCROP/tree/main/appendix))
    Good Luck!
