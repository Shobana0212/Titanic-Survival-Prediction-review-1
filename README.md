# Titanic Survival Prediction using Machine Learning

- Group Name: Team Vayu
- Group Leader Name: Shobana Balusamy
- Group Leader Registration Number: 23MID0212
- Students Name: Mohanraj I
- Students Registration Number: 23MID0121

A beginner-friendly supervised machine-learning project that predicts whether a Titanic passenger survived based on passenger information such as class, sex, age, fare, family relationships, and port of embarkation.

## Project Overview

This project uses the Kaggle **Titanic: Machine Learning from Disaster** dataset.

- `train.csv`: 891 passenger records with the `Survived` target label.
- `test.csv`: 418 passenger records without the target label.
- Target: `Survived` — `0` means did not survive; `1` means survived.
- Initial algorithm: Logistic Regression.
- Evaluation: validation accuracy, classification report, and confusion matrix.
- Final output: `submission.csv`, formatted for Kaggle.

This is an initial baseline model. Results should be reported only after running the notebook.

## Project Structure

Keep these files in the same folder:

```text
titanic-survival-prediction/
├── Titanic_Survival_Prediction_Initial_Model.ipynb
├── train.csv
├── test.csv
├── README.md
├── submission.csv                              # Generated after running the notebook
└── titanic_logistic_regression_pipeline.joblib # Optional saved model
```

## Requirements

- Python 3.9 or later recommended
- Jupyter Notebook or JupyterLab
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- joblib (for saving the fitted model)

## Setup

### 1. Download the dataset

Download `train.csv` and `test.csv` from the official Kaggle competition page:

https://www.kaggle.com/c/titanic/data

Place both CSV files in the same directory as the notebook.

### 2. Install dependencies

Open a terminal in the project directory and run:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib notebook
```

If your system uses `python3` instead of `python`, run:

```bash
python3 -m pip install pandas numpy matplotlib seaborn scikit-learn joblib notebook
```

### 3. Start Jupyter

```bash
jupyter notebook
```

Open `Titanic_Survival_Prediction_Initial_Model.ipynb` in your browser.

### 4. Run the notebook

Run each cell from top to bottom using **Shift + Enter**.

The notebook loads the data, explores missing values and survival patterns, engineers a few features, trains a Logistic Regression pipeline, and evaluates it on a held-out validation set.

## Features Used

The initial model uses:

- `Pclass` — passenger class
- `Sex` — passenger sex
- `Age` — passenger age
- `SibSp` — siblings/spouses aboard
- `Parch` — parents/children aboard
- `Fare` — ticket fare
- `Embarked` — port of embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton)
- `FamilySize` — `SibSp + Parch + 1`
- `IsAlone` — whether the passenger travelled alone according to the family fields
- `Title` — a title extracted from the passenger name

The pipeline imputes missing numerical values with medians, fills missing categorical values with the most frequent category, standardises numerical fields, and one-hot encodes categorical fields. Preprocessing is fitted inside the model pipeline to help prevent data leakage during validation.

## Model and Evaluation

The first model is **Logistic Regression**. The labelled data is split into training and validation sets using a stratified split, with 20% reserved for validation.

The notebook reports:

- Validation accuracy
- Precision, recall, and F1-score
- Confusion matrix
- Logistic Regression coefficients (when supported by the installed scikit-learn version)

The validation score estimates performance on unseen data. It is **not** the Kaggle leaderboard score.

## Output

After the notebook runs successfully, it creates:

### `submission.csv`

This file contains 418 rows and two columns:

```text
PassengerId,Survived
```

The `Survived` column contains predictions of `0` or `1`. Upload this file to the Kaggle Titanic competition to obtain a leaderboard result.

### `titanic_logistic_regression_pipeline.joblib`

An optional saved copy of the fitted preprocessing and model pipeline.

## Troubleshooting

- **`FileNotFoundError`:** Check that `train.csv` and `test.csv` are in the notebook's working directory, or update the paths in the data-loading cell.
- **`ModuleNotFoundError`:** Install the missing dependency using `python -m pip install <package-name>`.
- **Unexpected columns or shapes:** Confirm that you downloaded the official Kaggle competition files.
- **Different results:** Results may vary if the data, random seed, library versions, or feature engineering are changed.

## Limitations

- The dataset is relatively small and contains missing values.
- Model performance depends on feature choices and validation strategy.
- Predictions reflect patterns in the recorded historical data and do not prove causation.
- No accuracy or leaderboard score is claimed in this README; record actual results after running the notebook.

## Future Improvements

1. Compare Logistic Regression with Decision Tree and Random Forest.
2. Use stratified cross-validation and tune model hyperparameters.
3. Evaluate additional features such as ticket groups or cabin information where appropriate.
4. Compare validation metrics consistently and document the selected model.
5. Update this README and the project report with measured results.

## References

- Kaggle — Titanic: Machine Learning from Disaster: https://www.kaggle.com/c/titanic
- scikit-learn — Imputation of missing values: https://scikit-learn.org/stable/modules/impute.html
- scikit-learn — OneHotEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

## License and Dataset Use

The dataset is provided through Kaggle. Review the competition page and dataset terms before redistributing the data. This README and notebook describe an educational project workflow.
