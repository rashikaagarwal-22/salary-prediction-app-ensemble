#Salary Prediction Project

## Project Description
This project aims to predict an individual's salary based on various features such as age, gender, education level, job title, and years of experience. A machine learning model, specifically a Random Forest Regressor, is trained and evaluated to achieve this prediction.

## Setup Instructions
To run this project, you'll need a Python environment, preferably Google Colab, due to its seamless integration with Kaggle datasets. Follow these steps:

1.  **Install Kaggle API client:**
    ```bash
    !pip install -q kaggle
    ```
2.  **Download the dataset using `kagglehub`:**
    ```python
    import kagglehub
    path = kagglehub.dataset_download("rkiattisak/salaly-prediction-for-beginer")
    print("Path to dataset files:", path)
    ```
3.  **Ensure all necessary libraries are installed.** The following libraries are used in this project:
    *   `pandas`
    *   `numpy`
    *   `matplotlib`
    *   `seaborn`
    *   `sklearn`
    *   `kagglehub`

    (Most of these are pre-installed in Google Colab environments. If any are missing, you can install them using `!pip install <library_name>`.)

## Dataset Information
The dataset used for this project is titled "Salary Data.csv" from Kaggle (rkiattisak/salaly-prediction-for-beginer). It contains the following key columns:

*   **Age**: The age of the individual.
*   **Gender**: The gender of the individual (Male/Female).
*   **Education Level**: The highest education level achieved (e.g., Bachelor's, Master's, PhD).
*   **Job Title**: The professional role of the individual.
*   **Years of Experience**: The number of years of professional experience.
*   **Salary**: The target variable, representing the individual's annual salary.

## Analysis Breakdown

### Data Loading
The dataset was loaded into a pandas DataFrame using `pd.read_csv()` after identifying the CSV file within the downloaded Kaggle dataset path.

### Exploratory Data Analysis (EDA)
*   **Shape and Info**: Initial checks of `df.shape` and `df.info()` were performed to understand the dataset's dimensions and data types, revealing 375 entries and 6 columns, with some non-null values.
*   **Null Values**: `df.isnull().sum()` revealed 2 missing values across all columns.
*   **Visualizations**:
    *   A box plot of 'Salary' (`df.boxplot(column='Salary')`) was used to visualize the distribution and identify potential outliers.
    *   A scatter plot of 'Salary' vs. 'Years of Experience' was created to observe the relationship between these two key numerical features.
    *   A correlation heatmap was generated for numerical features to understand their linear relationships.

### Data Preprocessing
*   **Missing Value Imputation**: Missing values in numerical columns ('Age', 'Years of Experience', 'Salary') were imputed using their respective medians to maintain data distribution integrity.
*   **Categorical Feature Encoding**: Categorical features ('Gender', 'Education Level', 'Job Title') were converted into numerical format using one-hot encoding with `pd.get_dummies(df, columns=[...], drop_first=True)` to avoid multicollinearity.

### Feature Scaling
Numerical features `Age` and `Years of Experience` were scaled using `StandardScaler` from `sklearn.preprocessing`. This standardizes the features by removing the mean and scaling to unit variance, which is crucial for many machine learning algorithms.

### Model Training
*   **Model Selection**: A Random Forest Regressor (`RandomForestRegressor`) was chosen for its robustness and ability to handle complex relationships.
*   **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets using `train_test_split` with `random_state=42` for reproducibility. Features (`X`) included all columns except 'Salary', which was set as the target variable (`y`).
*   **Model Initialization**: The model was initialized with `n_estimators=100` and `random_state=42`.
*   **Training**: The model was trained on the `X_train` and `y_train` datasets.

### Model Evaluation
*   **Initial Evaluation**: The model was evaluated on the `X_test` set, yielding:
    *   **Mean Absolute Error (MAE):** `10354.0`
    *   **R-squared Score (R2):** `0.8928`
    *   This indicates the model explains about 89.3% of the variance in salary with an average prediction error of approximately $10,354.
*   **Cross-Validation**: K-Fold Cross-Validation (5 folds) was performed to assess model robustness:
    *   **Mean R-squared across folds:** `0.8868 (+/- 0.0052)`
    *   **Mean MAE across folds:** `10084.72 (+/- 986.10)`
    *   Cross-validation results confirmed the stability and generalizability of the model's performance.

### Feature Importance Analysis
An analysis of feature importances from the trained Random Forest model revealed:
*   **Age**: Approximately `0.588`
*   **Years of Experience**: Approximately `0.315`
*   **Education Level_PhD**: Approximately `0.011`
*   **Education Level_Master's**: Approximately `0.009`
*   Other `Job Title` categories had smaller, distributed importances.

This highlights that `Age` and `Years of Experience` are by far the most significant predictors of salary.

### Error Analysis
*   **Largest Errors**: Examination of the top 10 largest absolute errors showed a maximum error of approximately $77,750, suggesting specific instances where the model's predictions were significantly off.
*   **Error Distribution**: A histogram of absolute prediction errors indicated that while most predictions were accurate, a long tail represented a few instances with large discrepancies.
*   **Analysis of High-Error Instances**: Further investigation into the features of these high-error points (e.g., specific age, experience, education, job title combinations) was conducted to understand potential underlying causes.

## Libraries Used
*   **`pandas`**: For data manipulation and analysis.
*   **`numpy`**: For numerical operations, especially with arrays.
*   **`matplotlib.pyplot`**: For creating static, interactive, and animated visualizations.
*   **`seaborn`**: For statistical data visualization, built on Matplotlib.
*   **`sklearn` (scikit-learn)**: For machine learning tasks, including preprocessing (`StandardScaler`, `train_test_split`), model building (`RandomForestRegressor`), and evaluation metrics (`mean_absolute_error`, `r2_score`, `KFold`, `cross_val_score`).
*   **`kagglehub`**: For downloading datasets from Kaggle.

## Model Performance Summary
The Random Forest Regressor model effectively predicts salaries, achieving an R-squared score of approximately 0.89 (both on the test set and averaged across cross-validation folds) and a Mean Absolute Error (MAE) of around $10,354. The consistency between initial evaluation and cross-validation suggests a robust and generalizable model. `Age` and `Years of Experience` are the dominant factors influencing salary predictions. While the model performs well overall, the error analysis identified specific cases with high prediction discrepancies, indicating potential areas for further model refinement or deeper data investigation, possibly involving more advanced feature engineering or different modeling approaches for these outlier cases.
