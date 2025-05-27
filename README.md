# Diabetes-Analysis
 Diabetes Prediction using K-Nearest Neighbors (KNN)

This project builds a machine learning model to predict the likelihood of diabetes in patients using health metrics such as glucose level, blood pressure, BMI, and more. The model is implemented using the K-Nearest Neighbors (KNN) algorithm with data preprocessing, visualization, and evaluation in Python.

##  Dataset

- **Description**: The dataset contains diagnostic measurements for female patients of Pima Indian heritage aged 21 and above.
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (0 = No diabetes, 1 = Diabetes)

##  Methodology

1. **Data Loading**: Upload and read the CSV file in a Google Colab notebook.
2. **Data Cleaning**:
   - Replace biologically implausible zero values with NaN.
   - Impute missing values using the median.
3. **Visualization**:
   - Distributions of features
   - Correlation heatmap
   - Missing values heatmap
   - Class distribution of target variable
4. **Modeling**:
   - Use `KNeighborsClassifier` with `k=5`.
   - Split data into training and test sets (80/20).
5. **Evaluation**:
   - Accuracy score
   - Classification report (precision, recall, F1-score)
   - Confusion matrix

##  Tools & Libraries

- Python 3
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Google Colab

##  Results

- The model achieved a reasonable level of accuracy using KNN.
- Data imputation and visualization helped enhance model quality and understanding.
- Class imbalance and data distribution were important considerations during analysis.


