# Heart Disease Classification Using Machine Learning

This project involves predicting the presence of heart disease based on various medical attributes. It uses several machine learning models to classify whether a person has heart disease or not, based on features such as age, sex, cholesterol levels, resting blood pressure, and more.

## Objective

The goal of this project is to use machine learning algorithms to predict heart disease risk, which can aid in early detection and prevention. The project involves data cleaning, visualization, and the application of several classification models, including Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Support Vector Machine (SVM). The models' performances are compared, and hyperparameter tuning is used to enhance the SVM model.

## Technologies Used

- **Python**: The primary programming language used for data processing and model building.
- **Scikit-learn**: A machine learning library used for model implementation, training, and evaluation.
- **Pandas**: For data manipulation and cleaning.
- **Seaborn & Matplotlib**: For data visualization.
- **Jupyter Notebook**: Used for interactive development and analysis.

## Dataset

The dataset used for this project is the **Heart Disease Dataset**, available on [Kaggle](https://www.kaggle.com/datasets/mexwell/heart-disease-dataset/data). The dataset contains various attributes related to a patient's health, such as cholesterol, blood pressure, and age, which are used to predict the likelihood of heart disease.

## Steps Involved

1. **Data Preprocessing**:
   - Loading and cleaning the data.
   - Handling missing values and outliers.
   - Feature selection and extraction.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizing the relationships between features.
   - Generating histograms, pairplots, and heatmaps to understand feature distributions and correlations.

3. **Modeling**:
   - Splitting the dataset into training and testing sets.
   - Training multiple machine learning models: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Support Vector Machine (SVM).
   - Evaluating each model's performance using metrics such as accuracy, precision, recall, and F1-score.

4. **Hyperparameter Tuning**:
   - Using GridSearchCV to tune the hyperparameters of the SVM model for improved performance.

5. **Model Comparison**:
   - Comparing the accuracy scores of all models to determine the best performer.

## Results

- The **Random Forest Classifier** showed the best performance with the highest accuracy, followed by **Decision Tree Classifier** and **Logistic Regression**.
- Hyperparameter tuning with **GridSearchCV** improved the performance of the **SVM** model.
- The final model recommendation is the **Random Forest Classifier** based on its accuracy.
