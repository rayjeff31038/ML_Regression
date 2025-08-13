# ML_Regression
This project demonstrates how to build a regression machine learning model using the Scikit-learn package.  
The raw data (`Sleep_health_and_lifestyle_dataset.csv`) was downloaded from Kaggle:  
[Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/data)

This dataset contains information about sleep quality and various lifestyle factors.

## Project File Descriptions
1. Sleep_health_and_lifestyle_dataset.csv – The raw dataset containing sleep health and lifestyle information.

2. model_training.py – Python script containing the regression model training process.

3. Visualization PNG files – Plots generated from exploratory data analysis (EDA) and saved as .png images.

4. model_results.csv – A DataFrame summarizing the performance of each regression model.

5. RandomForestRegressor_best_model.pkl – The serialized best-performing model (RandomForestRegressor) saved using joblib.



## Aim
The goal of this project is to predict sleep quality based on lifestyle factors using regression models. 


## Data Description
Person ID: Unique identifier for each individual.

Gender: The gender of the person (Male/Female).

Age: The person’s age in years.

Occupation: The occupation or profession of the person.

Sleep Duration (hours): Average number of hours the person sleeps per day.

Quality of Sleep (scale: 1–10): Self-reported quality of sleep, rated from 1 (lowest) to 10 (highest).

Physical Activity Level (minutes/day): Number of minutes per day spent on physical activities.

Stress Level (scale: 1–10): Self-reported stress level, rated from 1 (lowest) to 10 (highest).

BMI Category: The body mass index category of the person (e.g., Underweight, Normal, Overweight).

Blood Pressure (systolic/diastolic): Blood pressure reading, expressed as systolic pressure over diastolic pressure.

Heart Rate (bpm): Resting heart rate in beats per minute.

Daily Steps: Number of steps taken per day.

Sleep Disorder: Type of sleep disorder present, if any (None, Insomnia, Sleep Apnea).


## Analysis Workflow
1. Data Exploring
   -Examine the data structure, fill in missing values, and use visualizations to observe the data distribution and correlations. Then perform preprocessing to organize the data into a usable format.
     
2. Data Preprocessing
   -Based on the EDA results, perform transformations on the dataset’s columns — for example, standardize numerical columns, and convert categorical columns into numerical format using OneHotEncoder or OrdinalEncoder.
   
3. Model training
   -Multiple regression models will be trained and compared to evaluate their performance.
The models include linear regression approaches suitable for capturing linear relationships in data (Linear Regression, Ridge Regression, Lasso Regression, and ElasticNet Regression) and non-linear regression approaches that can model complex, non-linear patterns (SVR, RandomForestRegressor, and GradientBoostingRegressor).

4. Model evaluation
   -The best-performing model will be determined by comparing R² score, mean squared error (MSE), and mean absolute error (MAE). These metrics respectively evaluate the model’s explanatory power, the magnitude of large prediction errors, and the average prediction error, providing a comprehensive assessment of model performance.
