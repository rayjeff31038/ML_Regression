###############################################################
# Project Description
#
#
#
#
#
###############################################################

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import make_column_transformer

#model import
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

#figure
import seaborn as sns
import matplotlib.pyplot as plt

############################
#Step1. data preprocessing
#############################
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
print(df.head(10),'\n')

#drop unnecessary columns
df = df.drop(columns = ['Person ID'])
print(df.dtypes, '\n')

'''
Gender                      object
Age                          int64
Occupation                  object
Sleep Duration             float64
Quality of Sleep             int64
Physical Activity Level      int64
Stress Level                 int64
BMI Category                object
Blood Pressure              object
Heart Rate                   int64
Daily Steps                  int64
Sleep Disorder              object
'''

# Object columns: Gender, Occupation, BMI Category,Blood Pressure,Sleep Disorder
# Data transformation is required for those columns before model building
# Strategy for data transformation:
# Gender, Occupation --> OneHotEncoder
# BMI Category, Sleep Disorder --> OrdinalEncoder
# Blood Pressure --> Add a new coulmn "Hypertension" for high blood pressure


# Split dataframe to train_df and test_df
# X = all the factors 
# y = 'Quality of Sleep'
X = df.drop(columns = ['Quality of Sleep'])
y = df['Quality of Sleep']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


#check the missing values in all columns
X_train_missing_value_columns = X_train.columns[X_train.isnull().any()].to_list()
X_test_missing_value_columns = X_test.columns[X_test.isnull().any()].to_list()
y_train_missing_value = y_train.isnull().any()
y_test_missing_value = y_test.isnull().any()

print(f'X_train_missing_value_columns:{X_train_missing_value_columns}')
print(f'X_test_missing_value_columns:{X_test_missing_value_columns}','\n')
print(f'y_train_missing_value :{y_train_missing_value}')
print(f'y_test_missing_value:{y_test_missing_value}')
#X_train_missing_value_columns:['Sleep Disorder']
#X_test_missing_value_columns:['Sleep Disorder']
#y_train_missing_value :False
#y_test_missing_value:False

X_train['Sleep Disorder'] = X_train['Sleep Disorder'].fillna('no')
X_test['Sleep Disorder'] = X_test['Sleep Disorder'].fillna('no')
#all the missing value have been fixed, ready for preprocessing data


#processimg the 'BMI Category' column
print(X_train['BMI Category'].unique().tolist())
#['Overweight', 'Normal', 'Obese', 'Normal Weight']
print(X_test['BMI Category'].unique().tolist())
#['Normal', 'Overweight', 'Normal Weight', 'Obese']

# Normal and Normal Weight mean the same
# replace 'Normal Weight' to 'Normal'
X_train['BMI Category'] = X_train['BMI Category'].replace('Normal Weight','Normal')
X_test['BMI Category'] = X_test['BMI Category'].replace('Normal Weight','Normal')


# preprocessing the 'Blood Pressure' column
# Blood pressure data requires further transformation into systolic and diastolic values
# And add a new column indicating "Hypertension" column
# Criterion: systolic pressure >= 140 or diastolic pressure >= 90 (mmHg)
print(X_train['Blood Pressure'].head(10))
print(X_test['Blood Pressure'].head(10))
#data format: 140/95

X_train['Systolic'] = X_train['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
X_train['Diastolic'] = X_train['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))
X_train['Hypertension'] = np.where((X_train['Systolic'] >= 140) | (X_train['Diastolic'] >=90),1,0)
print(X_train['Hypertension'])

X_test['Systolic'] = X_test['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
X_test['Diastolic'] = X_test['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))
X_test['Hypertension'] = np.where((X_test['Systolic'] >= 140) | (X_test['Diastolic'] >=90),1,0)
print(X_test['Hypertension'])


# preprocessing the 'Sleep Disorder' column
print(X_train['Sleep Disorder'].unique().tolist())
#['Sleep Apnea', 'Insomnia', 'no']
print(X_test['Sleep Disorder'].unique().tolist())
#['Insomnia', 'no', 'Sleep Apnea']

X_train = X_train.drop(columns = ['Blood Pressure'])
print(X_train.head(10))
X_test = X_test.drop(columns = ['Blood Pressure'])
print(X_test.head(10))
#all the columns has been ready to do the data transformation


############################
#Step2. data transformation
############################
# Gender, Occupation, Hypertension--> OneHotEncoder
# BMI Category, Sleep Disorder --> OrdinalEncoder
# other factors  --> StandardScaler

onehot_columns = ['Gender', 'Occupation']

BMI_Category_rank = ['Normal','Overweight','Obese']

#According to severity, it can be ranked as Sleep Apnea > Insomnia > no
Sleep_Disorder_rank = ['no','Insomnia','Sleep Apnea']


print(X_train.columns)
'''
Index(['Gender', 'Age', 'Occupation', 'Sleep Duration',
       'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate',
       'Daily Steps', 'Sleep Disorder', 'Systolic', 'Diastolic',
       'Hypertension']
'''
standardization_columns = [
    'Age',
    'Sleep Duration',
    'Physical Activity Level',
    'Stress Level',
    'Heart Rate',
    'Daily Steps',
    'Systolic',
    'Diastolic',
]

mct = make_column_transformer(
    (OneHotEncoder(handle_unknown = 'ignore'), onehot_columns),
    (OrdinalEncoder(categories = [BMI_Category_rank]), ['BMI Category']),
    (OrdinalEncoder(categories =[Sleep_Disorder_rank]), ['Sleep Disorder']),
    (StandardScaler(), standardization_columns),
    remainder = 'passthrough',
    n_jobs = -1
)

#check the correction between X and y
#feature_names = mct.get_feature_names_out().tolist() + ['Hypertension']
#figure_df = pd.DataFrame(X_train, columns = feature_names)
#print(figure_df)

'''
for column in figure_df.columns:
    plt.figure(figsize=(6,4))
    x1 = figure_df[column]
    y1 = y_train
    sns.scatterplot(x = x1, y = y1)
    plt.title(f'{column}')
    plt.show()
'''


############################
#Step3. Model training
############################

#LinearRegression model
lr = LinearRegression()

lr_param_grid = {
    'fit_intercept':[True, False],
    'positive': [True, False]
}

lr_CV = GridSearchCV(lr, lr_param_grid, cv =3, n_jobs=-1)

lr_CV_pipeline = make_pipeline(mct, lr_CV)
lr_CV_pipeline.fit(X_train, y_train)
print(lr_CV.best_params_)
print(lr_CV.best_score_)


# Lasso Regression
# to turn some coefficients to zero.

lasso_r = Lasso()

lasso_r_param_grid = {
    'alpha':[0.01, 0.1, 1.0, 10.0, 100.0]
}

lasso_r_CV = GridSearchCV(lasso_r, lasso_r_param_grid, cv=3, n_jobs=-1)

lasso_r_CV_pipeline = make_pipeline(mct, lasso_r_CV)
lasso_r_CV_pipeline.fit(X_train, y_train)
print(lasso_r_CV.best_params_)
print(lasso_r_CV.best_score_)


#Ridge Regression
# to turn some coefficients close to zero
ridge_r = Ridge()

ridge_r_param_grid = {
    'alpha':[0.01, 0.1, 1.0, 10.0, 100.0]
}

ridge_r_CV = GridSearchCV(ridge_r, ridge_r_param_grid, cv=5, n_jobs=-1)

ridge_r_CV_pipeline = make_pipeline(mct, ridge_r_CV)
ridge_r_CV_pipeline.fit(X_train, y_train)
print(ridge_r_CV.best_params_)
print(ridge_r_CV.best_score_)


#ElasticNet Regression
etn_r = ElasticNet()

etn_r_param_grid = {
    'alpha':[0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio':[0.1, 0.3, 0.5, 0.7, 0.9]
}

etn_r_CV = GridSearchCV(etn_r, etn_r_param_grid, cv=3)

etn_r_pipeline = make_pipeline(mct, etn_r_CV)
etn_r_pipeline.fit(X_train, y_train)
print(etn_r_CV.best_params_)
print(etn_r_CV.best_score_)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = etn_r_pipeline.predict(X_test)

print(f'mean_absolute_error: {mean_absolute_error(y_test, y_pred)}')
print(f'mean_squared_error: {mean_squared_error(y_test, y_pred)}')
print(f'r2_score: {r2_score(y_test, y_pred)}')


