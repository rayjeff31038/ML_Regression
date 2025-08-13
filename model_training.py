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
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#figure
import matplotlib.pyplot as plt
import seaborn as sns

#save model
import joblib

#The file was downloaded from Kaggle: "Sleep Health and Lifestyle Dataset".
#https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/code
#read the file
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

############################
#Step1. data exploring (EDA)
#############################

print(df.head(10),'\n')

print(df.shape, '\n')
#(374, 13)

print(df.info())
# Data columns (total 13 columns):
 #   Column                   Non-Null Count  Dtype
# ---  ------                   --------------  -----
#  0   Person ID                374 non-null    int64
#  1   Gender                   374 non-null    object
#  2   Age                      374 non-null    int64
#  3   Occupation               374 non-null    object
#  4   Sleep Duration           374 non-null    float64
#  5   Quality of Sleep         374 non-null    int64
#  6   Physical Activity Level  374 non-null    int64
#  7   Stress Level             374 non-null    int64
#  8   BMI Category             374 non-null    object
#  9   Blood Pressure           374 non-null    object
#  10  Heart Rate               374 non-null    int64
#  11  Daily Steps              374 non-null    int64
#  12  Sleep Disorder           155 non-null    object
# dtypes: float64(1), int64(7), object(5)

# 'Sleep Disorder' column need to check NA value


print(df.describe(include = ['O']), '\n')
#        Gender Occupation BMI Category Blood Pressure Sleep Disorder
# count     374        374          374            374            155
# unique      2         11            4             25              2
# top      Male      Nurse       Normal         130/85    Sleep Apnea
# freq      189         73          195             99             78


print(df['Sleep Disorder'].unique())
# [nan 'Sleep Apnea' 'Insomnia']

# Actually 'nan' in Sleep Disorder means 'no'
# fill 'nan' with 'no'
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('no')
print(df)

print(df.dtypes, '\n')
# Person ID                    int64
# Gender                      object
# Age                          int64
# Occupation                  object
# Sleep Duration             float64
# Quality of Sleep             int64
# Physical Activity Level      int64
# Stress Level                 int64
# BMI Category                object
# Blood Pressure              object
# Heart Rate                   int64
# Daily Steps                  int64
# Sleep Disorder              object
# dtype: object

#check all unique values in all objective columns
for c in df.columns:
    if df[c].dtypes == 'object':
        print(f'{c} unique values: {df[c].unique()}')
# Gender unique values: ['Male' 'Female']
# Occupation unique values: ['Software Engineer' 'Doctor' 'Sales Representative' 'Teacher' 'Nurse'
#  'Engineer' 'Accountant' 'Scientist' 'Lawyer' 'Salesperson' 'Manager']
# BMI Category unique values: ['Overweight' 'Normal' 'Obese' 'Normal Weight']
# Blood Pressure unique values: ['126/83' '125/80' '140/90' '120/80' '132/87' '130/86' '117/76' '118/76'
#  '128/85' '131/86' '128/84' '115/75' '135/88' '129/84' '130/85' '115/78'
#  '119/77' '121/79' '125/82' '135/90' '122/80' '142/92' '140/95' '139/91'
#  '118/75']
# Sleep Disorder unique values: ['no' 'Sleep Apnea' 'Insomnia']


# In the 'BMI Category' variable, there are both 'Normal' and 'Normal Weight'
# which have the same meaning
# replace 'Normal Weight' to 'Normal'
df['BMI Category'] = df['BMI Category'].apply(lambda x: 'Normal' if x == 'Normal Weight' else x)


# Blood Pressure is not useable in this status
# Blood pressure data requires further split into systolic and diastolic values
# And add a new column indicating "Cardiovascular Diseases" column
# criteria for "Cardiovascular Diseases": 
# Hypertension: systolic >= 130 or diastolic >= 80
# Hypotension: systolic <= 90 or diastolic <= 60
# But its difficult to 
df['Systolic'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
df['Diastolic'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))

def  Diseases(row):
    if row['Systolic'] >= 130 or row['Diastolic'] >= 80:
        return 'yes'
    elif row['Systolic'] <= 90 or row['Diastolic'] <= 60:
        return 'yes'
    else:
        return 'no'
    
df['Cardiovascular Diseases'] = df.apply(Diseases, axis = 1)


# check target column 'Quality of Sleep' distribution
# imbalance is not considered severe
print(df['Quality of Sleep'].value_counts())
print(df['Quality of Sleep'].value_counts(normalize = True))
# 8    109
# 6    105
# 7     77
# 9     71
# 5      7
# 4      5

# 8    0.291444
# 6    0.280749
# 7    0.205882
# 9    0.189840
# 5    0.018717
# 4    0.013369


# Drop the unnecessary columns 'Person ID ' and 'Blood Pressure' (already transformed)
df = df.drop(columns = ['Person ID','Blood Pressure'])

# Check Numerical data distribution by = 'Quality of Sleep'
Numerical = []
for c in df.columns:
    if df[c].dtype != 'object':
        if c != 'Quality of Sleep':
            Numerical.append(c)
            print(f'Numerical data: {c}')
# Numerical data: Age
# Numerical data: Sleep Duration
# Numerical data: Physical Activity Level
# Numerical data: Stress Level
# Numerical data: Heart Rate
# Numerical data: Daily Steps
# Numerical data: Systolic
# Numerical data: Diastolic

# Use boxplot to check data distribution by 'Quality of Sleep':
# Results show  
fig1, axes1 = plt.subplots(nrows = 2, ncols = 4, figsize = (20,15))
axes1 = axes1.flatten()

for i, numcol in enumerate(Numerical):
    if numcol == 'Stress Level' or numcol == 'Systolic' or numcol == 'Diastolic' or  numcol == 'Heart Rate':
            sns.boxplot(df, x = 'Quality of Sleep', y = numcol, ax = axes1[i], hue = 'Quality of Sleep')
            axes1[i].set_title(f'{numcol} distribution by Quality of Sleep')
            axes1[i].set_xlabel('Quality of Sleep')
            axes1[i].set_ylabel(f'{numcol}')
            axes1[i].legend(loc='lower left', fontsize=8)
    else:
        sns.boxplot(df, x = 'Quality of Sleep', y = numcol, ax = axes1[i], hue = 'Quality of Sleep')
        axes1[i].set_title(f'{numcol} distribution by Quality of Sleep')
        axes1[i].set_xlabel('Quality of Sleep')
        axes1[i].set_ylabel(f'{numcol}')
        axes1[i].legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('Numerical_data_boxplot.png', dpi=300)
#plt.show()
plt.close()


# Use barplot to check Categorical data distribution by 'Quality of Sleep'
Categorical = []
for c in df.columns:
    if df[c].dtype == 'object':
        Categorical.append(c)
        print(f'Categorical data: {c}')
# Categorical data: Gender
# Categorical data: Occupation
# Categorical data: BMI Category
# Categorical data: Sleep Disorder
# Categorical data: Cardiovascular Diseases

fig2, axes2 = plt.subplots(nrows = 2, ncols = 3, figsize = (15,10))
axes2 = axes2.flatten()
print(axes2)

for i, catcol in enumerate(Categorical):
    df_draw = df.groupby(['Quality of Sleep', catcol]).size().reset_index(name = 'count')
    sns.barplot(df_draw, x = 'Quality of Sleep', y = 'count', hue = catcol, ax = axes2[i])
    axes2[i].set_title(f'Count of {catcol} by Quality of Sleep')
    axes2[i].set_xlabel('Quality of Sleep')
    axes2[i].set_ylabel(f'{catcol} counts')

#close empty figure
for ax in axes2[len(Categorical):]:
    ax.axis('off')

plt.tight_layout()
plt.savefig('Categorical_data_barplot.png', dpi=300)
#plt.show()
plt.close()


# Use barplot to check 'Quality of Sleep' data distribution
plt.figure(figsize=(8, 5))
df_sleep = df.groupby('Quality of Sleep').size().reset_index(name = 'count')
sns.barplot(df_sleep, x = 'Quality of Sleep', y = 'count')
plt.savefig('Quality_of_Sleep_counts_barplot.png', dpi=300)
#plt.show()
plt.close()


#correlation map
cor = df.corr(numeric_only = True)
plt.figure(figsize = (20,20))

sns.heatmap(
    data = cor,
    annot = True,
    fmt = '.2f',
    annot_kws={"size": 14, "color": "black"},
    cmap='coolwarm',
    vmin=-1, vmax=1,
    center=0
)
plt.yticks(rotation = 0)
plt.title('Correlation Heatmap', size = 16)
plt.savefig('Correlation Heatmap.png', dpi = 300)
#plt.show()

########################################################################################################
# >>> EDA results evulation
#
## Systolic and Diastolic are highly correlated and should be combined into 'Mean arterial pressure'
## to reduce 'multicollinearity'
## Mean arterial pressure: (Systolic + Diastolic*2)/3
##
## The numerical data does not contain excessive outliers, and the category distributions are relatively 
## balanced, so StandardScaler can be applied for transformation.
##
## The proportions of the categorical data are not completely balanced,
## but most are still within an acceptable range.
## The distribution of Occupation is highly imbalanced and requires category merging
## to reduce the number of occupation types.
## Cardiovascular Diseases is derived from Systolic and Diastolic,
## but its distribution is also highly imbalanced and it is highly correlated
## with the Systolic and Diastolic values.
## Therefore, this column can be removed.
## The remaining columns can undergo further transformation.
##
## The target variable y is also imbalanced, especially for the
## Quality of Sleep scores of 4 and 5.
## This indicates that the model’s predictions for these scores
## are likely to be less accurate.
## If the model’s performance is unsatisfactory, maybe can further
## try increasing the weights for these cases to improve results
#########################################################################################################




############################
#Step2. Data preprocessing
############################

# Add new column 'Mean arterial pressure'
# Mean arterial pressure: (Systolic + Diastolic*2)/3
df['Mean arterial pressure'] = (df['Systolic'] + df['Diastolic'] * 2) / 3

# Reduce the number of occupation types.
# <5% to other
small_occupations = df['Occupation'].value_counts().reset_index()
small_occupations.columns = ['Occupation','counts']
small_occupations['ratio'] = (small_occupations['counts'] / small_occupations['counts'].sum(axis = 0)) * 100
#print(small_occupations)
small_list = small_occupations.loc[small_occupations['ratio'] < 5, 'Occupation'].to_list()
#print(small_list)

df['Occupation']  = df['Occupation'] .apply(lambda x: 'other' if x in small_list else x)


# Drop unwanted columns
print(df.columns)
# Index(['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
#        'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate',
#        'Daily Steps', 'Sleep Disorder', 'Systolic', 'Diastolic',
#        'Cardiovascular Diseases', 'Mean arterial pressure'],
#       dtype='object')

df = df.drop(columns = ['Systolic', 'Diastolic', 'Cardiovascular Diseases'])
print(df.info())

#input()


# Object columns: Gender, Occupation, BMI Category, Sleep Disorder
# Numeric columns: Age, Sleep Duration, Physical Activity Level, Stress Level, Heart Rate, Daily Steps, Mean arterial pressure
# Target columns(y) = Quality of Sleep 
# Data transformation is required for those columns before model building

# Strategy for data transformation:
# Gender, Occupation --> OneHotEncoder
# BMI Category, Sleep Disorder --> OrdinalEncoder
# Age, Sleep Duration, Physical Activity Level, 
# Stress Level, Heart Rate, Daily Steps, Mean arterial pressure --> StandardScaler


# Split dataframe to train_df and test_df
# X = all the factors 
# y = 'Quality of Sleep'
X = df.drop(columns = ['Quality of Sleep'])
y = df['Quality of Sleep']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#columns for data transformation
one_hot_columns = ['Gender','Occupation']
StandardScaler_columns = ['Age','Sleep Duration','Physical Activity Level','Stress Level','Heart Rate','Daily Steps','Mean arterial pressure']

BMI_Category_rank = ['Normal','Overweight','Obese']
Sleep_Disorder_rank = ['no','Sleep Apnea','Insomnia']

mct = make_column_transformer(
    (OneHotEncoder(handle_unknown = 'ignore'), one_hot_columns),

    (StandardScaler(), StandardScaler_columns),

    (Pipeline([
         ('ordinal', OrdinalEncoder(categories = [BMI_Category_rank])),
         ('scale', StandardScaler()), 
    ]), ['BMI Category']),

    (Pipeline([
        ('ordinal', OrdinalEncoder(categories = [Sleep_Disorder_rank])),
        ('scale', StandardScaler()), 
    ]), ['Sleep Disorder']),

    remainder = 'drop',
    n_jobs = -1
)



############################
#Step3. Model training
############################

result = []

#LinearRegression model
lr = LinearRegression()
lr_pipeline = make_pipeline(mct, lr)

lr_param_grid = {
    'linearregression__fit_intercept':[True, False],
    'linearregression__positive': [True, False]
}

lr_CV = GridSearchCV(lr_pipeline, lr_param_grid, cv =3, n_jobs=-1)
lr_CV.fit(X_train, y_train)

lr_best_pipe = lr_CV.best_estimator_
lr_y_pred = lr_best_pipe.predict(X_test)


print('\n')
print('> LinearRegression model information')
print(f'best_params: {lr_CV.best_params_}')
print(f'best_score: {lr_CV.best_score_}')

print('\n')
print('> LinearRegression model test results')
print(f'r2_score: {r2_score(y_test, lr_y_pred)}')
print(f'MSE: {mean_squared_error(y_test, lr_y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, lr_y_pred)}')

result.append([
    'LinearRegression', 
    lr_CV.best_params_, 
    lr_CV.best_score_,
    r2_score(y_test, lr_y_pred),
    mean_squared_error(y_test, lr_y_pred),
    mean_absolute_error(y_test, lr_y_pred)
    ])
#input()


#Ridge Regression
rd_r = Ridge()
rd_pipeline = make_pipeline(mct, rd_r)

rd_r_param_grid = {
    'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

rd_r_CV = GridSearchCV(rd_pipeline, rd_r_param_grid, cv = 3, n_jobs = -1)
rd_r_CV.fit(X_train, y_train)

rd_best_pipe = rd_r_CV.best_estimator_
rd_y_pred = rd_best_pipe.predict(X_test)

print('\n')
print('> Ridge Regression model information')
print(f'best_params: {rd_r_CV.best_params_}')
print(f'best_score: {rd_r_CV.best_score_}')

print('\n')
print('> Ridge Regression test results')
print(f'r2_score: {r2_score(y_test, rd_y_pred)}')
print(f'MSE: {mean_squared_error(y_test, rd_y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, rd_y_pred)}')

result.append([
    'Ridge Regressionn', 
    rd_r_CV.best_params_, 
    rd_r_CV.best_score_,
    r2_score(y_test, rd_y_pred),
    mean_squared_error(y_test, rd_y_pred),
    mean_absolute_error(y_test, rd_y_pred)
    ])


# Lasso regression
ls_r = Lasso()
ls_pipeline = make_pipeline(mct, ls_r)

ls_r_param_grid = {
    'lasso__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

ls_r_CV = GridSearchCV(ls_pipeline, ls_r_param_grid, cv=3, n_jobs=-1)
ls_r_CV.fit(X_train, y_train)

ls_best_pipe = ls_r_CV.best_estimator_
ls_y_pred = ls_best_pipe.predict(X_test)

print('\n')
print('> Lasso Regression model information')
print(f'best_params: {ls_r_CV.best_params_}')
print(f'best_score: {ls_r_CV.best_score_}')

print('\n')
print('> Lasso Regression test results')
print(f'r2_score: {r2_score(y_test, ls_y_pred)}')
print(f'MSE: {mean_squared_error(y_test, ls_y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, ls_y_pred)}')

result.append([
    'Lasso Regression', 
    ls_r_CV.best_params_, 
    ls_r_CV.best_score_,
    r2_score(y_test, ls_y_pred),
    mean_squared_error(y_test, ls_y_pred),
    mean_absolute_error(y_test, ls_y_pred)
    ])
 


# ElasticNet Regression
etn_r = ElasticNet()
etn_pipeline = make_pipeline(mct, etn_r)

etn_r_param_grid = {
    'elasticnet__alpha':[0.01, 0.1, 1.0, 10.0, 100.0],
    'elasticnet__l1_ratio':[0.1, 0.3, 0.5, 0.7, 0.9]
}

etn_r_CV = GridSearchCV(etn_pipeline, etn_r_param_grid, cv=3, n_jobs = -1)
etn_r_CV.fit(X_train, y_train)

etn_best_pipe = etn_r_CV.best_estimator_
etn_y_pred = etn_best_pipe.predict(X_test)

print('\n')
print('> ElasticNet Regression model information')
print(f'best_params: {etn_r_CV.best_params_}')
print(f'best_score: {etn_r_CV.best_score_}')

print('\n')
print('> ElasticNet Regression test results')
print(f'r2_score: {r2_score(y_test, etn_y_pred)}')
print(f'MSE: {mean_squared_error(y_test, etn_y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, etn_y_pred)}')

result.append([
    'ElasticNet Regression', 
    etn_r_CV.best_params_, 
    etn_r_CV.best_score_,
    r2_score(y_test, etn_y_pred),
    mean_squared_error(y_test, etn_y_pred),
    mean_absolute_error(y_test, etn_y_pred)
    ])



#SVR
svr_r = SVR()
svr_pipeline = make_pipeline(mct, svr_r)

svr_r_param_grid = {
    'svr__kernel': ['rbf', 'linear'],
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': ['scale', 'auto']
}

svr_r_CV = GridSearchCV(svr_pipeline, svr_r_param_grid, cv=3, n_jobs=-1)
svr_r_CV.fit(X_train, y_train)

svr_best_pipe = svr_r_CV.best_estimator_
svr_y_pred = svr_best_pipe.predict(X_test)

print('\n')
print('> SVR model information')
print(f'best_params: {svr_r_CV.best_params_}')
print(f'best_score: {svr_r_CV.best_score_}')

print('\n')
print('> SVR test results')
print(f'r2_score: {r2_score(y_test, svr_y_pred)}')
print(f'MSE: {mean_squared_error(y_test, svr_y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, svr_y_pred)}')

result.append([
    'SVR', 
    svr_r_CV.best_params_, 
    svr_r_CV.best_score_,
    r2_score(y_test, svr_y_pred),
    mean_squared_error(y_test, svr_y_pred),
    mean_absolute_error(y_test, svr_y_pred)
    ])


#RandomForestRegressor
rf_r = RandomForestRegressor(random_state = 1)
rf_pipeline = make_pipeline(mct, rf_r)

rf_r_param_grid = {
    'randomforestregressor__n_estimators': [100, 200, 500], 
    'randomforestregressor__max_depth': [None, 5, 10],
    'randomforestregressor__min_samples_split': [2, 5, 10],
    'randomforestregressor__min_samples_leaf': [1, 2, 4]
}

rf_r_CV = GridSearchCV(rf_pipeline, rf_r_param_grid, cv=3, n_jobs=-1)
rf_r_CV.fit(X_train, y_train)

rf_best_pipe = rf_r_CV.best_estimator_
rf_y_pred = rf_best_pipe.predict(X_test)

print('\n')
print('> RandomForestRegressor model information')
print(f'best_params: {rf_r_CV.best_params_}')
print(f'best_score: {rf_r_CV.best_score_}')

print('\n')
print('> RandomForestRegressor test results')
print(f'r2_score: {r2_score(y_test, rf_y_pred)}')
print(f'MSE: {mean_squared_error(y_test, rf_y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, rf_y_pred)}')

result.append([
    'RandomForestRegressor', 
    rf_r_CV.best_params_, 
    rf_r_CV.best_score_,
    r2_score(y_test, rf_y_pred),
    mean_squared_error(y_test, rf_y_pred),
    mean_absolute_error(y_test, rf_y_pred)
    ])



# GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=1)
gbr_pipeline = make_pipeline(mct, gbr)

gbr_param_grid = {
    'gradientboostingregressor__n_estimators': [100, 200, 500],
    'gradientboostingregressor__learning_rate': [0.03, 0.05, 0.1],
    'gradientboostingregressor__max_depth': [2, 3, 5],
    'gradientboostingregressor__subsample': [1.0, 0.7, 0.5],
    'gradientboostingregressor__min_samples_split': [2, 5, 10],
    'gradientboostingregressor__min_samples_leaf': [1, 2, 4],
}

gbr_CV = GridSearchCV(gbr_pipeline, gbr_param_grid, cv=3, n_jobs=-1)
gbr_CV.fit(X_train, y_train)

gbr_best_pipe = gbr_CV.best_estimator_
gbr_y_pred = gbr_best_pipe.predict(X_test)

print('\n')
print('> GradientBoostingRegressor model information')
print(f'best_params: {gbr_CV.best_params_}')
print(f'best_score: {gbr_CV.best_score_}')

print('\n')
print('> GradientBoostingRegressor test results')
print(f'r2_score: {r2_score(y_test, gbr_y_pred)}')
print(f'MSE: {mean_squared_error(y_test, gbr_y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, gbr_y_pred)}')


#model results df
result.append([
    'GradientBoostingRegressor', 
    gbr_CV.best_params_, 
    gbr_CV.best_score_,
    r2_score(y_test, gbr_y_pred),
    mean_squared_error(y_test,gbr_y_pred),
    mean_absolute_error(y_test, gbr_y_pred)
    ])

results_df = pd.DataFrame(result, columns = ['model','best_params','best_score','r2_score','mean_squared_error','mean_absolute_error'])
print(results_df)


results_df.to_csv('model_results.csv', index = False)


# Based on the results_df,
# the RandomForestRegressor shows the highest r2 score (0.98) and very low MSE (0.03) and MAE (0.05).
# 'RandomForestRegressor' is the best model choice for this project.


#model output
joblib.dump(rf_best_pipe, 'RandomForestRegressor_best_model.pkl')