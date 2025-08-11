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
import matplotlib.pyplot as plt
import seaborn as sns

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
# >5% to other
small_occupations = df['Occupation'].value_counts().reset_index()
small_occupations.columns = ['Occupation','counts']
small_occupations['ratio'] = (small_occupations['counts'] / small_occupations['counts'].sum(axis = 0)) * 100
print(small_occupations)
small_list = small_occupations.loc[small_occupations['ratio'] < 5, 'Occupation'].to_list()
print(small_list)


#df['Occupation']  = df['Occupation'] .apply(lambda x: 'other' if )

input()



# Drop unwanted columns
print(df.columns)
#df = df.drop(columns = ['', '', '', ''])






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


