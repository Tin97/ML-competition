import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

numeric_features = ['Year of Record', 'Age','Country', 'Size of City','Profession', 'Body Height [cm]']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Gender', 'University Degree']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
        ])

regressor = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LinearRegression())])

meansCountry = data_train.groupby('Country')['Income in EUR'].mean()
meansProfession = data_train.groupby('Profession')['Income in EUR'].mean()

data_train['Country'] = data_train['Country'].map(data_train.groupby('Country')['Income in EUR'].mean())
data_train['Profession'] = data_train['Profession'].map(data_train.groupby('Profession')['Income in EUR'].mean())

data_test['Country'] = data_test['Country'].map(meansCountry)
data_test['Profession'] = data_test['Profession'].map(meansProfession)

X = data_train.drop('Instance', axis=1)
X = X.drop('Hair Color', axis=1)
X = X.drop('Wears Glasses', axis=1)
X = X.drop('Income in EUR', axis=1)
y = data_train['Income in EUR']

X_train = data_train.drop('Instance', axis=1)
X_train = X_train.drop('Hair Color', axis=1)
X_train = X_train.drop('Wears Glasses', axis=1)
X_train = X_train.drop('Income in EUR', axis=1)
y_train = data_train['Income in EUR']

X_test = data_test.drop('Instance', axis=1)
X_test = X_test.drop('Hair Color', axis=1)
X_test = X_test.drop('Wears Glasses', axis=1)
instance = data_test['Instance']
X_test = X_test.drop('Income in EUR', axis=1)
y_test = data_test['Income in EUR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#df = pd.DataFrame({'Instance': instance, 'Income': y_pred})
#df.to_csv('submit.csv', sep= ',', index = False)

print(np.sqrt(mean_squared_error(y_test, y_pred)))
