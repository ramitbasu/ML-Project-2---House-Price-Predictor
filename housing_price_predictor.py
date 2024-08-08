#load libraries

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load dataset

url="https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
housing=read_csv(url)

print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

housing.hist(bins=50, figsize=(20,15))
#plt.show()

import pandas as pd
import numpy as np

housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()
#plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
     strat_train_set = housing.loc[train_index]
     strat_test_set = housing.loc[test_index]

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
     set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,s=housing["population"]/100, label="population", figsize=(10,7),c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()

numeric_features = housing.select_dtypes(include=[np.number])
corr_matrix = numeric_features.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)



housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]



numeric_features = housing.select_dtypes(include=[np.number])
corr_matrix = numeric_features.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing = housing.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('std_scaler', StandardScaler()),])

X = num_pipeline.fit_transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


'''
imputer.fit(housing_num)


X = imputer.transform(housing_num)


housing_tr = pd.DataFrame(X, columns=housing_num.columns)
'''



housing_cat = housing[["ocean_proximity"]]
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
Y = cat_encoder.fit_transform(housing_cat)
Y_dense = Y.toarray()
housing_cat_1hot = pd.DataFrame(Y_dense, columns=cat_encoder.get_feature_names_out(["ocean_proximity"]))

housing_prepared=pd.concat([housing_tr, housing_cat_1hot], axis=1);

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=50)

scores = cross_val_score(forest_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=5)
forest_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
     print("Scores:", scores)
     print("Mean:", scores.mean())
     print("Standard deviation:", scores.std())

display_scores(forest_rmse_scores)
'''
from sklearn.model_selection import GridSearchCV
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

final_model  = grid_search.best_estimator_
'''
final_model  = forest_reg

final_model.fit(housing_prepared, housing_labels)


X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test["rooms_per_household"] =X_test["total_rooms"]/X_test["households"]
X_test["bedrooms_per_room"] = X_test["total_bedrooms"]/X_test["total_rooms"]
X_test["population_per_household"]=X_test["population"]/X_test["households"]


X_test_num = X_test.drop("ocean_proximity", axis=1)

X_wip = num_pipeline.fit_transform(X_test_num)
X_test_tr = pd.DataFrame(X_wip, columns=X_test_num.columns)

X_test_cat = X_test[["ocean_proximity"]]

X_test_cat_enc = cat_encoder.fit_transform(X_test_cat)
X_test_dense = X_test_cat_enc.toarray()
X_test_cat_1hot = pd.DataFrame(X_test_dense, columns=cat_encoder.get_feature_names_out(["ocean_proximity"]))

X_test_prepared=pd.concat([X_test_tr, X_test_cat_1hot], axis=1);

from sklearn.metrics import mean_squared_error


final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(final_rmse)




















 


