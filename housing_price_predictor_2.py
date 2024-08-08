# Load libraries
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Load dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
housing = read_csv(url)

# Print initial information about the dataset
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

# Plot histograms of all numerical features
housing.hist(bins=50, figsize=(20,15))
# plt.show() # Uncomment to display the histogram plot

# Add a new categorical feature based on income
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

# Plot histogram of the new income category feature
housing["income_cat"].hist()
# plt.show() # Uncomment to display the histogram plot

# Perform stratified shuffle split to create train and test sets
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Split the data into train and test sets
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Print the proportion of each income category in the test set
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# Remove the income_cat column from both train and test sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Copy the train set for further processing
housing = strat_train_set.copy()

# Plot a scatter plot to visualize geographical distribution and median house value
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, 
             label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()

# Compute and print correlation matrix for numerical features
numeric_features = housing.select_dtypes(include=[np.number])
corr_matrix = numeric_features.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Plot scatter matrix for selected attributes
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

# Plot scatter plot for median income vs. median house value
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

# Create new features to capture additional information
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# Recompute correlation matrix with new features
numeric_features = housing.select_dtypes(include=[np.number])
corr_matrix = numeric_features.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Separate features and labels
housing = housing.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Define the numerical data pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

# Apply the numerical pipeline to the training data
housing_num = housing.drop("ocean_proximity", axis=1)
X = num_pipeline.fit_transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# Encode categorical feature
housing_cat = housing[["ocean_proximity"]]
cat_encoder = OneHotEncoder()
Y = cat_encoder.fit_transform(housing_cat)
Y_dense = Y.toarray()
housing_cat_1hot = pd.DataFrame(Y_dense, columns=cat_encoder.get_feature_names_out(["ocean_proximity"]))

# Combine numerical and categorical features into a single DataFrame
housing_prepared = pd.concat([housing_tr, housing_cat_1hot], axis=1)

# Define and evaluate a random forest model with cross-validation
forest_reg = RandomForestRegressor()
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)

# Display cross-validation scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(forest_rmse_scores)
'''
# Define parameter grid for GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# Get the best model from Grid Search
final_model = grid_search.best_estimator_
'''
final_model=forest_reg

# Fit the final model on the training data
final_model.fit(housing_prepared, housing_labels)

# Prepare the test data
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# Create new features for the test data
X_test["rooms_per_household"] = X_test["total_rooms"] / X_test["households"]
X_test["bedrooms_per_room"] = X_test["total_bedrooms"] / X_test["total_rooms"]
X_test["population_per_household"] = X_test["population"] / X_test["households"]

# Transform the test data using the same pipeline as the training data
X_test_num = X_test.drop("ocean_proximity", axis=1)
X_wip = num_pipeline.transform(X_test_num)
X_test_tr = pd.DataFrame(X_wip, columns=X_test_num.columns)

# Encode the categorical feature in the test data
X_test_cat = X_test[["ocean_proximity"]]
X_test_cat_enc = cat_encoder.transform(X_test_cat)
X_test_dense = X_test_cat_enc.toarray()
X_test_cat_1hot = pd.DataFrame(X_test_dense, columns=cat_encoder.get_feature_names_out(["ocean_proximity"]))

# Combine the transformed numerical and categorical features
X_test_prepared = pd.concat([X_test_tr, X_test_cat_1hot], axis=1)

# Make predictions and evaluate the final model
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("Final RMSE:", final_rmse)
