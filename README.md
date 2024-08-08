# ML-Project-2---House-Price-Predictor

# Housing Price Prediction

This project involves predicting housing prices using a dataset of housing information. The dataset contains various features related to housing and geographic information. The goal is to build a machine learning model to predict the median house value based on these features.

## Project Overview

The project involves the following main steps:
1. **Data Loading and Exploration**: Load the dataset and perform exploratory data analysis.
2. **Feature Engineering**: Create new features and prepare the data for modeling.
3. **Model Training and Evaluation**: Train various models and evaluate their performance using cross-validation and grid search.
4. **Final Model Testing**: Test the final model on a separate test dataset and evaluate its performance.

## Dataset

The dataset used in this project is from the [California Housing dataset](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv). It contains the following columns:
- `longitude`: Longitude coordinate of the housing location.
- `latitude`: Latitude coordinate of the housing location.
- `housing_median_age`: Median age of the houses.
- `total_rooms`: Total number of rooms in the house.
- `total_bedrooms`: Total number of bedrooms in the house.
- `population`: Population of the area.
- `households`: Number of households in the area.
- `median_income`: Median income of the area.
- `ocean_proximity`: Proximity to the ocean (categorical feature).
- `median_house_value`: Median value of the house (target variable).

## Requirements

The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- scikit-learn


