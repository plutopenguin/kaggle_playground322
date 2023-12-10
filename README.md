# Predicting horse survival with decision trees and random forests
This is a fairly simple Jupyter Notebook submission for a Kaggle Playground Series competition.
## Goal
Given various medical indicators, predict the health outcomes of horses.
## My Approach
- Converted categorical variables with *k* possible variables into *k-1* dummy variables (avoiding dummy variable trap)
- Replaced missing values with the mean or mode of the remaining values for the same variable
- Split the data into training and testing data for machine learning
  - I tried using **lightgbm.LGBMClassifier**, **sklearn.ensemble.RandomForestClassifier**, and **tfdf.keras.RandomForestModel** (TensorFlow).
## Results
- While my best public score was at 0.81097, my best score in the end was (an admittedly unimpressive) **0.74545**. This score was achieved with the lightgbm classifier, which is a gradient-boosting framework based on decision trees.
