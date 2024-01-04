# AUTHORS:
# Kurshakova Mariia
# Dobrzyniewicz Agata
# DATASET:
# Kaggle.com - Hotel Booking Cancellation Prediction by YOUSSEF ABOELWAFA

# imports
import pandas as pd

# -- DATA IMPORT --
data = pd.read_csv("booking.csv")

print(data.head())
print(data.describe())
print(data.size)
print(data.columns)

# -- ANALYZE -- - M
# stats (min, max, mean, median, std, var, range)

# look for missing or anomalies data

# -- VISUALISE -- - A
# find correlations

# -- PREPARE DATA -- - A
# clear

# data augmentation

# normalize

# -- SPLIT DATA -- - M
# train and test

# -- MODEL CREATION --
# from sklearn import LogicRegression (o ile będzie pasować)
# https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# -- PREDICTIONS --

# -- ANALYZE PREDICTIONS --
# calculate useful metrics
# use CleanML

# visualise important observations

# -- SAVING PREDICTIONS --


# -- SUMMARIZATION --
