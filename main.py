# AUTHORS:
# Kurshakova Mariia
# Dobrzyniewicz Agata
# DATASET:
# Kaggle.com - Hotel Booking Cancellation Prediction by YOUSSEF ABOELWAFA


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import os


os.environ['LOKY_MAX_CPU_COUNT'] = '4'
SPLIT_SIZE = 0.3

# -- DATA IMPORT --
data = pd.read_csv("booking.csv")

print(data.head())
print(data.size)
print(data.columns)

# -- ANALYZE -- 
# stats (min, max, mean, median, std, var, range)

data = data.iloc[:, 1:]
data.info()

print("-------Count null values--------")
print(data.isnull().sum().sort_values(ascending=False))
print("-------Count uniq values--------")
print(data.nunique())

not_num_columns = ['type of meal', 'car parking space', 'room type',
                   'market segment type', 'repeated', 'booking status']

for column in not_num_columns:
    print("----------")
    print(data[column].value_counts())

num_columns = ['number of adults', 'number of children', 'number of weekend nights',
               'number of week nights', 'lead time', 'P-C', 'P-not-C',
               'average price', 'special requests']
for column in num_columns:
    print("----------")
    print("\nVALUE:", column)
    print("MIN:", data[column].min())
    print("MAX:", data[column].max())
    print("MEAN:", data[column].mean())
    print("MEDIAN:", data[column].median())
    print("RANGE:", data[column].max() - data[column].min())
    print("STANDARD DEVIATION:", data[column].std())
    print("VARIANCE:", data[column].var())
    print("PERCENTILE 90%:", data[column].quantile(0.9))

# look for missing or anomalies data

# -- VISUALISE -- - A
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis').set(title='Number of null values')
plt.tight_layout()
plt.savefig("analysis_plots/null_values.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(data[num_columns].corr(), annot=True).set(title='Data correlations')
plt.tight_layout()
plt.savefig("analysis_plots/data_correlations.png")
plt.show()

# takes a long time to display, comment when not needed
#sns.pairplot(data, hue='booking status')
#plt.savefig("analysis_plots/pair_plot.png")
#plt.show()

sns.countplot(x='booking status', data=data)
plt.savefig("analysis_plots/count_booking_statuses.png")
plt.show()

sns.stripplot(data=data[num_columns], orient='h', alpha=.25).set(title='Data with outliers')
plt.savefig("analysis_plots/data_values_outliers.png")
plt.tight_layout()
plt.show()

# -- PREPARE DATA -- - A
# clear
print("--------------")
print("Cleaning data")

outliers_columns = ['lead time', 'average price']

for column in outliers_columns:
    min = data[column].min()
    max = data[column].max()

    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data[column] = data[column].apply(lambda x: np.median(data[column]) if x < lower_bound or x > upper_bound else x)

    print("\nAfter cleaning")
    print("COLUMN:", column)
    print("MIN:", data[column].min(), "(Before: ", min, ")")
    print("MAX:", data[column].max(), "(Before: ", max, ")")

max_children_before = data["number of children"].max()
data = data[data["number of children"] < 7]

print("\nAfter cleaning")
print("COLUMN: number of children")
print("MAX:", data["number of children"].max(), "(Before: ", max_children_before, ")")

sns.stripplot(data=data[num_columns], orient='h', alpha=.25).set(title='Data without outliers')
plt.savefig("analysis_plots/data_values_without_outliers.png")
plt.tight_layout()
plt.show()

print("--------------")

# change string values to numbers
print("--------------")
print("Change strings to numbers")
pd.set_option('display.max_columns', None)
print(data.head())

binary_not_num_columns = ['car parking space', 'repeated', 'booking status']

label_encoder = LabelEncoder()

for column in not_num_columns:
    data[column] = label_encoder.fit_transform(data[column])

# split date of reservation
data['date of reservation'] = pd.to_datetime(data['date of reservation'], errors='coerce')

data['year'] = data['date of reservation'].dt.year
data['month'] = data['date of reservation'].dt.month
data['day'] = data['date of reservation'].dt.day

data = data.dropna(subset=['date of reservation'])

data.drop('date of reservation', axis=1, inplace=True)

print(data.head())
print("--------------")

# -- SPLIT DATA -- - M

X = data.drop(['booking status'], axis=1)
y = data['booking status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_SIZE, random_state=42)

# -- PREPARE SPLIT DATA -- - A
# data augmentation
print("Before data augmentation: ", Counter(y_train))

smote = SMOTE(sampling_strategy=1.0)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After data augmentation: ", Counter(y_train))

sns.countplot(x='booking status', data=pd.DataFrame(data=y_train, columns=['booking status']))
plt.savefig("analysis_plots/count_booking_statuses_after_augmentation_y.png")
plt.show()

print("--------------")

# normalize

# -- MODEL CREATION 1 --
# Random Forest

# -- PREDICTIONS --

# -- ANALYZE PREDICTIONS --
# calculate useful metrics
# use CleanML - don't lol

# visualise important observations

# -- MODEL CREATION 2 --
# SVM

# -- PREDICTIONS --

# -- ANALYZE PREDICTIONS --
# calculate useful metrics
# use CleanML - don't lol

# visualise important observations

# -- MODEL CREATION 3 --
# Naïve Bayes

# -- PREDICTIONS --

# -- ANALYZE PREDICTIONS --
# calculate useful metrics
# use CleanML - don't lol

# visualise important observations

# -- SAVING PREDICTIONS --


# -- COMPARE MODELS --


# -- SUMMARIZATION --
