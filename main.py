# AUTHORS:
# Kurshakova Mariia
# Dobrzyniewicz Agata
# DATASET:
# Kaggle.com - Hotel Booking Cancellation Prediction by YOUSSEF ABOELWAFA
import numpy as np
# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

sns.stripplot(data=data[num_columns], orient='h', alpha=.25).set(title='Data without outliers')
plt.savefig("analysis_plots/data_values_without_outliers.png")
plt.tight_layout()
plt.show()

print("--------------")

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
