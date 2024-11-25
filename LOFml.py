# imports 
import pandas as pd
import numpy as np
import os
import random
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import dates as d
import datetime
from datetime import date, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.lines as mlines
import re
import seaborn as sns
from sklearn.cluster import KMeans
# !pip install kneed
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score

from individualFilteringFuncs import intersection, data_filtering_individual, find_yellow_times, get_time_period, broken_list, find_intersection, find_union, matthews_correlation



# data aggregation 

# use pytorch

daily_averages_df = pd.read_csv('aggregatedDailyAverages.csv')
individual_data_df = pd.read_csv('aggregatedIndividualData.csv', low_memory=False)
merged_df = pd.read_csv('aggregatedIndividualAndDailyUpdated.csv', low_memory=False)


##LOF WORK##
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

#set up data

desired_well = 'TCGW-064' #TODO: choose a different well
lof_df = merged_select[merged_select['Well'] == desired_well]
desired_features = ['CH4 (%)', 'Applied Vacuum - PA (in. H2O)']
X = lof_df[desired_features]
n_outliers = len(lof_df[lof_df['Follow Up Priority'] == 'Yellow'])
ground_truth = np.ones(len(X), dtype=int) #TODO: Fix this
ground_truth[-n_outliers:] = -1

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_



def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])


plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
scatter = plt.scatter(
    X[:, 0],
    X[:, 1],
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)
plt.axis("tight")
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.xlabel("prediction errors: %d" % (n_errors))
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)
plt.title("Local Outlier Factor (LOF)")
plt.show()



# Fit model for novelty detection
# Using fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict, decision_function and score_samples methods).


def lof(df_well_training, df_well_outliers, df_well_observations):
    selected_features = ['CH4 (%)', 'CO2 (%)']

    data_training = df_well_training[selected_features].values
    data_outliers = df_well_outliers[selected_features].values
    data_input = df_well_observations[selected_features].values

    # Fit model for novelty detection
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
    clf.fit(data_training)

    y_pred_test = clf.predict(data_input)
    y_pred_outliers = clf.predict(data_outliers)
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    # Plotting
    s = 40  # Adjust this value if needed

    plt.figure(figsize=(10, 6))
    b1 = plt.scatter(data_training[:, 0], data_training[:, 1], c="white", s=s, edgecolors="k", label="training observations")
    b2 = plt.scatter(data_input[:, 0], data_input[:, 1], c="blueviolet", s=s, edgecolors="k", label="new regular observations")
    c = plt.scatter(data_outliers[:, 0], data_outliers[:, 1], c="gold", s=s, edgecolors="k", label="new abnormal observations")

    plt.axis("tight")
    plt.xlabel('CH4 (%)')
    plt.ylabel('CO2 (%)')
    plt.legend(
        loc="upper left",
        #prop=matplotlib.font_manager.FontProperties(size=11),
    )

    plt.title("Local Outlier Factor (LOF)")
    plt.show()



# Time to use it!

# choose a well & time period
desired_well = 'TCGW-064' #TODO: choose a different well
startTraining = "2022-09-25"
endTraining = "2023-02-15"

startObs = "2024-01-01"
endObs = "024-06-09"

# merged_select is all NAN data dropped

#merged_gas is all NANs for gas dropped

lof_df = merged_select[merged_select['Well'] == desired_well]

# sort out yellow + greens for a certain amount of days
# start_index, end_index = get_time_period(lof_df, startTraining, endTraining)
mid_index = round(len(lof_df)/2)
df_well_training = lof_df[:mid_index]

df_well_outliers = df_well_training[df_well_training['Follow Up Priority'] == 'Yellow'];
df_well_good = df_well_training[df_well_training['Follow Up Priority'] == 'Green'];

# put in "raw" data for another period
# start_index, end_index = get_time_period(lof_df, startObs, endObs)
df_well_observations = lof_df[mid_index:]

# feed into ML model



#df_well = data_filtering_individual(well_csv)
#start_index, end_index = get_time_period(df_well, start, end)
#df_well_training = df_well[start_index:end_index]
#df_well_outliers = df_well[start_index:end_index]
#df_well_observations = df_well[start_index:end_index]

  # TODO: DATA MANIPULATION
  # Training observations (normal)
  # Novel Observations (normal)
  # Abnormal novel observations (outliers, aka when broken)
  # so take data for half the time

lof(df_well_good, df_well_outliers, df_well_observations )


