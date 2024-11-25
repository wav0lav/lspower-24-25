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



# data aggregation 

daily_averages_df = pd.read_csv('aggregatedDailyAverages.csv')
individual_data_df = pd.read_csv('aggregatedIndividualData.csv', low_memory=False)
merged_df = pd.read_csv('aggregatedIndividualAndDailyUpdated.csv', low_memory=False)
