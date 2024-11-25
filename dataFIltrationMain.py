import pandas as pd
import numpy as np
# import tensorflow as tf
import os
import random
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import dates as d
import datetime
from datetime import date, timedelta
import re
# from matplotlib import pylab
# from matplotlib.colors import ListedColormap

#functions for processing
from individualFilteringFuncs import intersection, data_filtering_individual, find_yellow_times, get_time_period, broken_list, find_intersection, find_union, matthews_correlation

def dataFiltrationMain():

    # getting data sets
    daily_averages_df = pd.read_csv('aggregatedDailyAverages.csv')
    individual_data_df = pd.read_csv('aggregatedIndividualData.csv', low_memory=False)
    merged_df = pd.read_csv('aggregatedIndividualAndDailyUpdated.csv', low_memory=False)


    merged_df_temp = merged_df.copy()
    merged_df = merged_df[~merged_df['Well'].str.startswith('HEADER')]
    merged_df = merged_df[~merged_df['Well'].str.startswith('INLET')]
    merged_df.reset_index(inplace=True, drop=True) #I CHANGED THIS IDK WHY

    # This cell just removes nan follow up priority values, essentially any wells that need to be moved
    # TODO: might need to come back to do this to decide if it is something we should actually be doing

    yellow_indices = merged_df[merged_df["Follow Up Priority"] == "Yellow"].index
    green_indices = merged_df[merged_df["Follow Up Priority"] == "Green"].index
    nan_indices = merged_df[merged_df["Follow Up Priority"].isna()].index
    yellow_df = merged_df.iloc[yellow_indices]
    green_df = merged_df.iloc[green_indices]

    # getting rid of non existing values of Follow Up Priority
    merged_df = pd.concat([yellow_df, green_df])
    merged_df.reset_index(inplace=True, drop=True)
    yellow_df = merged_df[merged_df["Follow Up Priority"] == "Yellow"]
    green_df = merged_df[merged_df["Follow Up Priority"] == "Green"]



    ## FILLING IN ANY ROWS (BOTH YELLOW< GREEN< & MERGED)
    # TODO: write a function for checking with col has 1 empty and filling it in (!!)

    yellow_df = merged_df[merged_df["Follow Up Priority"] == "Yellow"]
    green_df = merged_df[merged_df["Follow Up Priority"] == "Green"]

    # find all the NaN in the CH4, CO2, O2, Bal.Gas
    # If there is a row in which there are three out of four Nan Values, then fix (!!)

    green_df_temp = green_df[['CH4 (%)', 'CO2 (%)', 'O2 (%)', 'Bal. Gas (%)']].copy()
    green_df_temp.loc[:,'NaN_Count'] = green_df_temp.isna().sum(axis=1)
    green_df_temp

    yellow_df_temp = yellow_df[['CH4 (%)', 'CO2 (%)', 'O2 (%)', 'Bal. Gas (%)']].copy()
    yellow_df_temp.loc[:,'NaN_Count'] = yellow_df_temp.isna().sum(axis=1)
    yellow_df_temp

    # get indices where there is 1 NaN values
    indices = np.where(yellow_df_temp['NaN_Count'] == 1)[0]
    yellow_df_temp.loc[indices]


    # The result of this is empty, so commenting out for now with this data set (!!)
    # indices = np.where(green_df_temp['NaN_Count'] == 1)[0]
    # green_df_temp.loc[indices]

    # ROW 2789 has Bal. Gas missing
    # Since it is only one row, we are filling it in manually
    yellow_df_temp.loc[indices, "Bal. Gas (%)"] = 100 - (
        yellow_df_temp.loc[indices, "CH4 (%)"] +
        yellow_df_temp.loc[indices, "CO2 (%)"] +
        yellow_df_temp.loc[indices, "O2 (%)"]
    )

    # getting rid of non existing values of Follow Up Priority
    merged_df = pd.concat([yellow_df, green_df])
    merged_df.reset_index(inplace=True, drop=True)

    merged_df.loc[indices, "Bal. Gas (%)"] = 100 - (
        merged_df.loc[indices, "CH4 (%)"] +
        merged_df.loc[indices, "CO2 (%)"] +
        merged_df.loc[indices, "O2 (%)"]
    )


    yellow_df_temp.loc[indices]
    merged_df.loc[indices]

    # Drop any rows if there are 5 NaNs in the row or is 3,4 gas values are missing

    # Find the indices of where there are 5 Nans
    # Find the indices of where there are 3, 4 gas values missing
    gas = ['CH4 (%)', 'CO2 (%)', 'O2 (%)', 'Bal. Gas (%)']
    all = ['CH4 (%)', 'CO2 (%)', 'O2 (%)', 'Bal. Gas (%)', 'LFG Flow (SCFM)', 'LFG Temperature (F)', 'Applied Vacuum - PA (in. H2O)', 'Available Vacuum - PB (in. H2O)', 'Liquid Column (feet)']
    merged_df.loc[:,'NaN_Count_gas'] = merged_df[gas].isna().sum(axis=1)
    merged_df.loc[:,'NaN_Count_all'] = merged_df[all].isna().sum(axis=1)


    filtered_df = merged_df[(merged_df['NaN_Count_gas'] <= 3) & (merged_df['NaN_Count_all'] <= 5)]
    filtered_df # EDITED GREATER THAN EQUAL TO 3 GAS OR GREATER THAN EQUAL TO 4

    return daily_averages_df, individual_data_df, merged_df, yellow_df, green_df, filtered_df
  