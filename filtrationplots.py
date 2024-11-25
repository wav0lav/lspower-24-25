import pandas as pd
import numpy as np
# import tensorflow as tf
import os
import random
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import dates as d
# from matplotlib import pylab
# from matplotlib.colors import ListedColormap

from individualFilteringFuncs import intersection, data_filtering_individual, find_yellow_times, get_time_period, broken_list, find_intersection, find_union, matthews_correlation

from dataFIltrationMain import dataFiltrationMain


""" daily_averages_df = pd.read_csv('aggregatedDailyAverages.csv')
individual_data_df = pd.read_csv('aggregatedIndividualData.csv', low_memory=False)
merged_df = pd.read_csv('aggregatedIndividualAndDailyUpdated.csv', low_memory=False)


merged_df_temp = merged_df.copy()
merged_df = merged_df[~merged_df['Well'].str.startswith('HEADER')]
merged_df = merged_df[~merged_df['Well'].str.startswith('INLET')]
merged_df.reset_index(inplace=True)

# This cell just removes nan follow up priority values, essentially any wells that need to be moved
# TODO: might need to come back to do this to decide if it is something we should actually be doing

yellow_indices = merged_df[merged_df["Follow Up Priority"] == "Yellow"].index
green_indices = merged_df[merged_df["Follow Up Priority"] == "Green"].index
nan_indices = merged_df[merged_df["Follow Up Priority"].isna()].index
yellow_df = merged_df.iloc[yellow_indices]
green_df = merged_df.iloc[green_indices]
# n = merged_df.iloc[nan_indices]

# getting rid of non existing values of Follow Up Priority
merged_df = pd.concat([yellow_df, green_df])
merged_df.reset_index(inplace=True)
merged_df

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

merged_df.loc[indices, "Bal. Gas (%)"] = 100 - (
    merged_df.loc[indices, "CH4 (%)"] +
    merged_df.loc[indices, "CO2 (%)"] +
    merged_df.loc[indices, "O2 (%)"]
)


yellow_df_temp.loc[indices]
merged_df.loc[indices]

merged_df

# Find the indices of where there are 5 Nans
# Find the indices of where there are 3, 4 gas values missing
gas = ['CH4 (%)', 'CO2 (%)', 'O2 (%)', 'Bal. Gas (%)']
all = ['CH4 (%)', 'CO2 (%)', 'O2 (%)', 'Bal. Gas (%)', 'LFG Flow (SCFM)', 'LFG Temperature (F)', 'Applied Vacuum - PA (in. H2O)', 'Available Vacuum - PB (in. H2O)', 'Liquid Column (feet)']
merged_df.loc[:,'NaN_Count_gas'] = merged_df[gas].isna().sum(axis=1)
merged_df.loc[:,'NaN_Count_all'] = merged_df[all].isna().sum(axis=1)


filtered_df = merged_df[(merged_df['NaN_Count_gas'] <= 3) & (merged_df['NaN_Count_all'] <= 5)]
filtered_df # EDITED GREATER THAN EQUAL TO 3 GAS OR GREATER THAN EQUAL TO 4


 """

### COPY PASTE UP TO HERE FOR ANY FILTRATION WORK###

daily_averages_df, individual_data_df, merged_df, yellow_df, green_df, filtered_df = dataFiltrationMain()

# Check for null value trends by well type

# aggregate data for different types of wells -- LC, GW, Header

unique_wells = daily_averages_df["Well"].unique()
h = "Header"
lc = "TCLC"
gw = "TCGW"

header_averages_list = []
lc_averages_list = []
gw_averages_list = []

for well in unique_wells:
    well_df = daily_averages_df.loc[daily_averages_df["Well"] == well]
    if h in well:
        header_averages_list.append(well_df)
    if lc in well:
        lc_averages_list.append(well_df)
    if gw in well:
        gw_averages_list.append(well_df)

header_averages_df = pd.concat(header_averages_list, ignore_index=True)
lc_averages_df = pd.concat(lc_averages_list, ignore_index=True)
gw_averages_df = pd.concat(gw_averages_list, ignore_index=True)

# plot NR over time? for different parameters
param = "CH4 (%)" # change based on what you are looking at
#header_averages_df
Hwells = header_averages_df["Well"].unique()

#remove wells that don't seem to provide any data
wells_to_remove = {"N. Header", "S. Header", "NE Header", "E Header", "401 Header"}
Hwells = [element for element in Hwells if element not in wells_to_remove]
# loop thru for every header
desired_valsH_df = header_averages_df.loc[:, [param]]
days = header_averages_df["day"]
bin_df = desired_valsH_df.notna().astype(int) # turn NaN into 0 and actual values into 1
binary_h_df = pd.merge(bin_df, header_averages_df["day"], left_index=True, right_index=True)
binary_h_df = pd.merge(binary_h_df, header_averages_df["Well"], left_index=True, right_index=True)

# now we have all the headers for every day in one LONG df

#plot
start = 0
end = 0
offset = 0
plt.clf()
plt.figure(figsize=(10, 6))
for wells in Hwells:
  #reconfigure dataframe for current well
  group_df = binary_h_df[binary_h_df['Well'] == wells]

  group_df = group_df.sort_values(by='day')  # Sort by day

  # Plot the section
  plt.plot(group_df['day'], group_df[param]  + offset, label=f'{wells}')
  offset += 2

# label x every 10 days to clean up chart
sub_df = header_averages_df.iloc[0: header_averages_df['Well'].value_counts().get(Hwells[3], 0)]
xticks = sub_df['day'][::15]
plt.xticks(ticks=xticks, labels=xticks)
plt.xticks(rotation= 45)

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Value')
plt.title(f'Header Well Performance Over Time in {param}')
plt.legend()
plt.grid(False)
plt.show()


# binary_h_df