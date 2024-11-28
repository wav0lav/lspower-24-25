import pandas as pd
import numpy as np
# import tensorflow as tf
import os
import random
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import dates as d
import re
# from matplotlib import pylab
# from matplotlib.colors import ListedColormap

from individualFilteringFuncs import intersection, data_filtering_individual, find_yellow_times, get_time_period, broken_list, find_intersection, find_union, matthews_correlation

from dataFIltrationMain import dataFiltrationMain


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


## LEACHATE WELLS ##
#lc_averages_df
LCwells = lc_averages_df["Well"].unique()
midlength = round(len(LCwells)/2)
LCwells1 = LCwells[:midlength]
LCwells2 = LCwells[midlength:]
# loop thru for every header
desired_valsLC_df = lc_averages_df.loc[:, [param]]
days = lc_averages_df["day"]
bin_df = desired_valsLC_df.notna().astype(int) # turn NaN into 0 and actual values into 1
binary_lc_df = pd.merge(bin_df, lc_averages_df["day"], left_index=True, right_index=True)
binary_lc_df = pd.merge(binary_lc_df, lc_averages_df["Well"], left_index=True, right_index=True)

# now we have all the well data for every day in one LONG df



#plot
offset = 0
plt.clf()
plt.figure(figsize=(15, 11))
for wells in LCwells1:
  #reconfigure dataframe for current well
  group_df = binary_lc_df[binary_lc_df['Well'] == wells]

  group_df = group_df.sort_values(by='day')  # Sort by day

  # Plot the section
  plt.plot(group_df['day'], group_df[param] * 2 + offset, label=f'{wells}')
  offset += 5


# label x every 10 days to clean up chart
sub_df = lc_averages_df.iloc[0: lc_averages_df['Well'].value_counts().get(LCwells[0], 0)]
xticks = sub_df['day'][::20]
plt.xticks(ticks=xticks, labels=xticks)
plt.xticks(rotation=45)
# I don't know why the ticks are messed up

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Value')
plt.title(f'TCLC Well Performance Over Time {param}')
plt.legend()
plt.grid(False)
plt.show()


## LEACHATES WELLS ## 

offset = 0
plt.clf()
plt.figure(figsize=(15, 11))
for wells in LCwells2:
  #reconfigure dataframe for current well
  group_df = binary_lc_df[binary_lc_df['Well'] == wells]

  group_df = group_df.sort_values(by='day')  # Sort by day

  # Plot the section
  plt.plot(group_df['day'], group_df[param] * 2 + offset, label=f'{wells}')
  offset += 5


# label x every 10 days to clean up chart
sub_df = lc_averages_df.iloc[0: lc_averages_df['Well'].value_counts().get(LCwells[0], 0)]
xticks = sub_df['day'][::10]
plt.xticks(ticks=xticks, labels=xticks)
plt.xticks(rotation=45)
# I don't know why the ticks are messed up

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Value')
plt.title(f'TCLC Well Performance Over Time {param}')
plt.legend()
plt.grid(False)
plt.show()




## GAS WELLS ##
# Function to standardize names by removing leading zeros
def standardize_name(name):
    return re.sub(r'-(0+)', '-', name)


#gw_averages_df
GWwells = gw_averages_df["Well"].unique()


#standardize gw wells
standardized_GWwells = list(set([standardize_name(name) for name in GWwells]))

# take 1/2 of them
midlen = round(len(standardized_GWwells)/2)
GWwells1 = standardized_GWwells[:midlen]
GWwells2 = standardized_GWwells[midlen:]

# loop thru for every header
desired_valsGW_df = gw_averages_df.loc[:, [param]]
days = gw_averages_df["day"]
bin_df = desired_valsGW_df.notna().astype(int) # turn NaN into 0 and actual values into 1

#merge all necessary columns into 1 dataframe
binary_gw_df = pd.merge(bin_df, gw_averages_df["day"], left_index=True, right_index=True)
binary_gw_df = pd.merge(binary_gw_df, gw_averages_df["Well"], left_index=True, right_index=True)

#standardize the names of wells
binary_gw_df['standardized_GWwells'] = binary_gw_df['Well'].apply(standardize_name)

#plot


offset = 0
plt.clf()
plt.figure(figsize=(15, 11))
for wells in GWwells1:
  #reconfigure dataframe for current well
  group_df = binary_gw_df[binary_gw_df['standardized_GWwells'] == wells]

  group_df = group_df.sort_values(by='day')  # Sort by day

  # Plot the section
  plt.plot(group_df['day'], group_df[param] * 2 + offset, label=f'{wells}')

  offset += 5

# label x every 10 days to clean up chart
sub_df = binary_gw_df.iloc[0: binary_gw_df['standardized_GWwells'].value_counts().get(GWwells[1], 0)]
xticks = sub_df['day'][::10]
plt.xticks(ticks=xticks, labels=xticks)
plt.xticks(rotation=45)

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Value')
plt.title(f'TCGW Well Performance Over Time {param}')
plt.legend()
plt.grid(False)
plt.show()

# Unhelpful, but it seems like their system was down/everything was broken on 1-12-2023


