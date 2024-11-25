import pandas as pd


daily_averages_df = pd.read_csv('aggregatedDailyAverages.csv')
individual_data_df = pd.read_csv('aggregatedIndividualData.csv', low_memory=False)
merged_df = pd.read_csv('aggregatedIndividualAndDailyUpdated.csv', low_memory=False)