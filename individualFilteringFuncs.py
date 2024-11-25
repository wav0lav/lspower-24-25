# THIS BLOCK HAS ALL THE FUNCTIONS FOR DATA FILTERING FOR INDIVIDUAL WELLS

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

# filters individual well data given a filename for an individual well
def data_filtering_individual(filename):
  search_path = '/content/gdrive/Shareddrives/LSPowerData/IndividualWellData'
  # Search for the file
  file_path = os.path.join(search_path, filename)
  df = pd.read_csv(file_path, header=6)

  df.replace({'NR': np.nan, 'missing': np.nan, 'F': np.nan}, inplace=True)
  df.replace({'--': np.nan}, inplace=True) #fake value can ignore and add plot limits if needed

  df.loc[:, 'CH4 (%)'] = df['CH4 (%)'].astype(float)
  df.loc[:, 'O2 (%)'] = df['O2 (%)'].astype(float)
  df.loc[:, 'LFG Temperature (F)'] = df['LFG Temperature (F)'].astype(float)
  df.loc[:, 'Bal. Gas (%)'] = df['Bal. Gas (%)'].astype(float)
  df.loc[:, 'Applied Vacuum - PA (in. H2O)'] = df['Applied Vacuum - PA (in. H2O)'].astype(float)
  df['Datetime'] = pd.to_datetime(df['Timestamp (US/Eastern)'])
  return df

  # Get Indices

# given the overall aggregated data frame + an individual well, outputs all the times the well is yellow
def find_yellow_times(df, well):
  y =  ((df[df['Follow Up Priority'] == 'Yellow'].index))
  n =  ((df[df['Well'] == well].index))
  intersect = intersection(y, n)

  # time_list is a list of days where the well has yellow readings
  time_list = []
  for i in range(len(intersect)):
    time_list.append(df.loc[intersect[i], 'day'])
  return time_list

# Find indices in individual well data of specific start and end time
def get_time_period(df, start, end):
  # Getse start + end time (!!)
  dt = df['Datetime']
  time1 = pd.to_datetime(start) #date 1
  index = np.searchsorted(dt, time1)

  # TODO: make an end variable
  time2 = pd.to_datetime(end) # date 2
  index2 = np.searchsorted(dt, time2) + 1
  return (index, index2)

# Find indices in individual well data from when well was broken
# Useful for plotting
def broken_list(df, time_list):
  dt = df['Datetime']
  broken = []
  for i in range(len(time_list)):
    time = pd.to_datetime(time_list[i]) #date 1
    broken.append(np.searchsorted(dt, time))
  return broken


def find_intersection(lists):
    """Finds the intersection of multiple lists."""

    if not lists:
        return []

    # Convert lists to sets for efficient intersection
    sets = [set(lst) for lst in lists]

    # Find the intersection using the intersection method
    result = sets[0].intersection(*sets[1:])

    return list(result)


def find_union(lists):
    """Finds the intersection of multiple lists."""

    if not lists:
        return []

    # Convert lists to sets for efficient intersection
    sets = [set(lst) for lst in lists]

    # Find the intersection using the intersection method
    result = sets[0].union(*sets[1:])

    return list(result)


def matthews_correlation(tp, tn, fp, fn):
  tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
  num = tp * tn - fp * fn
  denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  return num/denom
