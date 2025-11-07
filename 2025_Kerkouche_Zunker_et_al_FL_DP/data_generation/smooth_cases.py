import pandas as pd
import os

extended = True
year = 2022
cwd = os.getcwd()
path_csv = os.path.join(cwd, 'casedata')
filename = "cases_agg_" + str(year)
if extended:
    filename += "_extended"
window_size = 7

df = pd.read_csv(path_csv + filename + ".csv")
df = df.groupby(['Date', 'ID_County', 'ID_Community'],
                as_index=False)['Count'].sum()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['ID_County', 'ID_Community', 'Date'])


df_smooth = df.copy()
df_smooth['Count'] = df_smooth.groupby(['ID_County', 'ID_Community'])['Count'].transform(
    lambda x: x.rolling(window_size, min_periods=1, center=True).mean())
df_smooth = df_smooth.sort_values(by=['ID_County', 'ID_Community', 'Date'])
df_smooth = df_smooth.reset_index(drop=True)

# save to csv
filename = filename + "_ma" + str(window_size) + ".csv"
df_smooth.to_csv(path_csv + filename, index=False)
