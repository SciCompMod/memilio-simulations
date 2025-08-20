import os
import pandas as pd

# if there are zero cases for a community, there are currently no entries in the file.
# Extend the file and add entries, if there are no cases reported.

cwd = os.getcwd()
path_community_data = os.path.join(cwd, 'casedata')
path_csv = os.path.join(cwd, 'casedata')
filename = "cases_agg_2022"
df = pd.read_csv(path_csv + filename + ".csv")
df = df.groupby(['Date', 'ID_County', 'ID_Community'],
                as_index=False)['Count'].sum()
df = df.sort_values(by=['ID_County', 'ID_Community']).reset_index(drop=True)

first_date = df['Date'].min()
last_date = df['Date'].max()

unique_pairs = df[['ID_County', 'ID_Community']].drop_duplicates()
all_dates = pd.date_range(start=first_date, end=last_date).strftime('%Y-%m-%d')

rows = []
for _, row in unique_pairs.iterrows():
    county = row['ID_County']
    community = row['ID_Community']
    for date in all_dates:
        rows.append({'Date': date, 'ID_County': county,
                    'ID_Community': community})
full_df = pd.DataFrame(rows)

# merge with the original data. Missing entries are filled with 0
merged = pd.merge(
    full_df, df, on=['Date', 'ID_County', 'ID_Community'], how='left')
merged['Count'] = merged['Count'].fillna(0).astype(int)

# sort
merged = merged.sort_values(
    by=['ID_County', 'ID_Community', 'Date']).reset_index(drop=True)

# some checks...
expected_length = len(all_dates) * len(unique_pairs)
if len(merged) != expected_length:
    print(
        f"Warning: Expected {expected_length} rows, but got {len(merged)} rows.")

# save to old dir with suffix _extended
filename = filename + "_extended.csv"
merged.to_csv(path_csv + filename, index=False)
