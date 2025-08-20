
import os
import json
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
from datetime import datetime


def get_random_age(age, df_city, gender):
    """
    This function samples a random age within a specified age group. Raises a
    ValueError: If the age group is not recognized.

    Parameters:
    age (str): The age group. It should be one of the following:
               'A00-A04', 'A05-A14', 'A15-A34', 'A35-A59', 'A60-A79', 'A80+', 'unbekannt'.
    df_city (DataFrame): A pandas DataFrame containing population data. 
                         Each column should be named in the format "{gender}_{age_group}" 
                         and contain the population for that gender and age group.
    gender (str): The gender. It should match the gender used in the column names of df_city.

    Returns:
    r_age (int): A random age within the specified age group.
    """
    population = [1]
    if age == 'A00-A04':
        age_groups = ["0-2", "3-5"]
        population = [df_city[f"{gender}_{age_group}"].sum()
                      for age_group in age_groups]
        # multiply the last entry by 2/3 to account for the fact that the age group is 0-4
        population[-1] = population[-1] * 2 / 3

    elif age == 'A05-A14':
        age_groups = ["3-5", "6-9", "10-14"]
        population = [df_city[f"{gender}_{age_group}"].sum()
                      for age_group in age_groups]
        population[0] = population[0] * 1 / 3

    elif age == 'A15-A34':
        age_groups = ["15-17", "18-19", "20-24", "25-29", "30-34"]
        population = [df_city[f"{gender}_{age_group}"].sum()
                      for age_group in age_groups]

    elif age == 'A35-A59':
        age_groups = ["35-39", "40-44", "45-49", "50-54", "55-59"]
        population = [df_city[f"{gender}_{age_group}"].sum()
                      for age_group in age_groups]

    elif age == 'A60-A79':
        age_groups = ["60-64", "65-74", "75-99"]
        population = [df_city[f"{gender}_{age_group}"].sum()
                      for age_group in age_groups]
        population[-1] = population[-1] * 1 / 5

    elif age == 'A80+':
        age_groups = ["75-99"]
        population = [df_city[f"{gender}_{age_group}"].sum()
                      for age_group in age_groups]
        population[0] = population[0] * 4 / 5

    elif age == 'unbekannt':
        age = 'A00-A99'
        age_groups = ["0-2", "3-5", "6-9", "10-14", "15-17", "18-19", "20-24", "25-29",
                      "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-74", "75-99"]
        population = [df_city[f"{gender}_{age_group}"].sum()
                      for age_group in age_groups]
    else:
        raise ValueError(f"Unknown age group: {age}")

    pop_probability = [pop / sum(population) for pop in population]
    age = random.choices(age_groups, weights=pop_probability, k=1)[0]
    age = age.split('-')
    age = [int(i) for i in age]
    r_age = random.randint(age[0], age[1])
    return r_age


def find_age_group(age, age_groups):
    """
    This function determines the age group of a given age.

    Parameters:
    age (int): The age to find the age group for.
    age_groups (list of str): A list of age groups. Each age group should be a string in the format "start-end", 
                              where "start" and "end" are the starting and ending ages of the age group.

    Returns:
    age_group (str): The age group that the age falls into. If the age does not fall into any of the provided age groups, 
                     the function returns "Age out of range".

    """
    for age_group in age_groups:
        # Split the age group into start and end ages
        start_age, end_age = age_group.split('-')
        start_age, end_age = int(start_age), int(end_age)

        # Check if the age falls within the range
        if start_age <= age <= end_age:
            return age_group
    return "Age out of range"


def get_random_region(age, df_city, gender):
    """
    This function selects a random region based on the age and gender of the population.

    Parameters:
    age (int): The age of the individual.
    df_city (DataFrame): A pandas DataFrame containing population data. 
                         Each column should be named in the format "{gender}_{age_group}" 
                         and contain the population for that gender and age group.
    gender (str): The gender of the individual. It should match the gender used in the column names of df_city.

    Returns:
    region (str): The randomly selected region. The region is selected based on the distribution of the population 
                  in the age group and gender of the individual.

    """
    age_groups = ["0-2", "3-5", "6-9", "10-14", "15-17", "18-19", "20-24", "25-29",
                  "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-74", "75-99"]
    # age is a number between 0 and 99, so we need to find the corresponding age group
    age_group = find_age_group(age, age_groups)
    column = gender + "_" + age_group

    data_col = df_city[column].values
    sum_data = sum(data_col)
    data_col = [data / sum_data for data in data_col]

    indx = random.choices(range(len(data_col)), weights=data_col, k=1)[0]

    return df_city['community_id'][indx]


#################### Global Vars (directories,..) ###################
dir = os.path.dirname(__file__)
# file from https://www.regionalstatistik.de/genesis//online?operation=table&code=12411-02-03-5&bypass=true&levelindex=1&levelid=1711452634060#abreadcrumb
path_community = os.path.join(dir, "12411-02-03-5.xlsx")
path_case_data = os.path.join(dir, "CaseDataFull.json")


#################### Excel file ###################
df = pd.read_excel(path_community)

age_groups = ["0-2", "3-5", "6-9", "10-14", "15-17", "18-19", "20-24", "25-29",
              "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-74", "75-99", "Sum"]
genders = ["M", "W"]
cols = ["date", "id", "name", "total_Sum"] + \
    [f"{gender}_{age_group}" for gender in genders for age_group in age_groups]

age_groups_rki = ["0-4", "5-14", "15-34", "35-59", "60-79", "+80"]

# deletes unneccessary rows
df = df[8:-5]

# delete the total columns, since we only look at the gender specific data
df = df.drop(df.columns[3:20], axis=1)
df.columns = cols

# delete evere row where total_Sum is '-' or '.'
df = df[df["total_Sum"] != '-']
df = df[df["total_Sum"] != '.']

# replace all '-' and '.' with 0 and convert to float
for i in range(4, len(cols)):
    df[cols[i]] = df[cols[i]].replace('-', 0)
    df[cols[i]] = df[cols[i]].replace('.', 0)
    df[cols[i]] = df[cols[i]].astype(float)


# delete all entries in id, which are only one or two characters long (to delete federal states etc.)
# except for Berlin and Hamburg
indx_2 = df[df["id"] == 2].index
df.loc[indx_2, "id"] = 2000
indx_11 = df[df["id"] == 11].index
df.loc[indx_11, "id"] = 11000
df = df[df["id"].astype(str).str.len() > 3]


# county_id is generated by concatenating the first three columns
list_of_cities = []
for index, row in df.iterrows():
    county_id = str(row.iloc[1])
    community_id = 0
    # check if the id is longer than 5, if yes, it contains a community id
    if len(str(row.iloc[1])) > 5:
        # the first five characters are the county id, the last three are the community id
        community_id = str(row.iloc[1])[-3:]
        county_id = str(row.iloc[1])[:-3]
    name = row.iloc[2]
    # delete all leading and trailing spaces
    name = name.strip()
    list_of_cities.append(
        {"county_id": int(county_id), "community_id": int(community_id), "name": name, **{cols[i]: row.iloc[i] for i in range(4, len(cols))}})

# list of entries to df
df = pd.DataFrame(list_of_cities)
# sort by id and second by community_id and reset index
df = df.sort_values(by=["county_id", "community_id"]).reset_index(drop=True)

#################### Case data ###################
with open(path_case_data, "r") as f:
    data = json.load(f)

sum_raw_data = 0

date_format = "%Y-%m-%d"
start_date = datetime.strptime("2022-03-01", date_format)
end_date = datetime.strptime("2022-04-01", date_format)

# # transform into df
# df_data = pd.DataFrame(data)
# # only id 7312
# df_data_7312 = df_data[df_data["IdLandkreis"] == 7312]
# # delete all entries outside the date range
# df_data_7312 = df_data_7312[(df_data_7312["Refdatum"] >= start_date.strftime(date_format)) & (
#     df_data_7312["Refdatum"] <= end_date.strftime(date_format))]

# start_date = datetime.strptime("2020-11-01", date_format)
# end_date = datetime.strptime("2020-12-01", date_format)

new_case_data = []
new_case_data_agg = []
count_unknown_age = 0
for entry in tqdm(data):
    entry_date = datetime.strptime(entry['Refdatum'], date_format)
    if entry_date < start_date or entry_date > end_date:
        continue
    num_cases = entry['AnzahlFall']
    sum_raw_data += num_cases
    age = entry['Altersgruppe']
    gender = entry['Geschlecht']
    county_id = int(entry["IdLandkreis"])
    # merge berlin
    ids_berlin = [11001, 11002, 11003, 11004, 11005,
                  11006, 11007, 11008, 11009, 11010, 11011, 11012]
    if county_id in ids_berlin:
        county_id = 11000
    df_city = df[df["county_id"] == county_id].reset_index(drop=True)
    for case in range(num_cases):
        if gender == 'unbekannt':
            gender = random.choice(['M', 'W'])

        if len(df_city) > 2:
            df_city = df_city[df_city["community_id"]
                              != 0].reset_index(drop=True)

        if age == 'unbekannt':
            count_unknown_age += 1

        r_age = get_random_age(age, df_city, gender)
        r_region = df_city['community_id'][0]

        if (len(df_city) > 1):
            r_region = get_random_region(r_age, df_city, gender)

        new_entry = {}
        new_entry["Date"] = entry["Refdatum"]
        new_entry["ID_County"] = county_id
        new_entry["ID_Community"] = r_region
        new_entry["Age"] = r_age
        new_entry["Gender"] = gender
        new_case_data.append(new_entry)

        # check if theres already an entry with the same age and region, if yes, increase the count by 1, else add a new entry
        new_entry_copy = new_entry.copy()
        found = False
        for i in range(len(new_case_data_agg) - case, len(new_case_data_agg)):
            if new_case_data_agg[i]["ID_Community"] == r_region and new_case_data_agg[i]["Age"] == r_age:
                new_case_data_agg[i]["Count"] += 1
                found = True
                break
        if not found:
            new_entry_copy["Age"] = entry['Altersgruppe']
            new_entry_copy["Count"] = 1
            new_case_data_agg.append(new_entry_copy)


#################### Write new Case data ###################
def default(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError

# path_new_agg = os.path.join(dir, "cases_agg.json")
# with open(path_new_agg, "w") as f:
#     json.dump(new_case_data_agg, f, indent=2, default=default)

# # delete all count entries in new_case_data
# for entry in new_case_data:
#     entry.pop("Count", None)
# path_new = os.path.join(dir, "cases_individual.json")
# with open(path_new, "w") as f:
#     json.dump(new_case_data, f, indent=2, default=default)


df_new_case_data_agg = pd.DataFrame(new_case_data_agg)
df_new_case_data = pd.DataFrame(new_case_data)
# Write the data to CSV files
path_new_agg = os.path.join(dir, "cases_agg_2022.csv")
df_new_case_data_agg.to_csv(path_new_agg, index=False)

# delete all count entries in new_case_data
for entry in new_case_data:
    entry.pop("Count", None)
path_new = os.path.join(dir, "cases_individual_2022.csv")
df_new_case_data.to_csv(path_new, index=False)

print(f"Unknown age: {count_unknown_age}")


#################### Validate results ###################
sum_cases_agg = 0
for entry in new_case_data_agg:
    sum_cases_agg += entry["Count"]


sum_cases_individual = len(new_case_data)

if sum_cases_agg != sum_raw_data:
    print(
        f"Sum of cases in new_case_data_agg ({sum_cases_agg}) does not match sum of cases in raw data ({sum_raw_data})")

if sum_cases_individual != sum_raw_data:
    print(
        f"Sum of cases in new_case_data ({sum_cases_individual}) does not match sum of cases in raw data ({sum_raw_data})")
