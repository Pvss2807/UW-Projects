#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('C:\\Users\\IdeasClinicCoops\\Downloads\\45kgdataset.xlsx')
df.to_csv('Training_Data.csv', index=False)
df.dropna(how='all', inplace=True)
df.dropna(axis=1, how='all', inplace=True)
df.columns = df.iloc[0]

# Drop the first row (since it's now used as column names)
df = df.iloc[1:].reset_index(drop=True)
df
# Replace 'dataset.csv' with the actual filename of your dataset


# In[3]:


df['Stress'] = df['Force in N']  / 6491
df['Stress'] = 1/df['Stress']
df


# In[5]:


# Sample stress values (replace this with your dataset)
stress_values = df['Stress']  # Example: stress values ranging from 10 to 100
fatigue_limit = 20

# Calculate the ratio of stress to fatigue limit (Ri)
df['Ri'] = df['Stress'] / fatigue_limit

# Calculate the number of cycles to failure (Ni) using Miner's Rule
df['Ni'] = 1 / df['Ri']

# Create a time column
df['Time'] = df.index + 100 # Using data points' indices as time values (time interval is 100)

# Plot the stress-time graph
plt.plot(df['Time'], df['Stress'], marker='o')
plt.xlabel('Time (Interval)')
plt.ylabel('Stress (MPa)')
plt.title('Stress-Time Graph')
plt.grid(True)
plt.show()

stress_time_df = pd.DataFrame({'Time (Interval)': df['Time'],
                               'Stress (MPa)': df['Stress']})
# Step 2: Plot the S-N Curve
# sns.regplot(x = random_fatigue_cycles, y = stress_values, color='red', label='Regression Line')
plt.scatter(df['Ni'], stress_values)
plt.xlabel('Fatigue Cycles')
plt.ylabel('Stress (MPa)')
plt.title('S-N Curve')
plt.grid(True)
plt.show()

sn_curve = pd.DataFrame({'Fatigue Cycles': df['Ni'],
                               'Stress (MPa)': df['Stress']})
stress_numpy = df['Stress'].to_numpy()


fatigue = df['Ni'].to_numpy()

dic = {}

for i in range(0, len(fatigue) , 8):
    dic[stress_numpy[i]] = fatigue[i]
    
time_numpy = df['Time'].to_numpy()
time_interval_numpy = [time_numpy[i] for i in range(0, len(time_numpy), 4)]
s_n = [key for key,value in dic.items()]
f_n = [value for key,value in dic.items()]

correspond_time = []
for i in range(0, len(time_interval_numpy)-1):
    correspond_time.append(time_interval_numpy[i+1]-time_interval_numpy[i])

dff1 = pd.DataFrame(stress_numpy)
dff2 = pd.DataFrame(fatigue)
dff3 = pd.DataFrame(correspond_time)
dataset = pd.DataFrame({'Stress': s_n, 'Fatigue': f_n}, columns=['Stress', 'Fatigue'])
plt.plot(dataset['Fatigue'], dataset['Stress'], marker='o')
plt.xlabel('Fatigue Cycles')
plt.ylabel('Stress (MPa)')
plt.title('S-N Curve')
plt.grid(True)
plt.show()
dataset
# Convert the 'Stress (MPa)' column in the S-N curve DataFrame to numeric
sn_curve['Stress (MPa)'] = pd.to_numeric(sn_curve['Stress (MPa)'], errors='coerce')

# Drop any rows with NaN values after conversion
sn_curve.dropna(subset=['Stress (MPa)'], inplace=True)

# Calculate the peak stress value for each stress-time cycle
peak_stress_values = stress_time_df.groupby('Time (Interval)')['Stress (MPa)'].max()

# Find the corresponding fatigue cycles for each peak stress value
corresponding_fatigue_cycles = []
for peak_stress in peak_stress_values:
    closest_stress_row = sn_curve.iloc[(sn_curve['Stress (MPa)'] - peak_stress).abs().idxmin()]
    corresponding_fatigue_cycles.append(closest_stress_row['Fatigue Cycles'])

# Calculate the number of cycles applied (Nf) for each stress-time cycle (assumed to be equal to the time interval)
cycles_applied = stress_time_df['Time (Interval)'].values

# Calculate the damage using Miner's Rule
damage = corresponding_fatigue_cycles / cycles_applied

# Calculate the cumulative damage
cumulative_damage = damage.cumsum()

if (cumulative_damage >= 0.9).any():
    print("WARNING: The chair is about to get damaged. Consider replacing or repairing it.")
    
# Print the results
print("Peak Stress Values for Each Stress-Time Cycle:")
print(peak_stress_values)
print("\nCorresponding Fatigue Cycles for Each Stress-Time Cycle:")
print(corresponding_fatigue_cycles)
print("\nNumber of Cycles Applied for Each Stress-Time Cycle:")
print(cycles_applied)
print("\nDamage for Each Stress-Time Cycle:")
print(damage)
print("\nCumulative Damage:")
print(cumulative_damage)

