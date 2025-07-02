import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew

# Load your CSV
df = pd.read_csv("LD084 LD070 LD071 LD076 LD078 LD079 LD094 RB016.csv")

# Map machine to Tier
tier_mapping = {
    'LD070': 'Tier 4', 'LD071': 'Tier 4', 'LD076': 'Tier 4', 'LD078': 'Tier 4',
    'LD079': 'Tier 5', 'LD083': 'Tier 5', 'LD084': 'Tier 5',
    'RB016': 'Tier 3A', 'LD210': 'Tier 3A'
}
df['Tier'] = df['Machine Number'].map(tier_mapping)

# Split category into condition and pollutant
df[['Condition', 'Pollutant']] = df['Category'].str.extract(r'^(IDLE|HIGH IDLE|STALL|DPM)\s*(CO|NO|NO2|NOX)?', expand=True)

# Drop rows with missing key info
df_clean = df.dropna(subset=['Tier', 'Condition', 'Reading'])

# Summary stats
summary = (
    df_clean.groupby(['Tier', 'Condition', 'Pollutant'])['Reading']
    .agg(['mean', 'max', 'std'])
    .rename(columns={'mean': 'Mean', 'max': 'Max', 'std': 'StdDev'})
    .round(2)
    .reset_index()
)

# Add skewness
def calculate_skew(group):
    return skew(group['Reading'].dropna(), bias=False)

df_clean['Skewness'] = df_clean.groupby(['Tier', 'Condition', 'Pollutant'])['Reading'].transform(
    lambda x: calculate_skew(pd.DataFrame({'Reading': x}))
)

# Detect outliers using IQR
def detect_outliers_iqr(group):
    Q1 = group['Reading'].quantile(0.25)
    Q3 = group['Reading'].quantile(0.75)
    IQR = Q3 - Q1
    return group[(group['Reading'] < Q1 - 1.5 * IQR) | (group['Reading'] > Q3 + 1.5 * IQR)]

outliers = df_clean.groupby(['Tier', 'Condition', 'Pollutant']).apply(detect_outliers_iqr).reset_index(drop=True)

# Save summary and outliers
summary.to_csv("emissions_summary.csv", index=False)
outliers.to_csv("emissions_outliers.csv", index=False)

# Plot violin plots
sns.set(style="whitegrid")
plot_data = df_clean[df_clean['Pollutant'].notna()]
g = sns.catplot(
    data=plot_data,
    x="Pollutant", y="Reading",
    col="Condition", row="Tier",
    kind="violin", inner="box", height=4, aspect=1.2,
    sharey=False
)
g.set_titles("{row_name} - {col_name}")
g.set_axis_labels("Pollutant", "Emission Reading")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Emission Distributions by Tier and Operating Condition", fontsize=16)
plt.show()
