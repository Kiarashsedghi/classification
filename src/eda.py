# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from discretize import discretize


datasetPath="./dataset.xls"


# %%
plt.rcParams['figure.figsize'] = [5, 5]

# %%
data = pd.read_excel(datasetPath)

# %%
data.drop('Unnamed: 0', axis = 1, inplace = True)




# %%
data.isna().sum()

# %%
# fill the empty values with the mean value of the column Close_Value
close_value_mean =data['Close_Value'].mean()
data['Close_Value'].fillna(close_value_mean, inplace = True)

# %%
ax = sns.boxplot(data=data['Close_Value'], orient="h")

# %%
#precentage
data['Product'].value_counts(normalize=True)

# %%
data['Product'].value_counts(normalize=True).plot.barh()

# %%
data.groupby('Product')['Close_Value'].mean().plot.bar()
plt.show()

# %%
data['Stage'] = data['Stage'].astype('category')
data.dtypes

# %%
discretize(data)

# data['Stage']=data['Stage'].cat.codes
#2->won
#1->lost
#0-> in progress

# %%


# %%
data.groupby('Product')['Stage'].mean().plot.bar()

# %%
result = pd.pivot_table(data=data, index='Agent', columns='Product',values='Stage')

# %%
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117 )
plt.show()

# %%
