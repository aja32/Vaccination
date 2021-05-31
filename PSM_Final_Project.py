# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:27:26 2021

@author: THINKPAD
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import scipy.stats as stats
import seaborn as sns
from pandas import Timestamp


# Importing Dataset
dataset_imported = pd.read_csv("country_vaccinations_by_manufacturer.csv")
dataset_imported_transormation = dataset_imported


Temporary_dataset = dataset_imported.loc[dataset_imported["location"]=="Germany"]
Temporary_dataset = Temporary_dataset.reset_index(drop=True)

Temporary_dataset_chile = dataset_imported.loc[dataset_imported["location"]=="Chile"]
Temporary_dataset_chile = Temporary_dataset.reset_index(drop=True)


#Preparing new datasets for each vaccine alone- in germany
Temporary_dataset_vaccine_Moderna = Temporary_dataset.loc[Temporary_dataset["vaccine"]=="Moderna"]
Temporary_dataset_vaccine_Moderna = Temporary_dataset_vaccine_Moderna.reset_index(drop=True)


Temporary_dataset_vaccine_Pfizer = Temporary_dataset.loc[Temporary_dataset["vaccine"]=="Pfizer/BioNTech"]
Temporary_dataset_vaccine_Pfizer = Temporary_dataset_vaccine_Pfizer.reset_index(drop=True)


Temporary_dataset_vaccine_Oxford = Temporary_dataset.loc[Temporary_dataset["vaccine"]=="Oxford/AstraZeneca"]
Temporary_dataset_vaccine_Oxford = Temporary_dataset_vaccine_Oxford.reset_index(drop=True)

Temporary_dataset_vaccine_Sinovac = Temporary_dataset.loc[Temporary_dataset["vaccine"]=="Sinovac"]
Temporary_dataset_vaccine_Sinovac = Temporary_dataset_vaccine_Sinovac.reset_index(drop=True)

Temporary_dataset_vaccine_Johnson = Temporary_dataset.loc[Temporary_dataset["vaccine"]=="Johnson&Johnson"]
Temporary_dataset_vaccine_Johnson = Temporary_dataset_vaccine_Johnson.reset_index(drop=True)


# Plotting graph for comparison of different vaccines in Germany
ax = plt.gca()
Temporary_dataset_vaccine_Moderna.plot(kind='line',x='date',y='total_vaccinations',label = "Moderna",lw = 0.75 , ax=ax)
Temporary_dataset_vaccine_Pfizer.plot(kind='line',x='date',y='total_vaccinations', color='red', label = "Pfizer",lw = 0.75, ax=ax)
Temporary_dataset_vaccine_Oxford.plot(kind='line',x='date',y='total_vaccinations',color = "green", label = "Oxford",lw = 0.75, ax=ax)
plt.yscale("log")

plt.xlabel('Date')
# naming the y axis
plt.ylabel('Total vacination')
  
# giving a title to my graph
plt.title('Comparison between vaccines used for vaccination in Germany.')
plt.xticks(rotation=45)
plt.show()

# Drawing scatter plot to check outliers.
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(Temporary_dataset.index.values, Temporary_dataset['total_vaccinations'])
ax.set_xlabel('date')
ax.set_ylabel('total vaccination')
plt.xticks(rotation=45)
plt.title("Scatter plot for total vaccinated people in Germany using different vaccine types.")
plt.show()

# Drawing distribution histogram for amount of vaccination
plt.figure()
plt.hist(Temporary_dataset.iloc[0:100,3],edgecolor = "black",color = "blue",bins = 100)
plt.title("Histogram shows the Vaccination in Germany")
plt.xlabel("Vaccination")
plt.ylabel("Count")
plt.xscale("log")
plt.show()



data = Temporary_dataset.iloc[0:100,3].tolist()
weights = np.ones_like(data)/len(data)

# Drawing Probability mass function histogram
plt.hist(data, weights=weights, color="blue",edgecolor = "black",bins=100, alpha=0.5)
plt.ylabel("Probability")
plt.xlabel("Outcome")
plt.title("Probability mass function")
plt.xscale("log")
plt.show()


statistics.mean(data)
print("Mean of the sample is % s " %(statistics.mean(data)))
print("Standard Deviation of the sample is % s "%(statistics.stdev(data)))
stats.skew(data, bias=False)
#stats.kurt(data, bias=False)
print("Skewness of the sample is % s " %(stats.skew(data, bias=False)))
print("Kurtosis of the sample is % s "%(stats.kurtosis(data, bias=False)))


## Done to merge our data based on Month, so that we can compare location and vaccination amount based on Month.

play = dataset_imported
play["Month"] = play.iloc[:,1].apply(lambda x: Timestamp(x).strftime("%B"))
New_Data_Frame = play.pivot_table(index = "location",values = "total_vaccinations",aggfunc = "mean",columns = "Month")
New_Data_Frame = New_Data_Frame[["December","January","February","March","April"]]



## Draw Heatmap
sns.heatmap(New_Data_Frame, annot=False, cmap="coolwarm")
plt.title("Heatmap shows average vaccination")
plt.show()

## Transformation of DVS from categorical to numerical values.

type = {'Pfizer/BioNTech': 1, 'Moderna': 2,'Oxford/AstraZeneca': 3,'Sinovac': 4 ,'Johnson&Johnson':5}
type_2 = {"Chile": 1, 'Czechia':2,'Germany':3,'Iceland':4,'Italy':5,'Latvia':6,'Lithuania':7,'Romania':8,'United States':9}
dataset_imported_transormation.vaccine = [type[item] for item in dataset_imported_transormation.vaccine]
dataset_imported_transormation.location =  [type_2[item] for item in dataset_imported_transormation.location]
dataset_imported_transormation = dataset_imported_transormation.drop(columns = ["date"])

# Drawing scatter matrix
axes = pd.plotting.scatter_matrix(dataset_imported_transormation, alpha=0.5, diagonal='kde')

# Adding correlation to our scatter matrix
corr = dataset_imported_transormation.corr(method='pearson').values
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
plt.show()





