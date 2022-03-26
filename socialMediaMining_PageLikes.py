import numpy as n #numpy is a package for scientific computing in Python.
import matplotlib.pyplot as plt #matplotlib is a plotting library for Python.
import pandas as pd #pandas is a library for data processing and analysis in Python.
import statsmodels.api as st #statsmodels is a collection of modules for statistical computations in Python.

pageData = pd.read_csv('./fbpageinsight.csv') #read the data from the csv file.

print(pageData.corr()) #print the correlation matrix.

plt.figure() #create a figure.
hist,edge = n.histogram(pageData.LifetimeTotalLikes) #create a histogram of the data.
plt.bar(edge[:-1],hist,width=edge[1:]-edge[:1]) #plot the histogram.
plt.show() #show the histogram.