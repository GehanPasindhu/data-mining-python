import numpy as n #numpy is a package for scientific computing in Python.
import matplotlib.pyplot as plt #matplotlib is a plotting library for Python.   
import pandas as pd #pandas is a library for data processing and analysis in Python.
import statsmodels.api as st # statsmodels is a collection of modules for statistical computations in Python.

pageData = pd.read_csv('./fbpageinsight.csv') #read the data from the csv file.

print(pageData.corr()) #print the correlation matrix.

plt.scatter(pageData.QuotaTotalReach,pageData.QuotaPageEngagedUsers) #create a scatter plot of the data.

y = pageData.QuotaPageEngagedUsers  #create a variable for the y-axis.
x = pageData.QuotaTotalReach #create a variable for the x-axis.
x = st.add_constant(x) #add a constant to the x-axis.

modal1 = st.OLS(y,x).fit() #create a model of the data.

print(modal1.summary()) #print the summary of the model.

xprime1 = n.linspace(x.QuotaTotalReach.min(),x.QuotaTotalReach.max(),100) #create a variable for the x-axis.
xprime1 = st.add_constant(xprime1) # add a constant to the x-axis.

yhat1 = modal1.predict(xprime1) #create a variable for the y-hat.
plt.scatter(x.QuotaTotalReach,y) #plot the scatter plot.
plt.ylabel("Page Engagement") # label the y-axis.
plt.xlabel("Page Reach") # label the x-axis.
plt.plot(xprime1[:,1],yhat1) # plot the regression line.

plt.show() #show the plot.