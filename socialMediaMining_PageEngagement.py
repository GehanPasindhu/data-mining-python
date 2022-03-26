import numpy as n #numpy is a package for scientific computing in Python.
import matplotlib.pyplot as plt # matplotlib is a plotting library for Python.
import pandas as pd #pandas is a library for data processing and analysis in Python.
import statsmodels.api as st #statsmodels is a collection of modules for statistical computations in Python.

pageData = pd.read_csv('./fbpageinsight.csv') #read the data from the csv file.

print(pageData.corr()) #print the correlation matrix.

plt.scatter(pageData.QuotaPageEngagedUsers,pageData.LifetimeTotalLikes) #create a scatter plot of the data.

x = pageData.QuotaPageEngagedUsers #create a variable for the x-axis.
y = pageData.LifetimeTotalLikes #create a variable for the y-axis.
x = st.add_constant(x) #add a constant to the x-axis.

modal = st.OLS(y,x).fit() #create a model of the data.

print(modal.summary()) #print the summary of the model.

xprime = n.linspace(x.QuotaPageEngagedUsers.min(),x.QuotaPageEngagedUsers.max(),100) #create a variable for the x-axis.
xprime = st.add_constant(xprime) # add a constant to the x-axis.

yhat = modal.predict(xprime) #create a variable for the y-hat.
plt.scatter(x.QuotaPageEngagedUsers,y) #plot the scatter plot.
plt.xlabel("Page Engagement") # label the x-axis.
plt.ylabel("Page Likes") # label the y-axis.
plt.plot(xprime[:,1],yhat) # plot the regression line.

plt.show() #show the plot.