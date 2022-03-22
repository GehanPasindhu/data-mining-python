import numpy as n
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as st

pageData = pd.read_csv('./fbpageinsight.csv')

print(pageData.corr())

plt.scatter(pageData.QuotaPageEngagedUsers,pageData.LifetimeTotalLikes)

x = pageData.QuotaPageEngagedUsers
y = pageData.LifetimeTotalLikes
x = st.add_constant(x)

modal = st.OLS(y,x).fit()

print(modal.summary())

xprime = n.linspace(x.QuotaPageEngagedUsers.min(),x.QuotaPageEngagedUsers.max(),100)
xprime = st.add_constant(xprime)

yhat = modal.predict(xprime)
plt.scatter(x.QuotaPageEngagedUsers,y)
plt.xlabel("Page Engagement")
plt.ylabel("Page Likes")
plt.plot(xprime[:,1],yhat)

plt.show()