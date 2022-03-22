import numpy as n
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as st

pageData = pd.read_csv('./fbpageinsight.csv')

print(pageData.corr())

plt.scatter(pageData.QuotaTotalReach,pageData.QuotaPageEngagedUsers)

y = pageData.QuotaPageEngagedUsers 
x = pageData.QuotaTotalReach
x = st.add_constant(x)

modal1 = st.OLS(y,x).fit()

print(modal1.summary())

xprime1 = n.linspace(x.QuotaTotalReach.min(),x.QuotaTotalReach.max(),100)
xprime1 = st.add_constant(xprime1)

yhat1 = modal1.predict(xprime1)
plt.scatter(x.QuotaTotalReach,y)
plt.ylabel("Page Engagement")
plt.xlabel("Page Reach")
plt.plot(xprime1[:,1],yhat1)

plt.show()