# To ignore the warnings

import warnings
warnings.filterwarnings('ignore')

# importing necessary packages

import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

## . Demo on strptime function
day = datetime.strptime('february/1-2019','%B/%d-%Y')
print(day)
print(day.month)


newdate = 'Jan-19-22'
datetime.strptime(newdate,'%b-%y-%d')


def parser(x):
    return datetime.strptime(x,'%Y-%m')
sales = pd.read_csv(r'C:\Users\JanakiRam\Desktop\ML_datamites\sales-cars.csv',index_col=0, parse_dates=[0] ,date_parser=parser)


sales.head()

sales.plot()


# Stationary means mean, variance and covariance is constant over periods.




#ACF -> Auto Colleration Function 
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(sales)
plt.show()


# ### Converting series to stationary

sales.head()

sales.shift(1)




sales_diff = sales.diff(periods=1)


# integrated of order 1, denoted by d (for diff), one of the parameter of ARIMA model
sales.head()





sales_diff = sales_diff[1:]
sales_diff.head()



plot_acf(sales_diff)
plt.show()




sales_diff.plot()


X = sales.values.astype('float')
train = X[0:27] # 27 data as train data
test = X[26:]  # 9 data as test data
predictions = []



train.size


# # ARIMA model

from statsmodels.tsa.arima_model import ARIMA



#p,d,q  p = periods taken for autoregressive model
#d -> Integrated order, difference
# q periods in moving average model

model_arima = ARIMA(train,order=(3,2,4))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)




predictions= model_arima_fit.forecast(steps=15)[0]
predictions




plt.plot(test, color='green')
plt.plot(predictions,color='red')
plt.show()




