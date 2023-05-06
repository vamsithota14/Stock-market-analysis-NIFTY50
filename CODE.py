import pandas as pd
x=pd.read_csv("D:\\New data\\DATA1.csv")
df= pd.DataFrame(x)

Y = df['Nifty50_Open']
X = df[[ 'SPOTRATE_CLOSE','SR_OUTPUT',
       'SP500_CLOSE', 'SP500_OUTPUT','NASDAQ100_CLOSE',
       'NASDAQ_OUTPUT', 'CRUDE_OIL_CLOSE','CRUDE_OIL_OUTPUT',
       'LSE_CLOSE','LSE_OUTPUT']]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
from sklearn.linear_model import LinearRegression


#using Stat model
from statsmodels.api import OLS
model_lin = OLS.from_formula("Nifty50_Open~SPOTRATE_CLOSE+SR_OUTPUT+SP500_CLOSE+SP500_OUTPUT+NASDAQ100_CLOSE+NASDAQ_OUTPUT+CRUDE_OIL_CLOSE+CRUDE_OIL_OUTPUT+LSE_CLOSE+LSE_OUTPUT",data=df)

result_lin = model_lin.fit()
result_lin.summary()


lm = LinearRegression(copy_X=True, fit_intercept=False)

lm.fit(X_train,Y_train)
print(lm.intercept_)
print(lm.coef_)
print(lm.score(X_train,Y_train))