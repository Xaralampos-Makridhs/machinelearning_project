import pandas as pd #data manipulation
import matplotlib.pyplot as plt #data and prediction visualization
import numpy as np #mathematical operations on arrays
from sklearn.model_selection import train_test_split #splits data into testing and training sets
from sklearn.linear_model import LinearRegression #importing the linear regression algorithm
from sklearn.metrics import mean_squared_error, r2_score #for evaluating the model's performance

#loading the dataset
df=pd.read_csv('train.csv')

#selecting specific columns for analysis
df=df[['GrLivArea','SalePrice']]

#removing missing values
df=df.dropna()

#calculating the correlation between GrLivArea and SalePrice
correlation=df.corr()
print(f"Correlation between GrLivArea and Sales is {correlation.iloc[0,1]}")

#splitting the data
X=df[['GrLivArea']] #independent variable
y=df['SalePrice'] #dependent variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) #margin 42

#creating and training the model
model=LinearRegression()
model.fit(X_train,y_train)

#making predictions
y_predict=model.predict(X_test)

#model evaluation
r2=r2_score(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
rmse=np.sqrt(mse)

#displaying results
print(f'r**2: {r2}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

#creating the graph
plt.scatter(X_test,y_test,color='blue', label='Real Prices')
plt.plot(X_test,y_predict,color='red', linewidth=2,label='Predicted line')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.title('Linear Regression: GrLivArea vs SalePrice')
plt.legend()
plt.show()
