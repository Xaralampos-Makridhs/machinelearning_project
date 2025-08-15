import pandas as pd #επεξεργασια δεδομενων
import matplotlib.pyplot as plt #γραφικη αναπαρασταση δεδομενων και προβλεψεων
import numpy as np #υπολογισμος μαθηματικων πραξεων με πινακες
from sklearn.model_selection import train_test_split #ξεχωριζει τα δεδομενα σε δεδομενα testing και δεδομενα training
from sklearn.linear_model import LinearRegression #εισαγουμε τον αλγοριθμο γραμμικης παλινδρομησης
from sklearn.metrics import mean_squared_error, r2_score #για να υπολογισουμε την αποδοση του αλγοριθμου

#φορτωση dataset
df=pd.read_csv('train.csv')

#επιλογη συγκεκριμενων στηλων για την αναλυση
df=df[['GrLivArea','SalePrice']]

#Αφαιρεση κενων γραμμων
df=df.dropna()

#υπολογισμος συσχετισης μεταξυ GrLiveArea SalesPrice
correlation=df.corr()
print(f"Correlation between GrLiveArea and Sales is {correlation.iloc[0,1]}")

#διαχωρισμος δεδομενων
X=df[['GrLivArea']] #ανεξαρτητη τιμη
y=df['SalePrice'] #εξαρτημενη τιμη
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) #margin 42

#Δημιουργια και εκπαιδευση μοντελου
model=LinearRegression()
model.fit(X_train,y_train)

#προβλεψη
y_predict=model.predict(X_test)

#αξιολογηση μοντελου
r2=r2_score(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
rmse=np.sqrt(mse)

#Εμφανιση απιοτελεσματων
print(f'r**2: {r2}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

#Δημιουργια γραφηματος
plt.scatter(X_test,y_test,color='blue', label='Real Prices')
plt.plot(X_test,y_predict,color='red', linewidth=2,label='Predicted line')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.title('Linear Regression: GrLivArea vs SalePrice')
plt.legend()
plt.show()

