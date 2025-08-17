import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load CSV file into a DataFrame
df = pd.read_csv('movies.csv')

# Clean numeric column: remove $ and , from Worldwide Gross and convert to float
df['Worldwide Gross'] = df['Worldwide Gross'].replace('[\$,]', '').astype(float)

# Select features (X) and target (y)
X = df[['Audience score %','Profitability','Rotten Tomatoes %','Worldwide Gross']]
y = df['Genre']

# Remove rows with missing values in important columns
df = df.dropna(subset=['Audience score %','Profitability','Rotten Tomatoes %','Worldwide Gross','Genre'])

# Recreate features and target after dropping NaN rows
X = df[['Audience score %','Profitability','Rotten Tomatoes %','Worldwide Gross']]
y = df['Genre']

# Encode target labels (Genre) into numbers (e.g. Action=0, Comedy=1, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3)

# Create and train KNN model (k=3 neighbors)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Predict the genre for a new movie (with made-up features)
new_movie = pd.DataFrame({
    'Audience score %': [85],
    'Profitability': [1.5],
    'Rotten Tomatoes %':[90],
    'Worldwide Gross': [150]
})

# Decode prediction back into original genre labels
predict_genre = le.inverse_transform(knn.predict(new_movie))
print(f"Predicted Genre for new movie: {predict_genre[0]}")
