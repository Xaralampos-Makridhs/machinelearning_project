import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('traffic_accidents.csv')

# Display the columns to confirm the dataset structure
print('This is the labels of the Data that we will use to train the predict model:\n')
print(df.columns)

# Define the columns to be used for the model
data_columns = [
    'crash_date', 'traffic_control_device', 'weather_condition',
    'lighting_condition', 'first_crash_type', 'trafficway_type',
    'alignment', 'roadway_surface_cond', 'road_defect', 'crash_type',
    'intersection_related_i', 'damage', 'prim_contributory_cause',
    'num_units', 'most_severe_injury', 'injuries_total', 'injuries_fatal',
    'injuries_incapacitating', 'injuries_non_incapacitating',
    'injuries_reported_not_evident', 'injuries_no_indication', 'crash_hour',
    'crash_day_of_week', 'crash_month'
]

# Convert categorical data to numeric using LabelEncoder
label_encoder = {}  # Dictionary to store LabelEncoders for each column

for column in data_columns:
    if df[column].dtype == 'object':  # Apply LabelEncoder only for categorical columns
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])  # Convert categorical data to numeric
        label_encoder[column] = le  # Store the encoder for later use

# Split the data into features (X) and target (y)
X = df.drop(columns=['most_severe_injury', 'crash_date'])  # Drop 'crash_date' and target column
y = df['most_severe_injury']  # The target is 'most_severe_injury'

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_predict = model.predict(X_test)

# Convert predicted numeric labels back to their original string labels
inverse_labels = label_encoder['most_severe_injury'].inverse_transform(y_predict)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_predict) * 100

# Print the predictions and the accuracy of the model
print('Predictions:', inverse_labels)
print('Model Accuracy:', accuracy)
