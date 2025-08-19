from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('/content/rainfall_prediction.csv')

# Encode target variable
if data['Rainfall'].dtype == 'object':
    le = LabelEncoder()
    data['Rainfall'] = le.fit_transform(data['Rainfall'])

# Separate features and target
X = data.drop(['Previous_Rainfall', 'Temperature'], axis=1)  # Drop Multiple Columns Correctly
y = data[['Rainfall', 'Wind_Speed']]  # Select Multiple Target Columns Correctly

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Handle missing values
X = X.fillna(0)
y = y.fillna(0)

# Convert Data Types
X = X.astype(float)
y = y.astype(int)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y['Rainfall'], test_size=0.2, random_state=42)

# Create Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the classifier
dt.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt.predict(X_test)

# Calculate the accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)
Accuracy_dt = 100 - (accuracy_dt * 100)
print("Decision Tree Accuracy:", Accuracy_dt)


OUTPUT:
Accuracy:76.9
