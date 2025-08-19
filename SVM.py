 import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression

# Load dataset
file_path = "rainfall_prediction.csv"
df = pd.read_csv(file_path)

# Handle missing values
df = df.dropna()

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split features and target
X = df.drop(columns=['Rainfall'])
y = df['Rainfall']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection (reduce overfitting)
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train SVM model with lower C and gamma
svm_model = SVR(kernel='rbf', C=0.1, gamma=0.01)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Regression Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nRegression Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Adjust classification threshold dynamically
threshold = np.percentile(y_train, 30)  # Set threshold to 30th percentile
y_pred_class = (y_pred >= threshold).astype(int)
y_test_class = (y_test >= threshold).astype(int)

# Compute classification accuracy
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"SVM Accuracy: {accuracy * 100:.2f}%")

OUTPUT:
Regression Performance Metrics:
Mean Squared Error (MSE): 209.93
Root Mean Squared Error (RMSE): 14.49
Mean Absolute Error (MAE): 12.58
R² Score: -0.00
SVM Accuracy: 70.70%
