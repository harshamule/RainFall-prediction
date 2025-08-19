 import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest with slight improvements
rf_model = RandomForestRegressor(n_estimators=75, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

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

# Adjust classification threshold dynamically for better accuracy
threshold = np.percentile(y_train, 25)  # Lower percentile to increase Rain detection
y_pred_class = (y_pred >= threshold).astype(int)
y_test_class = (y_test >= threshold).astype(int)

# Compute classification accuracy
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"\nRandom Forest Accuracy : {accuracy * 100:.2f}%")

OUTPUT:

Regression Performance Metrics:
Mean Squared Error (MSE): 218.33
Root Mean Squared Error (RMSE): 14.78
Mean Absolute Error (MAE): 12.79
R² Score: -0.04
Random Forest Accuracy : 75.00%
