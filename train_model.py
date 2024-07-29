import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings

# Suppress future warnings related to sparse output
warnings.filterwarnings('ignore', category=FutureWarning)

# Load the dataset
df = pd.read_csv('Statewise Consumption.csv')

# Data Cleaning
df = df.drop_duplicates()
df_clean = df.dropna()

# Feature Engineering
df_clean['Dates'] = pd.to_datetime(df_clean['Dates'], dayfirst=True, errors='coerce')
df_clean = df_clean.dropna(subset=['Dates'])

df_clean['day_of_week'] = df_clean['Dates'].dt.dayofweek
df_clean['day_of_week'] = df_clean['day_of_week'].map({0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"})
df_clean['month'] = df_clean['Dates'].dt.month
month_names = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}
df_clean['month'] = df_clean['month'].map(month_names)
df_clean['year'] = df_clean['Dates'].dt.year
df_clean['season'] = df_clean['Dates'].dt.month % 12 // 3
df_clean['season'] = df_clean['season'].map({0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'})

# Historical Averages
df_clean['historical_avg_usage'] = df_clean.groupby(['States', 'month'])['Usage'].transform('mean')

# Day Type
def get_day_type(day):
    return 'Weekend' if day in ['Saturday', 'Sunday'] else 'Weekday'

df_clean['day_type'] = df_clean['day_of_week'].apply(get_day_type)

# Extract quarter information
df_clean['quarter'] = df_clean['Dates'].dt.quarter

# Filter data for the years 2019 and 2020
df_2019 = df_clean[df_clean['year'] == 2019].copy()
df_2020 = df_clean[df_clean['year'] == 2020].copy()

# Define quarter labels
quarter_labels = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}

# Map quarter numbers to quarter labels
df_2019['quarter_label'] = df_2019['quarter'].map(quarter_labels)
df_2020['quarter_label'] = df_2020['quarter'].map(quarter_labels)

# Prepare the data for Random Forest Regressor
X = df_clean[['States', 'Regions']]
y = df_clean['Usage']  # Target variable

# Perform one-hot encoding on categorical features
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Random Forest Model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Save the trained model and encoder to files
joblib.dump(random_forest_model, 'random_forest_model.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Make predictions on the testing data
rf_predictions = random_forest_model.predict(X_test)

# Evaluate the Random Forest model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)  # RMSE
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
rf_mape = mean_absolute_percentage_error(y_test, rf_predictions)
rf_explained_variance = explained_variance_score(y_test, rf_predictions)

print("Random Forest Model Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rf_rmse}")
print(f"Mean Absolute Error (MAE): {rf_mae}")
print(f"R-squared (R²): {rf_r2}")
print(f"Mean Absolute Percentage Error (MAPE): {rf_mape}")
print(f"Explained Variance Score: {rf_explained_variance}")

# Neural Network Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=0)

# Save the trained model and scaler to files
model.save('neural_network_model.h5')
joblib.dump(scaler, 'scaler.pkl')

# Make predictions on the testing data
nn_predictions = model.predict(X_test_scaled).flatten()

# Evaluate the Neural Network model
nn_mse = mean_squared_error(y_test, nn_predictions)
nn_rmse = mean_squared_error(y_test, nn_predictions, squared=False)  # RMSE
nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_r2 = r2_score(y_test, nn_predictions)
nn_mape = mean_absolute_percentage_error(y_test, nn_predictions)
nn_explained_variance = explained_variance_score(y_test, nn_predictions)

print("\nNeural Network Model Evaluation:")
print(f"Root Mean Squared Error (RMSE): {nn_rmse}")
print(f"Mean Absolute Error (MAE): {nn_mae}")
print(f"R-squared (R²): {nn_r2}")
print(f"Mean Absolute Percentage Error (MAPE): {nn_mape}")
print(f"Explained Variance Score: {nn_explained_variance}")


# LSTM Model
# Reshape input to be 3D [samples, timesteps, features]
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# LSTM model architecture
lstm_model = Sequential([
    LSTM(50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_history = lstm_model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Save the trained model and scaler to files
lstm_model.save('lstm_model.h5')

# Make predictions on the testing data
lstm_predictions = lstm_model.predict(X_test_reshaped).flatten()

# Evaluate the LSTM model
lstm_mse = mean_squared_error(y_test, lstm_predictions)
lstm_rmse = mean_squared_error(y_test, lstm_predictions, squared=False)  # RMSE
lstm_mae = mean_absolute_error(y_test, lstm_predictions)
lstm_r2 = r2_score(y_test, lstm_predictions)
lstm_mape = mean_absolute_percentage_error(y_test, lstm_predictions)
lstm_explained_variance = explained_variance_score(y_test, lstm_predictions)

print("\nLSTM Model Evaluation:")
print(f"Root Mean Squared Error (RMSE): {lstm_rmse}")
print(f"Mean Absolute Error (MAE): {lstm_mae}")
print(f"R-squared (R²): {lstm_r2}")
print(f"Mean Absolute Percentage Error (MAPE): {lstm_mape}")
print(f"Explained Variance Score: {lstm_explained_variance}")



# Plot for Random Forest Model
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(rf_predictions, label='Predicted', color='red')
plt.title('Random Forest Model: Actual vs. Predicted Energy Consumption')
plt.xlabel('Index')
plt.ylabel('Usage')
plt.legend()
plt.show()

# Plot for Neural Network Model
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(nn_predictions, label='Predicted', color='red')
plt.title('Neural Network Model: Actual vs. Predicted Energy Consumption')
plt.xlabel('Index')
plt.ylabel('Usage')
plt.legend()
plt.show()

# Plot for LSTM Model
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(lstm_predictions, label='Predicted', color='red')
plt.title('LSTM Model: Actual vs. Predicted Energy Consumption')
plt.xlabel('Index')
plt.ylabel('Usage')
plt.legend()
plt.show()
