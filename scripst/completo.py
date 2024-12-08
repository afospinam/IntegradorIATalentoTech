import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D

# 1. Cargar los datos
data = pd.read_csv("../datasetoriginal/synthetic_energy_data.csv")

# 2. Preprocesamiento
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month

X = data[['Temperature_C', 'Day_of_Week', 'Hour_of_Day', 'Month']]
y = data['Consumption_kWh']

# Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ajustar la forma para el modelo CNN
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 3. Definir y entrenar el modelo
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train_cnn, y_train, epochs=5, batch_size=16, validation_data=(X_test_cnn, y_test))

# 4. Función de predicción (como se mostró anteriormente)
def predict_consumption(day, month, hour, temperature=None):
    if temperature is None:
        temperature = data['Temperature_C'].mean()
    date = datetime.datetime(2023, month, day)
    day_of_week = date.weekday()
    input_data = pd.DataFrame({
        'Temperature_C': [temperature],
        'Day_of_Week': [day_of_week],
        'Hour_of_Day': [hour],
        'Month': [month]
    })
    input_scaled = scaler.transform(input_data)
    input_cnn = input_scaled.reshape(1, input_scaled.shape[1], 1)
    prediction = model.predict(input_cnn)
    print(f"Predicción de consumo de energía para el {day}/{month} a las {hour}:00 horas:")
    print(f"{prediction[0][0]:.2f} kWh")
    return prediction[0][0]

# 5. Probar la función
predict_consumption(day=15, month=8, hour=14, temperature=25)
