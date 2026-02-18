import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

data = pd.read_csv("traffic_data.csv")

values = data.values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(values)

X = []
y = []

time_steps = 3

for i in range(len(scaled_data) - time_steps):
    X.append(scaled_data[i:i + time_steps])
    y.append(scaled_data[i + time_steps])

X = np.array(X)
y = np.array(y)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_steps, 1)))
model.add(Dense(1))

model.compile(optimizer=Adam(), loss='mse')

model.fit(X, y, epochs=50, batch_size=1, verbose=1)

last_data = scaled_data[-time_steps:]
last_data = last_data.reshape(1, time_steps, 1)

prediction = model.predict(last_data)

predicted_traffic = scaler.inverse_transform(prediction)

print("Predicted Traffic Volume:", int(predicted_traffic[0][0]))