import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import os
import argparse

def train_model(args):
    df_union_1 = pd.read_csv(os.path.join(args.data_dir, 'df_atuador1_dsnu_100ms.csv'))
    df_union_1_filtered = df_union_1.reset_index(drop=True)
    
    df_union_1_filtered['timestamp_horario_utc'] = pd.to_datetime(df_union_1_filtered['timestamp_horario_utc'])
    df_normal = df_union_1_filtered[df_union_1_filtered['timestamp_horario_utc'] < '2025-09-05 00:15:50.767']
    
    data = df_normal
    X = data[['Avancado 1S2']].values
    y = data['Avancado 1S2'].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    def create_dataset(X, y, time_steps=1):
        X_data, y_data = [], []
        for i in range(len(X) - time_steps):
            X_data.append(X[i:i+time_steps])
            y_data.append(y[i+time_steps])
        return np.array(X_data), np.array(y_data)
    
    time_steps = 10
    X_data, y_data = create_dataset(X_scaled, y_scaled, time_steps)
    X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=False)

    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=5, activation='relu', input_shape=(time_steps, 1)))
    model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_test, y_test))
    
    model.save(os.path.join(args.model_dir, 'modelo_cnn_regression.h5'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()
    train_model(args)