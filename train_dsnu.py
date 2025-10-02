import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import os
import argparse
import tarfile
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    return parser.parse_args()

def train():
    args = parse_args()

    s3_path = os.path.join(args.data_dir, 'data.csv')  # Adjust filename as needed
    df_union_1 = pd.read_csv(s3_path)
    df_union_1_filtered = df_union_1.reset_index(drop=True)

    # Processar timestamp com data dinâmica
    df_union_1_filtered['timestamp_horario_utc'] = pd.to_datetime(df_union_1_filtered['timestamp_horario_utc'])
    cutoff_time = pd.to_datetime('2025-09-05 00:15:50.767')
    df_normal = df_union_1_filtered[df_union_1_filtered['timestamp_horario_utc'] < cutoff_time]
    data = df_normal
    X = data[['Avancado 1S2']].values
    y = data['Avancado 1S2'].values

    # Validar tamanho do conjunto de dados
    time_steps = 10
    if len(X) <= time_steps:
        raise ValueError("Conjunto de dados muito pequeno para o número de time_steps.")

    # Normalizar dados
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

    # Criar dataset
    X_data, y_data = create_dataset(X_scaled, y_scaled, time_steps)
    X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=False)

    # Criar modelo
    model = Sequential([
        Conv1D(filters=512, kernel_size=5, activation='relu', input_shape=(time_steps, 1)),
        Conv1D(filters=256, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    # Compilar modelo
    model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='mse', metrics=['mae'])

    # Treinar modelo
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    # Salvar o modelo
    model_dir = os.path.join(args.model_dir, '1')
    model.save(model_dir)

    # Criar arquivo tar.gz
    with tarfile.open(os.path.join(args.model_dir, 'model.tar.gz'), 'w:gz') as tar:
        tar.add(model_dir, arcname='1')

if __name__ == '__main__':
    train()
