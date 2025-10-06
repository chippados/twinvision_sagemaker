import sagemaker
from sagemaker.tensorflow import TensorFlowModel

# Configurar a sessão
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

tf_model = TensorFlowModel(
                        model_data               = 's3://sagemaker-us-east-1-435133245123/tensorflow-training-2025-10-05-23-35-14-668/model/tensorflow_model/1/model.tar.gz',
                        role                     = role,
                        framework_version        ='2.11.0',
                        entry_point              = 'inference_script_dsnu.py',
                        sagemaker_session        = sagemaker_session
)

predictor = tf_model.deploy(
    initial_instance_count = 1,
    instance_type          = 'ml.m5.large'
)



# -------- TESTANDO O ENDPOINT

# Carregar os dados
df_union_1 = pd.read_csv('df_atuador1_dsnu_100ms.csv')
df_union_1_filtered = df_union_1.reset_index(drop=True)
df_normal = df_union_1_filtered[df_union_1_filtered['timestamp_horario_utc'] < '2025-09-05 00:15:50.767']

df_anormal = df_union_1_filtered[(df_union_1_filtered['timestamp_horario_utc'] > '2025-09-05 00:15:50.0') &
                                (df_union_1_filtered['timestamp_horario_utc'] < '2025-09-05 00:16:50.767')]

data = df_anormal

# Usar 'Avancado 1S2' diretamente como entrada e saída
data['timestamp_unix'] = data[['Avancado 1S2']]
X = data[['timestamp_unix']].values
y = data['Avancado 1S2'].values

# Normalizar os dados
X_scaled = X
y_scaled = y.reshape(-1, 1)

# Criar janelas de tempo
def create_dataset(X, y, time_steps=1):
    X_data, y_data = [], []
    for i in range(len(X) - time_steps):
        X_data.append(X[i:i+time_steps])
        y_data.append(y[i+time_steps])
    return np.array(X_data), np.array(y_data)

time_steps = 10
X_data, y_data = create_dataset(X_scaled, y_scaled, time_steps)


# Make predictions
y_pred = predictor.predict(X_data)  # X_data should be in the correct format
y_pred_rescaled = np.array(y_pred['predictions']).flatten()  # Extract numerical predictions

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Real')
plt.plot(y_pred_rescaled, label='Predito')
plt.xlabel('Time Steps')
plt.ylabel('Avancado 1S2')
plt.title('Real vs Predicted Values')
plt.legend()
plt.show()
