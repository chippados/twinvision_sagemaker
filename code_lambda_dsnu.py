import json
import boto3
import os
import pandas as pd
import numpy as np
import io

def create_dataset(X, y, time_steps=1):
    X_data, y_data = [], []
    for i in range(len(X) - time_steps):
        X_data.append(X[i:i+time_steps])
        y_data.append(y[i+time_steps])
    return np.array(X_data), np.array(y_data)

def lambda_handler(event, context):
    # Inicializa o cliente do SageMaker Runtime
    sagemaker_client = boto3.client('sagemaker-runtime')
    
    # Obtém o nome do endpoint do SageMaker a partir das variáveis de ambiente
    endpoint_name = os.environ['SAGEMAKER_ENDPOINT_NAME']
    
    try:
        # Extrai o CSV do evento (assumindo que é enviado como string base64 ou texto)
        # csv_data = event['body']
        # if event.get('isBase64Encoded', False):
        #     csv_data = base64.b64decode(csv_data).decode('utf-8')
        
        # Caminho do arquivo CSV local no ambiente Lambda (ex.: incluído no ZIP)
        csv_path = 'df_atuador1_dsnu_100ms.csv'  # Ajuste o nome do arquivo conforme necessário
        
        # Carrega o CSV em um DataFrame
        df = pd.read_csv(csv_path)
        df_filtered = df.reset_index(drop=True)

        # Prepara os dados
        data = df_anormal
        data['timestamp_unix'] = data['Avancado 1S2']
        X = data[['timestamp_unix']].values
        y = data['Avancado 1S2'].values
        
        # Normaliza os dados (mantendo a lógica original, sem escalonamento adicional)
        X_scaled = X
        y_scaled = y.reshape(-1, 1)
        
        # Cria janelas de tempo
        time_steps = 10
        X_data, y_data = create_dataset(X_scaled, y_scaled, time_steps)
        
        # Converte os dados para o formato JSON esperado pelo endpoint
        input_data = json.dumps(X_data.tolist())
        
        # Invoca o endpoint do SageMaker
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=input_data
        )
        
        # Obtém a resposta do endpoint
        result = json.loads(response['Body'].read().decode())
        
        # Retorna a resposta
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': result
            })
        }
    
    except Exception as e:
        # Retorna erro em caso de falha
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
