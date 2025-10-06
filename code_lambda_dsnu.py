import json
import boto3
import pandas as pd
import numpy as np
import logging
import os

# Configure logging for CloudWatch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def create_dataset(X, y, time_steps=1):
    X_data, y_data = [], []
    for i in range(len(X) - time_steps):
        X_data.append(X[i:i+time_steps])
        y_data.append(y[i+time_steps])
    return np.array(X_data), np.array(y_data)

def lambda_handler(event, context):
    # Initialize clients
    s3_client = boto3.client('s3')
    sagemaker_client = boto3.client('sagemaker-runtime')
    
    # Get environment variables
    endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT_NAME')
    bucket = event.get('bucket', 'modelos-challenge')
    key = event.get('key', 'modelo_final_v2/dados/df_atuador1_dsnu_100ms.csv')
    
    if not endpoint_name:
        logger.error("SAGEMAKER_ENDPOINT_NAME not set")
        return {'statusCode': 500, 'body': json.dumps({'error': 'SAGEMAKER_ENDPOINT_NAME not set'})}
    
    try:
        # Get timestamp range from event
        start_time = event.get('start_time', '2025-09-05 00:15:50.0')
        end_time = event.get('end_time', '2025-09-05 00:16:50.767')
        
        # Download CSV from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(response['Body'])
        df_filtered = df.reset_index(drop=True)
        
        # Validate required columns
        required_columns = ['timestamp_horario_utc', 'Avancado 1S2']
        if not all(col in df_filtered.columns for col in required_columns):
            logger.error(f"Missing required columns: {required_columns}")
            return {'statusCode': 400, 'body': json.dumps({'error': f'Missing required columns: {required_columns}'})}
        
        # Filter data by timestamp
        df_anormal = df_filtered[(df_filtered['timestamp_horario_utc'] > start_time) &
                                 (df_filtered['timestamp_horario_utc'] < end_time)]
        
        if df_anormal.empty:
            logger.error("No data after timestamp filtering")
            return {'statusCode': 400, 'body': json.dumps({'error': 'No data after timestamp filtering'})}
        
        # Prepare data
        data = df_anormal.copy()
        data['timestamp_unix'] = pd.to_datetime(data['timestamp_horario_utc']).astype(np.int64) // 10**9
        X = data[['timestamp_unix']].values
        y = data['Avancado 1S2'].values
        
        # Normalize data
        X_scaled = X
        y_scaled = y.reshape(-1, 1)
        
        # Create time windows
        time_steps = 10
        X_data, y_data = create_dataset(X_scaled, y_scaled, time_steps)
        
        if X_data.size == 0:
            logger.error("No data after creating time windows")
            return {'statusCode': 400, 'body': json.dumps({'error': 'No data after creating time windows'})}
        
        # Convert to JSON for SageMaker
        input_data = json.dumps(X_data.tolist())
        
        # Invoke SageMaker endpoint
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=input_data
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': result})
        }
    
    except Exception as e:
        logger.exception(f"Error processing request: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({'error': f'Failed to process request: {str(e)}'})}
