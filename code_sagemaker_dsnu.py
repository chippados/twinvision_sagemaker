import sagemaker
from sagemaker.tensorflow import TensorFlow

# Configurar a sessão
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Caminho para os dados (assumindo que já estão no S3)
#data_path = 's3://seu-bucket-s3/caminho-dos-seus-dados/' # Substitua pelo caminho do seu S3

#caminho local:
data_path = 'shared/df_atuador1_dsnu_100ms.csv' 

# Criar um estimador TensorFlow
estimator = TensorFlow(
    entry_point='train.py', 
    source_dir='.',         
    role=role,
    instance_count=1,
    instance_type='ml.t3.medium', 
    framework_version='2.10',     
    py_version='py39',            
    hyperparameters={
        'epochs': 100,
        'batch-size': 64,
        'learning-rate': 0.0001
    }
)

# Iniciar o trabalho de treinamento
estimator.fit({'training': data_path})

print("Treinamento finalizado!")