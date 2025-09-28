import sagemaker
from sagemaker.tensorflow import TensorFlow

# Configurar a sessão
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Caminho para os dados no S3
data_path = 's3://seu-bucket/caminho-para-dados/'  # Substitua pelo seu bucket S3

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