import sagemaker
from sagemaker.tensorflow import TensorFlow, TrainingCompilerConfig

# Configurar a sessão
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()



data_path = 's3://modelos-challenge/modelo_final_v2/dados/'  # Substitua pelo seu bucket S3


hyperparameters={
    "n_gpus": 1,
    "batch_size": 64,
    "learning_rate": 0.0001
}

# Criar um estimador TensorFlow
estimator = TensorFlow(
    entry_point='train_dsnu.py',
    source_dir='.',
    role=role,  # Adicionado o role aqui
    instance_count=1,
    instance_type='ml.t3.medium',
    framework_version='2.9.1',
    py_version='py39',
    hyperparameters=hyperparameters,
    compiler_config=TrainingCompilerConfig(),
    disable_profiler=True,
    debugger_hook_config=False
)

# Iniciar o trabalho de treinamento
estimator.fit({'training': data_path}, wait=True, logs='All')
print("Job de treinamento iniciado. Verifique o status no console do SageMaker.")
