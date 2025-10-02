import sagemaker
from sagemaker.tensorflow import TensorFlow, TrainingCompilerConfig

# Configurar a sessão
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

tensorflow_estimator = TensorFlow(
                        entry_point              = 'train_dsnu.py',
                        source_dir               = './',
                        role                     = role,
                        instance_count           = 1,
                        instance_type            = 'ml.p2.xlarge',
                        framework_version        ='2.19.0',
                        outputh_path             = 's3://modelos-challenge/modelo_final_v2/modelo_sage/',
                        py_version               = 'py39',
                        hyperparameters          = {'epochs'       : 100,
                                                    'batch_size'   : 64,
                                                    'learning_rate': 0.0001},
                        enable_sagemaker_metrics = True
)

tensorflow_estimator.fit("s3://modelos-challenge/modelo_final_v2/dados/")
