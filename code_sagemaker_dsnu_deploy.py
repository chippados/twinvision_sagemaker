import sagemaker
from sagemaker.tensorflow import TensorFlowModel

# Configurar a sessão
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

tf_model = TensorFlowModel(
                        model_data               = 's3://sagemaker-us-east-1-435133245123/tensorflow-training-2025-10-05-22-41-18-493/source/sourcedir.tar.gz',
                        role                     = role,
                        framework_version        ='2.11.0',
                        entry_point              = 'inference_script_dsnu.py',
                        sagemaker_session        = sagemaker_session
)

predictor = tf_model.deploy(
    initial_instance_count = 1,
    instance_type          = 'ml.m5.large'
)
