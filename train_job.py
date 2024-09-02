import sagemaker
from sagemaker.sklearn import SKLearn

# Define your AWS IAM role
role = 'arn:aws:iam::730335322557:role/SagemakerRoleNet1'  # Replace with your IAM role

# Create a SageMaker session
sagemaker_session = sagemaker.Session()

# Define the S3 paths for storing code and model artifacts
code_path = 's3://datasetscybersec/model/'  # Replace with your S3 bucket name
output_path = 's3://datasetscybersec/model/'  # Replace with your S3 bucket name

# Set up the SKLearn estimator
# sklearn_estimator = SKLearn(
#     entry_point='train.py',
#     source_dir='.',
#     role=role,
#     instance_type='ml.m5.large',
#     instance_count=1,
#     framework_version='1.2-1',  # Use the appropriate sklearn version
#     py_version='py3',
#     sagemaker_session=sagemaker_session,
#     code_location=code_path,
#     output_path=output_path
# )

# 1. Set Instance Type
instance_type = 'ml.m5.xlarge'

# The 'instance_type' is configured to specify the type of SageMaker instance to be used for model training.
# In this case, 'ml.m5.xlarge' is chosen.

# 2. Configure the SKLearn Estimator
sklearn_estimator = SKLearn(entry_point='train.py',
                    framework_version="0.23-1",
                    py_version='py3',
                    instance_type=instance_type,
                    role=role,
                    output_path=output_path,
                    base_job_name='sklearn-iris',
                    hyperparameters={'n_estimators': 50, 'max_depth': 5})



# Start the training job with the input data
sklearn_estimator.fit()
