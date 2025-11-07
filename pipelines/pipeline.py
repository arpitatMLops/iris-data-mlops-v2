import os
import boto3
from sagemaker.session import Session
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker import get_execution_role
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep,TrainingStep
from sagemaker.workflow.parameters import ParameterString,ParameterInteger
from sagemaker.processing import ProcessingOutput,ScriptProcessor,ProcessingInput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
# from sagemaker.workflow.model_step import RegisterModel
from sagemaker.workflow.step_collections import RegisterModel

REGION = "ap-south-1"
boto_sess = boto3.Session(region_name=REGION)
sm_sess = Session(boto_sess)
pipe_sess = PipelineSession(boto_sess)
role = get_execution_role()
bucket = "sagemaker-pipelines-iris"
IMAGE_URI="718036509811.dkr.ecr.ap-south-1.amazonaws.com/iris-mlops-pipeline:latest"

output_prefix = ParameterString(
    name="OutputPrefix",
    default_value=f"s3://{bucket}/pipelines/artifacts/preprocess"
)

n_estimators_param = ParameterInteger(
    name="N_Estimators", default_value=20
)

preproc = ScriptProcessor(
    image_uri=IMAGE_URI,
    role=role,
    command=["python3"],
    # instance_type="ml.t3.medium",
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=pipe_sess,
)

trainer = Estimator(
    image_uri=IMAGE_URI,
    role=role,
    # instance_type="ml.t3.medium",
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=pipe_sess,
    entry_point="../model_training/sagemaker_train.py",
    source_dir=os.path.join(os.path.dirname(__file__), "../model_training"),
    hyperparameters={"n-estimators": n_estimators_param},
)

step_preprocess = ProcessingStep(
    name="PreprocessIris",
    processor=preproc,
    code=os.path.join(os.path.dirname(__file__), "../preprocessing/preprocessing.py"),
    job_arguments=["--output-dir", "/opt/ml/processing/output"],
    outputs=[
        ProcessingOutput(
            output_name="processed",
            source="/opt/ml/processing/output"
        )
    ],
)

step_train = TrainingStep(
    name="TrainModel",
    estimator=trainer,
    inputs={
        "train": TrainingInput(
            s3_data=step_preprocess.properties
                .ProcessingOutputConfig.Outputs["processed"]
                .S3Output.S3Uri
        )
    }
)

step_register = RegisterModel(
    name="RegisterIrisModel",
    estimator=trainer,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    image_uri=IMAGE_URI,
    content_types=["text/csv","application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="IrisModels",
)

pipeline = Pipeline(
    name="IrisPreprocessTrain",
    parameters=[output_prefix,n_estimators_param],
    steps=[step_preprocess,step_train,step_register],
    sagemaker_session=pipe_sess,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    params = {}
    if args.output_prefix:
        params["OutputPrefix"] = args.output_prefix

    pipeline.upsert(role_arn=role)
    execution = pipeline.start(parameters=params)
    print("Pipeline started. Execution ARN:", execution.arn)
