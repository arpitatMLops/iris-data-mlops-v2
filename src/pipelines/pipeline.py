import os
import pathlib
import boto3

from sagemaker import get_execution_role
from sagemaker.session import Session
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.processing import ProcessingOutput, Processor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.step_collections import RegisterModel

REGION = os.getenv("AWS_REGION", "ap-south-1")
PROJECT_NAME = os.getenv("PROJECT_NAME", "iris-mlops")
ENV = os.getenv("ENV", "dev")

IMAGE_URI = os.getenv(
    "IMAGE_URI",
    "718036509811.dkr.ecr.ap-south-1.amazonaws.com/iris-mlops-dev:latest",
)

ARTIFACTS_BUCKET = os.getenv("ARTIFACTS_BUCKET", "sagemaker-pipelines-iris")
ROLE_ARN = os.getenv("SM_EXEC_ROLE_ARN")

boto_sess = boto3.Session(region_name=REGION)
sm_sess = Session(boto_sess)            
pipe_sess = PipelineSession(boto_sess) 

role = ROLE_ARN or get_execution_role()

# -----------------------------
# Pipeline parameters
# -----------------------------
output_prefix = ParameterString(
    name="OutputPrefix",
    default_value=f"s3://{ARTIFACTS_BUCKET}/pipelines/{ENV}/artifacts/preprocess",
)

n_estimators_param = ParameterInteger(name="N_Estimators", default_value=20)

# Processing
preproc = Processor(
    image_uri=IMAGE_URI,
    role=role,
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    sagemaker_session=pipe_sess,
)

step_preprocess = ProcessingStep(
    name="PreprocessIris",
    processor=preproc,
    job_arguments=[
        "/opt/ml/src/preprocessing/preprocessing.py",
        "--output-dir", "/opt/ml/processing/output",
    ],
    outputs=[ProcessingOutput(output_name="processed", source="/opt/ml/processing/output")],
)

trainer = Estimator(
    image_uri=IMAGE_URI,
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    sagemaker_session=pipe_sess,
    hyperparameters={"n-estimators": n_estimators_param},
    environment={
        "SAGEMAKER_PROGRAM": "/opt/ml/src/model_training/sagemaker_train.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/src",
    },
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
    },
)

step_register = RegisterModel(
    name="RegisterIrisModel",
    estimator=trainer,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    image_uri=IMAGE_URI,
    content_types=["text/csv", "application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="IrisModels",
)


pipeline = Pipeline(
    name=f"{PROJECT_NAME}-{ENV}", 
    parameters=[output_prefix, n_estimators_param],
    steps=[step_preprocess, step_train, step_register],
    sagemaker_session=pipe_sess,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", help="Override OutputPrefix parameter")
    args = parser.parse_args()

    if args.output_prefix:
        output_prefix.default_value = args.output_prefix

    out_dir = pathlib.Path("pipelines") / ENV
    out_dir.mkdir(parents=True, exist_ok=True)
    
    definition_body = pipeline.definition()

    out_path = out_dir / "pipeline.json"
    with open(out_path, "w") as f:
        f.write(definition_body)

    print(f"Wrote pipeline definition to {out_path}")

