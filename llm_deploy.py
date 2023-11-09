import os
import yaml
import pandas as pd


from huggingface_hub import snapshot_download
from huggingface_hub._login import _login

from transformers import AutoModelForCausalLM, AutoTokenizer

import mlflow
from mlflow.models import infer_signature
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from pyspark.dbutils import DBUtils
import SparkSession

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

def load_model(config_data):

    _login(token=config_data['Secrete']['huggingface_hub'], add_to_git_credential=False)
    model_path = os.path.join(config_data['Deployment']['model_download_path'],config_data['Model']['name'])
    snapshot_location = os.path.expanduser(model_path)
    os.makedirs(snapshot_location, exist_ok=True)
    snapshot_location = snapshot_download(repo_id="HuggingFaceH4/zephyr-7b-alpha", local_dir=snapshot_location)
    
    return snapshot_location

def choose_workload_size(quantization, parameter_size):
    workload_type = "GPU_MEDIUM"
    if quantization =="8_bit":
        if parameter_size < 18:
            workload_type = "GPU_MEDIUM" 
        elif parameter_size > 18 and parameter_size < 71:
            workload_type = "GPU_MEDIUM_4"
        elif parameter_size > 160:
            print("Model can't be fit on available workload sizes")
        else:
            workload_type = "GPU_MEDIUM_8"
    else:
        if parameter_size < 11:
            workload_type = "GPU_MEDIUM" 
        elif parameter_size > 10 and parameter_size < 35:
            workload_type = "GPU_MEDIUM_4"
        elif parameter_size > 80:
            print("Model can't be fit on available workload sizes, try 8_bit quantization")
        else:
            workload_type = "GPU_MEDIUM_8"
    return workload_type
    

class Deploy_model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model snapshot directory.
        """

        # Initialize tokenizer and language model
        self.tokenizer = AutoTokenizer.from_pretrained(
            context.artifacts["snapshot"], padding_side="left"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            context.artifacts["snapshot"],
            # torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            trust_remote_code=True,
            device_map="auto"
        )

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """

        return instruction


    def predict(self, context, model_input, params=None):
        """
        This method generates prediction for the given input.
        """
        prompt = model_input["prompt"][0]

        # Retrieve or use default values for temperature and max_tokens
        temperature = params["temperature"] if params else 0.1
        max_tokens = params["max_tokens" ]if params else 100

        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        # NB: Sending the tokenized inputs to the GPU here explicitly will not work if your system does not have CUDA support.
        # If attempting to run this with GPU support, change 'cpu' to 'cuda' for maximum performance
        encoded_input = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(
            encoded_input,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

        # Decode the prediction to text
        generated_response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        '''
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors="pt")[0])
        generated_response = self.tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        )
        '''

        return {"predictions": [generated_response]}
    

def main(config_path):

    with open(config_path, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)

    snapshot_location = load_model(config_data)

    ## Log the model using MLflow 
    parameters = config_data['Deployment']['parameters']
    input_example = pd.DataFrame({"prompt": config_data['Examples']['input_example']})
    output_example = pd.DataFrame({"predictions": config_data['Examples']['output_example']})
    signature = infer_signature(input_example, output_example, params=parameters)
    
    ### save the model in unity catalog
    ### NOTE: this catalog should be given perssion for the peoplewho run the project 
    mlflow.set_registry_uri(config_data['Deployment']['unity_catalog']['unity_catalog_uri'])
    catalog =config_data['Deployment']['unity_catalog']['unity_catalog_name']
    schema = config_data['Deployment']['unity_catalog']['unity_catalog_schema']
    model_name = config_data['Model']['Name'].split('/')[-1]
    registered_model_name = f"{catalog}.{schema}.{model_name}"

    with mlflow.start_run(run_name="llm_as_pyfunc") as run:
        ### log the model information 
        mlflow.log_param(config_data['Model'])
        model_info = mlflow.pyfunc.log_model(
            artifact_path="llama_deployment_uc",
            python_model=Deploy_model(),
            ### NOTE: the artifacts dictionary mapping is critical! This dict is used by the load_context() method in our MPT() class.
            artifacts={"snapshot": snapshot_location},
            pip_requirements=config_data['pip_reqirements'],
            input_example=input_example,
            signature=signature,
        )
    mlflow.register_model(f'runs:/{run.info.run_id}/llama_deployment_uc', registered_model_name)

    ## configure and create the model serving endpoint 
    endpoint_name = model_name
    model_name = registered_model_name
    model_version = MlflowClient().get_registered_model(model_name).latest_versions[0].version 
    ### choose the workload type according to quantization and parameter size 
    quantization = config_data['Deployment']['quantization']
    parameter_size = config_data['Deployment']['parameter_size']
    workload_type = choose_workload_size(quantization, parameter_size)
    workload_size = "Small" 
    scale_to_zero = False 
    API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
    API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()




if __name__ == "__main__":

    #import argparse
    #parser = argparse.ArgumentParser(description="Description of your script")
    #parser.add_argument("--config", type=str, help="Path to configration file")
    #args = parser.parse_args()

    config = "./configs/deployment_config.yaml"
    main(config)


