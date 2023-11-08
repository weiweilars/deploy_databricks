import yaml
import numpy as np
import pandas as pd

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking.client import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

config_path = dbutils.widgets.get("config_path")
with open(config_path) as yaml_file:
    content = yaml.safe_load(yaml_file)
    
model_name = content["model"][0]["model_name"]
tokenizer_name = content["model"][1]["tokenizer_name"]
task = content["model"][2]["task"]
parameter_size = content["model"][3]["parameter_size"]
quantizaion = content["model"][4]["quantizaion"]
catalog = content["model"][5]["catalog"]
schema = content["model"][6]["schema"]
context_length = content["model"][7]["context_length"]
pip_requirements =  content["model"][8]["pip_requirements"]

print("Building pipeline.........")
if quantizaion == "8_bit":
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                        device_map='auto',
                        load_in_8bit= True)
        
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                        device_map='auto',
                        torch_dtype=torch.float16)
    
tokenizer=AutoTokenizer.from_pretrained(tokenizer_name)

print("Pipeline Created.........")



input_example = pd.DataFrame({"prompt": ["Hello, I'm a language model,"]})
output_example = pd.DataFrame({"predictions": ["How can I help you"]})
inference_config={"max_new_tokens": 50, "temperature": 0.1}
signature = infer_signature(input_example,output_example, params=inference_config)


print("Logging model.........")
mlflow.set_registry_uri('databricks-uc')
CATALOG =catalog
SCHEMA = schema
registered_model_name = f"{CATALOG}.{SCHEMA}.{model_name.replace('/','-')}_llm_model"

with mlflow.start_run(run_name=f"{model_name.replace('/','-')}_run") as run:

    components = {
        "model": model,
        "tokenizer": tokenizer,
    }

    model_logged = mlflow.transformers.log_model(
        transformers_model=components,
        task = task,
        artifact_path="llama_deployment_uc",
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements
    )
print("Model Logged.........")


print("Registering Model.........")
latest_model = mlflow.register_model(f'runs:/{run.info.run_id}/llama_deployment_uc', registered_model_name)
print("Model Registered.........")


client = MlflowClient()
client.update_model_version(
name=latest_model.name,
version=latest_model.version,
description=f"""
Name : **{model_name}**   
task : **{task}**      
Size : **{parameter_size}**    
quantizaion : **{quantizaion}**    
context_length : **{context_length}**     
"""
)
print("Metadata Added.........")

endpoint_name = f"{model_name.replace('/','-')}_llm_endpoint"
logged_model_name = registered_model_name
 
model_version = latest_model.version

if quantizaion =="8_bit":
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


print(f'Selected workload type is {workload_type}.........')
 
workload_size = "Small" 
scale_to_zero = False 
 
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()


