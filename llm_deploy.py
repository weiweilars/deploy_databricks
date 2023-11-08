import os
import yaml
import logging

from huggingface_hub import snapshot_download

import mlflow
from mlflow.models import infer_signature
from mlflow.models.signature import infer_signature

from huggingface_hub._login import _login


def load_model(config_path):

    with open(config_path, "r") as yaml_file:
        config_data = yaml.safe_load(yaml_file)

    _login(token=config_data['Secrete']['huggingface_hub'], add_to_git_credential=False)
    model_path = os.path.join(config_data['Deployment']['model_download_path'],config_data['Name'])
    snapshot_location = os.path.expanduser(model_path)
    if ~model_path.isdir():
        print('downloading the model file')
        os.makedirs(snapshot_location, exist_ok=True)
        snapshot_location = snapshot_download(repo_id="HuggingFaceH4/zephyr-7b-alpha", local_dir=snapshot_location)
    

    

class Deploy_model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model snapshot directory.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


        # Initialize tokenizer and language model
        self.tokenizer = AutoTokenizer.from_pretrained(
            context.artifacts["snapshot"], padding_side="left"
        )

        # config = AutoConfig.from_pretrained(
        #     context.artifacts["snapshot"], trust_remote_code=True
        # )
        # If you are running this in a system that has a sufficiently powerful GPU with available VRAM,
        # uncomment the configuration setting below to leverage triton.
        # Note that triton dramatically improves the inference speed performance

        # config.attn_config["attn_impl"] = "triton"

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


# Get the current base version of torch that is installed, without specific version modifiers
# torch_version = torch.__version__.split("+")[0]

# Start an MLflow run context and log the MPT-7B model wrapper along with the param-included signature to
# allow for overriding parameters at inference time
'''
parameters = {"temperature":0.1,"max_tokens":150}

input_example = pd.DataFrame({"prompt": ["Hello, I'm a language model,"]})
output_example = pd.DataFrame({"predictions": ["How can I help you"]})
# inference_config={"max_new_tokens": 50, "temperature": 0.1}
signature = infer_signature(input_example, output_example, params=parameters)

# signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=parameters)


# Define input example
input_example = pd.DataFrame({"prompt": ["What is machine learning?"]})

mlflow.set_registry_uri('databricks-uc')
CATALOG ="capgemini"
SCHEMA = "chatbot"
registered_model_name = f"{CATALOG}.{SCHEMA}.llama_pyfunc_model"

with mlflow.start_run(run_name="llm_as_pyfunc") as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="llama_deployment_uc",
        python_model=Llama_pyfunc(),
        # NOTE: the artifacts dictionary mapping is critical! This dict is used by the load_context() method in our MPT() class.
        artifacts={"snapshot": snapshot_location},
        pip_requirements=['pandas==1.4.4',
    'torch==2.1.0',
    'transformers==4.34.0',
    'accelerate==0.23.0',
    'bitsandbytes==0.41.1',
    'tiktoken==0.5.1'],
        input_example=input_example,
        signature=signature,
    )

latest_model = mlflow.register_model(f'runs:/{run.info.run_id}/llama_deployment_uc', registered_model_name)
'''

if __name__ == "__main__":

    import argparse

    # 
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--config", type=str, help="Path to configration file")
    args = parser.parse_args()


    load_model(args.config)


