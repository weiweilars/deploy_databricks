import yaml

# Load the YAML config file
with open("config.yaml", "r") as yaml_file:
    config_data = yaml.safe_load(yaml_file)