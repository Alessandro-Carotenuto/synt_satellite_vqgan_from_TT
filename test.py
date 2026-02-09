import yaml
import os

path=os.getcwd()+'/model.yaml'
with open (path, 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

#current path is obtainable via 
print(config['model']['base_learning_rate'])