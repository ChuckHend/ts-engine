import json


# reads the config file and returns the values for keys passed in
def load_config(keys):
	with open ('./config/config.json', 'r') as f:
		config = json.load(f)
	return config[keys]