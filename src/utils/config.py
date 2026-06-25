import json

def get_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data