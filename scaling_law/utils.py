import json


def load_data_info(json_file):
    with open(json_file, 'r') as file:
        data_info = json.load(file)
    return data_info


