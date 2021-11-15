import json


def save_as_json(obj, file_name):
    with open(file_name, "w") as f:
        json.dump(obj, f)
    return file_name


def load_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)
