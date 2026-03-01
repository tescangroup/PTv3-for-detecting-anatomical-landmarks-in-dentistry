import os
import json

class TLSerializer(object):

    def __init__(self, json_path, only_with_landmarks=True, drop_unavailable_files=True):
        self.input_json_path = json_path
        self.root_path = os.path.dirname(json_path)

        self.body = list()
        self.header = dict()
        self.deserialize_json()

        if only_with_landmarks:
            self.drop_without_landmarks()

        if drop_unavailable_files:
            self.drop_unavailable_files()

    def drop_unavailable_files(self):
        for record in self.body:
            record["MeshPath"] = os.path.join(self.root_path, record["MeshPath"])
            if not os.path.exists(record["MeshPath"]):
                print(f'Dropping {record["MeshPath"]} due to unavailability.')
        self.body = [record for record in self.body if os.path.exists(record["MeshPath"])]

    def drop_without_landmarks(self):
        self.body = [record for record in self.body if record["Landmarks"]]

    def deserialize_json(self):
        loaded_from_file = self.get_json_from_file()
        if isinstance(loaded_from_file, list):
            loaded_from_file = loaded_from_file[0]
        self.header = loaded_from_file["Header"]
        if isinstance(self.header, list):
            self.header = self.header[0]
        self.body = loaded_from_file["Body"]

    def serialize_json(self, path_to_json, top_level_list=True):
        try:
            f = open(path_to_json, "w")
            to_write = {}
            to_write["Header"] = self.header
            to_write["Body"] = self.body
            if top_level_list:
                to_write = [to_write]
            f.write(json.dumps(to_write, indent=2))
            f.close()
        except Exception as e:
            print("Couldn't export json to file " + str(path_to_json) + " due to: " + str(e))

    def get_json_from_file(self):
        """
            Opens file specified by self.input_json_path and returns parsed json.
        """
        if not os.path.exists(self.input_json_path):
            print("Json file doesn't exist.")
            return None

        with open(self.input_json_path, 'r') as f:
            try:
                return json.loads(f.read())
            except json.decoder.JSONDecodeError:
                print(f'Error in json decoding.')
                exit(2)
