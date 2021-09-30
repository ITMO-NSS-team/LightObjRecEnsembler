import json


class File(object):
    """[summary]

    Args:
        object ([type]): [description]
    """
    def __init__(self, filepath: str) -> None:
        if not filepath.endswith('.json'):
            raise ValueError("filepath must be an .json file:", filepath)

        self.filepath = filepath

    def read(self) -> dict:
        with open(self.filepath, "r") as f:
            data = json.load(f)

        return data
