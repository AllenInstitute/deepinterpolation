import json

class JsonSaver:
    """     
    JsonSaver is used to save dict data into individual file.
    """

    def __init__(self, dict_save):
        self.dict = dict_save

    def save_json(self, path):
        """ 
        This function save the json file into the path provided. 

        Parameters: 
        str: path: str

        Returns: 
        None
        """

        with open(path, "w") as write_file:
            json.dump(self.dict, write_file)


class ClassLoader:
    """     
    ClassLoader allows to select and create a specific Type and Name object from the available library of objects. It then
    uses the parameters in the json file to create a specific instance of that object. 
    It returns that object and the ClassLoader object should then be deleted. 
    """

    from deepinterpolation import network_collection
    from deepinterpolation import generator_collection
    from deepinterpolation import trainor_collection
    from deepinterpolation import inferrence_collection

    def __init__(self, json_path):
        with open(json_path, "r") as read_file:
            json_data = json.load(read_file)

        self.json_path = json_path
        self.local_type = json_data["type"] 
        self.local_name = json_data["name"]

    def find_and_build(self):
        """
        This function searches the available classes available for object 'type' and 'name' and returns a callback to instantiate.

        Parameters:
        None

        Returns: 
        obj: an instantiation callback of the object requested when creating ClassLoader with a json file
        """

        if self.local_type == "network":
            local_object = getattr(self.network_collection, self.local_name)
            return local_object
        elif self.local_type == "generator":
            local_object = getattr(self.generator_collection, self.local_name)
            return local_object
        elif self.local_type == "trainer":
            local_object = getattr(self.trainor_collection, self.local_name)
            return local_object
        elif self.local_type == "inferrence":
            local_object = getattr(self.inferrence_collection, self.local_name)
            return local_object
