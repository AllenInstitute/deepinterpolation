import json


class JsonLoader:
    """     
    JsonLoader is used to load the data from all structured json files associated with the DeepInterpolation package.
    """

    def __init__(self, path):
        self.path = path

        self.load_json()

    def load_json(self):
        """
        This function load the json file from the path recorded in the class instance. 

        Parameters:
        None

        Returns:
        None
        """

        with open(self.path, "r") as read_file:
            self.json_data = json.load(read_file)

    def set_default(self, parameter_name, default_value):
        """
        set default forces the initialization of a parameter if it was not present in
        the json file. If the parameter is already present in the json file, nothing
        will be changed.

        Parameters:
        parameter_name (str): name of the paramter to initialize
        default_value (Any): default parameter value

        Returns:
        None
        """

        if not (parameter_name in self.json_data):
            self.json_data[parameter_name] = default_value

    def get_type(self):
        """
        json types define the general category of the object the json file applies to.    
        For instance, the json can apply to a data Generator type

        Parameters: 
        None
    
        Returns: 
        str: Description of the json type 
        """

        return self.json_data["type"]

    def get_name(self):
        """     
        Each json type is sub-divided into different names. The name defines the exact construction logic of the object and how the
        parameters json data is used. For instance, a json file can apply to a Generator type using the AudioGenerator name when 
        generating data from an audio source. Type and Name fully defines the object logic. 

        Parameters: 
        None
    
        Returns: 
        str: Description of the json name 
        """

        return self.json_data["name"]


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
        json_class = JsonLoader(json_path)

        self.json_path = json_path
        self.local_type = json_class.get_type()
        self.local_name = json_class.get_name()

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
