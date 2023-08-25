import json
import os

import h5py


class _MockGenerator:
    """for these mocked tests, the generator needs
    no actual functionality
    """

    def __init__(self, arg):
        pass


class _MockTraining:
    """for mocked tests, training only needs to produce a file"""

    def __init__(
        self,
        trainer_json_path,
        network_obj=None
    ):
        self.training_json_path = trainer_json_path

    def run(self, train_generator, test_generator):
        with open(self.training_json_path, "r") as f:
            j = json.load(f)

        local_model_path = os.path.join(
            j["output_dir"], j["run_uid"] + "_" + j["model_string"] +
            f"{'_transfer' if j['name'] == 'transfer_trainer' else ''}_model.h5"
        )

        with h5py.File(local_model_path, "w") as f:
            f.create_dataset("data", data=[1, 2, 3])

    def finalize(self):
        pass


class _MockInference:
    """for mocked tests, inference only needs to produce a file"""

    def __init__(self, inference_json_path, data_generator=None):
        self.inference_json_path = inference_json_path

    def run(self):
        with open(self.inference_json_path, "r") as f:
            j = json.load(f)
        with h5py.File(j["output_file"], "w") as f:
            f.create_dataset("data", data=[1, 2, 3])


class MockClassLoader:
    """mocks the behavior of the ClassLoader"""

    def __init__(self, arg=None):
        pass

    @staticmethod
    def find_and_build():
        return MockClassLoader()

    def __call__(self, **kwargs):
        if 'trainer_json_path' in kwargs:
            json_path = kwargs['trainer_json_path']
        elif 'inferrence_json_path' in kwargs:
            json_path = kwargs['inferrence_json_path']
        elif 'json_path' in kwargs:
            json_path = kwargs['json_path']
        elif 'path_json' in kwargs:
            json_path = kwargs['path_json']
        else:
            raise ValueError(f'Could not find json path in {kwargs}')
        with open(json_path) as f:
            d = json.load(f)

        # return something when called
        if d['type'] == 'generator':
            return _MockGenerator(json_path)
        elif d['type'] == 'trainer':
            return _MockTraining(json_path)
        elif d['type'] == 'inferrence':
            return _MockInference(json_path)
        else:
            return None
