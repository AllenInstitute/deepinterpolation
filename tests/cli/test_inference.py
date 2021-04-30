import pytest
import h5py
import json
from pathlib import Path

import deepinterpolation.cli.inference as inf_cli


@pytest.fixture
def inference_args(tmpdir):
    # make some dummy files so the schema validation is satisfied
    model_path = tmpdir / "model.h5"
    output_path = tmpdir / "output.h5"
    with h5py.File(model_path, "w") as f:
        f.create_dataset("data", data=[1, 2, 3])
    args = {
            "model_path": str(model_path),
            "output_file": str(output_path)
            }
    yield args


@pytest.fixture
def generator_args(tmpdir):
    # make some dummy files so the schema validation is satisfied
    train_path = tmpdir / "train_data.h5"
    with h5py.File(train_path, "w") as f:
        f.create_dataset("data", data=[1, 2, 3])
    args = {
            "train_path": str(train_path)
            }
    yield args


class MockGenerator():
    """for these mocked tests, the generator needs
    no actual functionality
    """
    def __init__(self, arg):
        pass


class MockInference():
    """for mocked tests, inference only needs to produce a file
    """
    def __init__(self, inference_json_path, arg):
        self.inference_json_path = inference_json_path

    def run(self):
        with open(self.inference_json_path, "r") as f:
            j = json.load(f)
        with h5py.File(j["output_file"], "w") as f:
            f.create_dataset("data", data=[1, 2, 3])


class MockClassLoader():
    """mocks the behavior of the ClassLoader
    """
    def __init__(self, arg=None):
        pass

    @staticmethod
    def find_and_build():
        return MockClassLoader()

    def __call__(self, *args):
        # return something when called
        if len(args) == 1:
            return MockGenerator(args[0])
        if len(args) > 1:
            return MockInference(args[0], args[1])


def test_inference_cli(generator_args, inference_args, monkeypatch):
    """this tests that the inference CLI validates the schemas
    and executes its logic. Calls to generator and inference
    are minimally mocked.
    """
    args = {
            "inference_params": inference_args,
            "generator_params": generator_args,
            "output_full_args": True
            }
    monkeypatch.setattr(inf_cli, "ClassLoader", MockClassLoader)
    inference = inf_cli.Inference(input_data=args, args=[])
    inference.run()
    assert Path(args["inference_params"]["output_file"]).exists()
