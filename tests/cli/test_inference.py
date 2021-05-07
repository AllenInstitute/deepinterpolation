import pytest
import h5py
import json
from pathlib import Path
import os

import deepinterpolation.cli.inference as inf_cli


@pytest.fixture
def inference_args(tmpdir, request):
    output_path = tmpdir / "output.h5"
    
    if request.param.get('load_model_from_mlflow'):
        mlflow_params = {
            'tracking_uri': 'localhost',
            'model_name': 'test'
        }
        args = {
            "mlflow_params": mlflow_params
        }
    else:
        # make some dummy files so the schema validation is satisfied
        model_path = tmpdir / "model.h5"

        with h5py.File(model_path, "w") as f:
            f.create_dataset("data", data=[1, 2, 3])
        args = {
            "model_path": str(model_path)
        }
    args["output_file"] = str(output_path)

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


@pytest.mark.parametrize('inference_args',
                         [{'load_model_from_mlflow': True},
                          {'load_model_from_mlflow': False}],
                         indirect=['inference_args'])
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


def test_integration_cli_ephys_inference(tmp_path):

    generator_param = {}
    inferrence_param = {}

    generator_param["type"] = "generator"
    generator_param["name"] = "EphysGenerator"
    generator_param["pre_post_frame"] = 30
    generator_param["pre_post_omission"] = 1
    generator_param[
        "steps_per_epoch"
    ] = -1

    generator_param["train_path"] = os.path.join(
        Path(__file__).parent.absolute(),
        "..",
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )

    generator_param["batch_size"] = 100
    generator_param["start_frame"] = 100
    generator_param["end_frame"] = 200  # -1 to go until the end.
    generator_param[
        "randomize"
    ] = 0

    inferrence_param["type"] = "inferrence"
    inferrence_param["name"] = "core_inferrence"

    # Replace this path to where you stored your model
    inferrence_param[
        "model_path"
    ] = os.path.join(
        Path(__file__).parent.absolute(),
        "..",
        "..",
        "sample_data",
        "2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5",
    )

    # Replace this path to where you want to store your output file
    inferrence_param[
        "output_file"
    ] = os.path.join(
        tmp_path,
        "ephys_tiny_continuous_deep_interpolation.h5"
    )

    args = {
        "inference_params": inferrence_param,
        "generator_params": generator_param,
        "output_full_args": True
    }

    inference = inf_cli.Inference(input_data=args, args=[])
    inference.run()
    assert Path(args["inference_params"]["output_file"]).exists()
