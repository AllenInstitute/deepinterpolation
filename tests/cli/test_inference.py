import json
import os
from pathlib import Path

import h5py
import pytest

import deepinterpolation.cli.inference as inf_cli


@pytest.fixture
def inference_args(tmpdir, request):
    output_path = tmpdir / "output.h5"

    args = {"use_multiprocessing": False, "model_source": {}}
    if request.param.get("load_model_from_mlflow"):
        args["model_source"]["mlflow_registry"] = {
            "tracking_uri": "localhost",
            "model_name": "test",
        }
    else:
        # make some dummy files so the schema validation is satisfied
        model_path = tmpdir / "model.h5"

        with h5py.File(model_path, "w") as f:
            f.create_dataset("data", data=[1, 2, 3])
        args["model_source"]["local_path"] = str(model_path)

    args["output_file"] = str(output_path)

    yield args


@pytest.fixture
def generator_args(tmpdir):
    # make some dummy files so the schema validation is satisfied
    data_path = tmpdir / "train_data.h5"
    with h5py.File(data_path, "w") as f:
        f.create_dataset("data", data=[1, 2, 3])
    args = {"data_path": str(data_path)}
    yield args


class MockGenerator:
    """for these mocked tests, the generator needs
    no actual functionality
    """

    def __init__(self, arg):
        pass


class MockInference:
    """for mocked tests, inference only needs to produce a file"""

    def __init__(self, inference_json_path, arg):
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

    def __call__(self, *args):
        # return something when called
        if len(args) == 1:
            return MockGenerator(args[0])
        if len(args) > 1:
            return MockInference(args[0], args[1])


@pytest.mark.parametrize(
    "inference_args",
    [{"load_model_from_mlflow": True}, {"load_model_from_mlflow": False}],
    indirect=["inference_args"],
)
def test_inference_cli(generator_args, inference_args, monkeypatch):
    """this tests that the inference CLI validates the schemas
    and executes its logic. Calls to generator and inference
    are minimally mocked.
    """
    args = {
        "inference_params": inference_args,
        "generator_params": generator_args,
        "output_full_args": True,
    }
    monkeypatch.setattr(inf_cli, "ClassLoader", MockClassLoader)
    inference = inf_cli.Inference(input_data=args, args=[])
    inference.run()
    assert Path(args["inference_params"]["output_file"]).exists()


def test_integration_cli_ephys_inference_padding(tmp_path):

    generator_param = {}
    inferrence_param = {}

    generator_param["name"] = "EphysGenerator"
    generator_param["pre_frame"] = 30
    generator_param["post_frame"] = 30
    generator_param["pre_post_omission"] = 1

    generator_param["data_path"] = os.path.join(
        Path(__file__).parent.absolute(),
        "..",
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )

    generator_param["batch_size"] = 50
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = 100  # -1 to go until the end.

    inferrence_param["name"] = "core_inferrence"

    # Replace this path to where you stored your model
    filename = "2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5"
    local_path = os.path.join(
        Path(__file__).parent.absolute(), "..", "..", "sample_data", filename
    )
    inferrence_param["model_source"] = {"local_path": local_path}

    inferrence_param["output_padding"] = True
    inferrence_param["use_multiprocessing"] = False

    # Replace this path to where you want to store your output file
    inferrence_param["output_file"] = os.path.join(
        tmp_path, "ephys_tiny_continuous_deep_interpolation.h5"
    )

    args = {
        "inference_params": inferrence_param,
        "generator_params": generator_param,
        "output_full_args": True,
    }

    inference = inf_cli.Inference(input_data=args, args=[])
    inference.run()

    path_output = args["inference_params"]["output_file"]

    assert Path(path_output).exists()

    with h5py.File(path_output, "r") as h5_handle:
        nb_frame = h5_handle["data"].shape[0]

    assert nb_frame == (
        generator_param["end_frame"] - generator_param["start_frame"] + 1
    )


def test_integration_cli_ephys_inference_no_padding(tmp_path):

    generator_param = {}
    inferrence_param = {}

    generator_param["name"] = "EphysGenerator"
    generator_param["pre_frame"] = 30
    generator_param["post_frame"] = 30
    generator_param["pre_post_omission"] = 1

    generator_param["data_path"] = os.path.join(
        Path(__file__).parent.absolute(),
        "..",
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )

    generator_param["batch_size"] = 50
    generator_param["start_frame"] = 50
    generator_param["end_frame"] = 100  # -1 to go until the end.

    inferrence_param["name"] = "core_inferrence"

    # Replace this path to where you stored your model
    filename = "2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5"
    local_path = os.path.join(
        Path(__file__).parent.absolute(), "..", "..", "sample_data", filename
    )
    inferrence_param["model_source"] = {"local_path": local_path}

    inferrence_param["output_padding"] = False

    # Replace this path to where you want to store your output file
    inferrence_param["output_file"] = os.path.join(
        tmp_path, "ephys_tiny_continuous_deep_interpolation.h5"
    )

    args = {
        "inference_params": inferrence_param,
        "generator_params": generator_param,
        "output_full_args": True,
    }

    inference = inf_cli.Inference(input_data=args, args=[])
    inference.run()

    path_output = args["inference_params"]["output_file"]

    assert Path(path_output).exists()

    with h5py.File(path_output, "r") as h5_handle:
        nb_frame = h5_handle["data"].shape[0]

    assert nb_frame == 51
