import pytest
import h5py
import json
from pathlib import Path
import os

import deepinterpolation.cli.training as cli


@pytest.fixture
def training_args(tmpdir, request):
    args = {}

    args["output_dir"] = str(tmpdir)
    args["model_string"] = "test_model_string"

    yield args


@pytest.fixture
def generator_args(tmpdir):
    # make some dummy files so the schema validation is satisfied
    data_path = tmpdir / "train_data.h5"
    with h5py.File(data_path, "w") as f:
        f.create_dataset("data", data=[1, 2, 3])
    args = {
        "data_path": str(data_path)
    }
    yield args


@pytest.fixture
def network_args():
    args = {}
    # make some dummy files so the schema validation is satisfied
    args["name"] = "unet_single_1024"

    yield args


class MockGenerator():
    """for these mocked tests, the generator needs
    no actual functionality
    """

    def __init__(self, arg):
        pass


class MockTraining():
    """for mocked tests, training only needs to produce a file
    """

    def __init__(self, data_generator, data_test_generator, data_network,
                 training_json_path):
        self.training_json_path = training_json_path

    def run(self):
        with open(self.training_json_path, "r") as f:
            j = json.load(f)

        local_model_path = os.path.join(j["output_dir"], j['run_uid']
                                        + "_" + j['model_string']
                                        + "_model.h5")

        with h5py.File(local_model_path, "w") as f:
            f.create_dataset("data", data=[1, 2, 3])

    def finalize(self):
        pass


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
            return MockTraining(args[0], args[1], args[2], args[3])


def test_training_cli(generator_args, training_args, network_args,
                      monkeypatch):
    """this tests that the training CLI validates the schemas
    and executes its logic. Calls to generator, network and training
    are minimally mocked.
    """
    args = {
        "run_uid": "test_uid",
        "training_params": training_args,
        "generator_params": generator_args,
        "test_generator_params": generator_args,
        "network_params": network_args,
        "output_full_args": True
    }
    monkeypatch.setattr(cli, "ClassLoader", MockClassLoader)
    training = cli.Training(input_data=args, args=[])
    training.run()

    model_path = os.path.join(args["training_params"]["output_dir"],
                              args["run_uid"] + "_" +
                              args["training_params"]["model_string"]
                              + "_model.h5")

    assert Path(model_path).exists()


def test_integration_cli_ephys_inference(tmp_path):

    generator_param = {}
    generator_test_param = {}

    training_param = {}
    network_param = {}

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

    generator_param["batch_size"] = 1
    generator_param["start_frame"] = 100
    generator_param["end_frame"] = 102  # -1 to go until the end.
    generator_param[
        "randomize"
    ] = 1

    generator_test_param["name"] = "EphysGenerator"
    generator_test_param["pre_frame"] = 30
    generator_test_param["post_frame"] = 30
    generator_test_param["pre_post_omission"] = 1

    generator_test_param["data_path"] = os.path.join(
        Path(__file__).parent.absolute(),
        "..",
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )

    generator_test_param["batch_size"] = 1
    generator_test_param["start_frame"] = 103
    generator_test_param["end_frame"] = 105  # -1 to go until the end.
    generator_test_param[
        "randomize"
    ] = True

    training_param["name"] = "core_trainer"
    training_param["model_string"] = "test_model_string"
    # Replace this path to where you want to store your output file
    training_param["output_dir"] = str(tmp_path)
    training_param["steps_per_epoch"] = 2
    network_param["name"] = "unet_single_ephys_1024"

    args = {
        "run_uid": "test_uid",
        "training_params": training_param,
        "generator_params": generator_param,
        "test_generator_params": generator_test_param,
        "network_params": network_param,
        "output_full_args": True
    }

    training = cli.Training(input_data=args, args=[])
    training.run()

    model_path = os.path.join(args["training_params"]["output_dir"],
                              args["run_uid"] + "_" +
                              training_param["model_string"]
                              + "_model.h5")

    assert Path(model_path).exists()
