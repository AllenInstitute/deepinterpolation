import json
import os
from pathlib import Path

import h5py
import pytest

import deepinterpolation.cli.fine_tuning as cli
from deepinterpolation.testing.utils import MockClassLoader


@pytest.fixture
def training_args(tmpdir, request):
    args = {}

    filename = "2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5"
    local_path = os.path.join(
        Path(__file__).parent.absolute(), "..", "..", "sample_data", filename
    )
    args["output_dir"] = str(tmpdir)
    args["model_string"] = "test_model_string"
    args["model_source"] = {"local_path": local_path}

    yield args


@pytest.mark.parametrize('load_movie_into_memory', (True, False))
def test_finetuning_cli(
    load_movie_into_memory,
    movie_json_generator_args,
    training_args,
    monkeypatch
):
    """this tests that the training CLI validates the schemas
    and executes its logic. Calls to generator, network and training
    are minimally mocked.
    """
    training_args['load_movie_into_memory'] = load_movie_into_memory
    args = {
        "run_uid": "test_uid",
        "finetuning_params": training_args,
        "generator_params": movie_json_generator_args,
        "test_generator_params": movie_json_generator_args,
        "output_full_args": True,
    }

    monkeypatch.setattr(cli, "ClassLoader", MockClassLoader)
    training = cli.FineTuning(input_data=args, args=[])
    training.run()

    model_path = os.path.join(
        args["finetuning_params"]["output_dir"],
        args["run_uid"]
        + "_"
        + args["finetuning_params"]["model_string"]
        + "_transfer_model.h5",
    )

    assert Path(model_path).exists()
