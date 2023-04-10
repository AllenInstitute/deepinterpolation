import os
import pathlib
import tempfile
from typing import Tuple

import h5py
import numpy as np
import pytest

from deepinterpolation.generic import ClassLoader, JsonSaver
from deepinterpolation.inferrence_collection import core_inferrence


@pytest.fixture()
def ophys_movie(tmp_path: str):
    """yields a path to the movie fixture that is loaded
    into the generator object
    """
    rng = np.random.default_rng(1234)
    data = rng.random((71, 512, 512))
    outpath = os.path.join(tmp_path, "ophys_movie.h5")

    with h5py.File(outpath, "w") as f:
        f.create_dataset("data", data=data)
    yield outpath


def create_generator_json(
    tmp_path: str,
    ophys_movie: str,
) -> str:
    """Creates a json param file for an OphysGenerator object

    Returns
    ------
    str: full path to json
    """
    generator_params = {
        "type": "generator",
        "name": "InferenceOphysGenerator",
        "pre_frame": 30,
        "post_frame": 30,
        "pre_post_omission": 0,
        "steps_per_epoch": -1,
        "train_path": ophys_movie,
        "batch_size": 4,
        "start_frame": 0,
        "end_frame": 67,
        "randomize": 0,
    }
    path_generator = os.path.join(tmp_path, "generator.json")
    json_obj = JsonSaver(generator_params)
    json_obj.save_json(path_generator)
    return path_generator


def create_inference_json(
    tmp_path: str,
    output_path: str,
    save_raw: bool,
) -> Tuple[str, dict]:
    """Creates a params dict and json param file for
    a core_inference object

    Returns
    ------
    path_generator: str - full path to json
    inference_params: dict - dictionary containing core_inference params
    """
    model_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "unet_single_256-mean_absolute_error_model.h5",
    )
    output_file = os.path.join(
        output_path, "ophys_tiny_continuous_deep_interpolation.h5"
    )
    inference_params = {
        "type": "inferrence",
        "name": "core_inferrence",
        "steps_per_epoch": -1,
        "nb_workers": 4,
        "model_source": {"local_path": model_path},
        "rescale": True,
        "save_raw": save_raw,
        "output_file": output_file,
    }
    path_generator = os.path.join(tmp_path, "inference.json")
    json_obj = JsonSaver(inference_params)
    json_obj.save_json(path_generator)
    return path_generator, inference_params


def load_model(
    tmp_path: str, output_path: str, ophys_movie: str, save_raw: bool
) -> Tuple[core_inferrence, dict]:
    """Creates an inference_obj with associated parameters

    Returns
    ------
    core_inferrence: instantiation of a core_inferrence object
    inference_params: dict - dictionary containing core_inference params
    """
    generator_json_path = create_generator_json(tmp_path, ophys_movie)
    generator_obj = ClassLoader(generator_json_path)
    data_generator = generator_obj.find_and_build()(generator_json_path)
    inference_json_path, inference_params = create_inference_json(
        tmp_path,
        output_path,
        save_raw,
    )
    inferrence_obj = ClassLoader(inference_json_path)
    return (
        inferrence_obj.find_and_build()(inference_json_path, data_generator),
        inference_params,
    )


# def test__core_inference_runner__run_multiprocessing_equals_run(
#     tmp_path: str, ophys_movie: str
# ):
#     """Test core_inferrence runner and multiprocessing runner to
#     ensure they produce identical outputs
#     """
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     os.environ["OMP_NUM_THREADS"] = "1"
#     save_raw = True
#     with tempfile.TemporaryDirectory() as jobdir:
#         model, inference_params = load_model(tmp_path, jobdir, ophys_movie, save_raw)
#         model.run_multiprocessing()
#         expected_output_frames_count = model.generator_obj.list_samples.shape[0]
#         with h5py.File(inference_params["output_file"], "r") as file_handle:
#             multiprocessing_output = file_handle["data"][()]
#             assert multiprocessing_output.shape == (
#                 expected_output_frames_count,
#                 512,
#                 512,
#             )
#             if save_raw:
#                 raw_shape = file_handle["raw"].shape
#                 assert raw_shape == (expected_output_frames_count, 512, 512)
#             else:
#                 with pytest.raises(KeyError):
#                     file_handle["raw"]
#         model.run()
#         with h5py.File(inference_params["output_file"], "r") as file_handle:
#             output = file_handle["data"][()]
#             assert output.shape == (expected_output_frames_count, 512, 512)
#             if save_raw:
#                 raw_shape = file_handle["raw"].shape
#                 assert raw_shape == (expected_output_frames_count, 512, 512)
#             else:
#                 with pytest.raises(KeyError):
#                     file_handle["raw"]
#     np.testing.assert_almost_equal(output, multiprocessing_output)


def test__core_inference_runner__outputs_h5_with_correct_shape(
    tmp_path: str, ophys_movie: str
):
    """Test core_inferrence runner ensure they produce correct outputs"""
    save_raw = True
    with tempfile.TemporaryDirectory() as jobdir:
        model, inference_params = load_model(tmp_path, jobdir, ophys_movie, save_raw)
        expected_output_frames_count = model.generator_obj.list_samples.shape[0]
        model.run()
        with h5py.File(inference_params["output_file"], "r") as file_handle:
            output = file_handle["data"][()]
            assert output.shape == (expected_output_frames_count, 512, 512)
            if save_raw:
                raw_shape = file_handle["raw"].shape
                assert raw_shape == (expected_output_frames_count, 512, 512)
            else:
                with pytest.raises(KeyError):
                    file_handle["raw"]


def _get_generator_params():
    train_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )

    return {
        "type": "generator",
        "name": "EphysGenerator",
        "pre_post_frame": 30,
        "pre_post_omission": 1,
        "steps_per_epoch": -1,
        "train_path": train_path,
        "batch_size": 10,
        "start_frame": 100,
        "end_frame": 200,
        "randomize": 0,
    }


def _get_inference_params(
    output_path, mlflow_params=False, use_legacy_model_path=False
):
    model_name = "2020_02_29_15_28_unet_single_ephys_1024_" "mean_squared_error-1050"

    output_file = os.path.join(
        output_path, "ephys_tiny_continuous_deep_interpolation.h5"
    )

    params = {
        "type": "inferrence",
        "name": "core_inferrence",
        "model_source": {},
        "output_file": output_file,
    }

    if mlflow_params:
        params["model_source"]["mlflow_registry"] = {
            "tracking_uri": f"sqlite:///{output_path}/mlruns.db",
            "model_name": model_name,
        }
    else:
        model_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            "..",
            "sample_data",
            f"{model_name}.h5",
        )
        if use_legacy_model_path:
            params["model_path"] = model_path
        else:
            params["model_source"]["local_path"] = model_path
    return params


def _get_ephys_model(jobdir, generator_params, inference_params):
    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_params)
    json_obj.save_json(path_generator)

    path_infer = os.path.join(jobdir, "inferrence.json")
    json_obj = JsonSaver(inference_params)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    inferrence_obj = ClassLoader(path_infer)
    model = inferrence_obj.find_and_build()(path_infer, data_generator)
    return model


@pytest.mark.parametrize("use_legacy_model_path", [True, False])
def test_ephys_inference(use_legacy_model_path):
    with tempfile.TemporaryDirectory() as jobdir:
        generator_params = _get_generator_params()
        inference_params = _get_inference_params(
            output_path=jobdir, use_legacy_model_path=use_legacy_model_path
        )
        ephys_model = _get_ephys_model(
            jobdir=jobdir,
            generator_params=generator_params,
            inference_params=inference_params,
        )
        ephys_model.run()

        with h5py.File(inference_params["output_file"], "r") as file_handle:
            local_size = file_handle["data"].shape

        # We check we get 100 frames out
        assert local_size[0] == 101
