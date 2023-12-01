import os
import pathlib
import tempfile

import h5py

from deepinterpolation.inferrence_collection import core_inferrence
from deepinterpolation.generator_collection import EphysGenerator

def _get_generator_params():
    train_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )

    return {
        "pre_post_frame": 30,
        "pre_post_omission": 1,
        "steps_per_epoch": -1,
        "train_path": train_path,
        "batch_size": 10,
        "start_frame": 100,
        "end_frame": 200,
        "randomize": 0
    }


def _get_inference_params(output_path):
    model_name = "2020_02_29_15_28_unet_single_ephys_1024_" \
                 "mean_squared_error-1050"

    output_file = os.path.join(
        output_path,
        "ephys_tiny_continuous_deep_interpolation.h5"
    )

    params = {
        "type": "inferrence",
        "name": "core_inferrence",
        "model_source": {},
        "output_file": output_file
    }

    model_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        f"{model_name}.h5",
    )
    params['model_path'] = model_path

    return params


def _get_ephys_model(generator_params, inference_params):

    generator_obj = EphysGenerator(generator_params)
    model = core_inferrence(inference_params, generator_obj)
    return model


def test_ephys_inference():
    with tempfile.TemporaryDirectory() as jobdir:
        generator_params = _get_generator_params()
        inference_params = _get_inference_params(output_path=jobdir)
        ephys_model = _get_ephys_model(
                                       generator_params=generator_params,
                                       inference_params=inference_params)
        ephys_model.run()

        with h5py.File(inference_params["output_file"], 'r') as file_handle:
            local_size = file_handle['data'].shape

        # We check we get 100 frames out
        assert local_size[0] == 100