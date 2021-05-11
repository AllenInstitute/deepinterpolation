import os
import pathlib
import tempfile

import h5py
import mlflow
import numpy as np
import pytest

from deepinterpolation.generic import JsonSaver, ClassLoader


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
        "batch_size": 100,
        "start_frame": 100,
        "end_frame": 200,
        "randomize": 0
    }


def _get_inference_params(output_path, mlflow_params=False,
                          use_legacy_model_path=False):
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

    if mlflow_params:
        params['model_source']['mlflow_registry'] = {
            'tracking_uri': f"sqlite:///{output_path}/mlruns.db",
            'model_name': model_name
        }
    else:
        model_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            "..",
            "sample_data",
            f"{model_name}.h5",
        )
        if use_legacy_model_path:
            params['model_path'] = model_path
        else:
            params['model_source']['local_path'] = model_path
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


@pytest.mark.parametrize('use_legacy_model_path', [True, False])
def test_ephys_inference(use_legacy_model_path):
    with tempfile.TemporaryDirectory() as jobdir:
        generator_params = _get_generator_params()
        inference_params = _get_inference_params(
            output_path=jobdir,
            use_legacy_model_path=use_legacy_model_path)
        ephys_model = _get_ephys_model(jobdir=jobdir,
                                       generator_params=generator_params,
                                       inference_params=inference_params)
        ephys_model.run()

        with h5py.File(inference_params["output_file"], 'r') as file_handle:
            local_size = file_handle['data'].shape

        # We check we get 100 frames out
        assert local_size[0] == 100


def test_mlflow_inference():
    """Tests that local model and model registered with mlflow provide same
    outputs"""

    def _get_local_model(jobdir):
        generator_params = _get_generator_params()
        inference_params = _get_inference_params(output_path=jobdir)
        ephys_model = _get_ephys_model(jobdir=jobdir,
                                       generator_params=generator_params,
                                       inference_params=inference_params)
        return ephys_model

    def _get_local_out():
        with tempfile.TemporaryDirectory() as jobdir:
            ephys_model = _get_local_model(jobdir=jobdir)
            ephys_model.run()

            output_file = ephys_model.json_data['output_file']
            with h5py.File(output_file, 'r') as file_handle:
                out_local = file_handle['data'][:]
                return ephys_model.model, out_local

    def _get_mlflow_out(local_model):
        def _register_model(model, tracking_uri, artifact_path, model_name):
            mlflow.set_tracking_uri(tracking_uri)

            experiment_id = \
                mlflow.create_experiment(artifact_location=artifact_path,
                                         name='test')
            with mlflow.start_run(experiment_id=experiment_id):
                mlflow.keras.log_model(model, "model",
                                       registered_model_name=model_name)

        with tempfile.TemporaryDirectory() as jobdir:
            generator_params = _get_generator_params()
            inference_params = _get_inference_params(output_path=jobdir,
                                                     mlflow_params=True)

            mlflow_registry_params = \
                inference_params['model_source']['mlflow_registry']
            tracking_uri = mlflow_registry_params['tracking_uri']
            model_name = mlflow_registry_params['model_name']
            _register_model(model=local_model, tracking_uri=tracking_uri,
                            model_name=model_name,
                            artifact_path=jobdir)

            ephys_model = _get_ephys_model(jobdir=jobdir,
                                           generator_params=generator_params,
                                           inference_params=inference_params)
            ephys_model.run()

            output_file = inference_params["output_file"]
            with h5py.File(output_file, 'r') as file_handle:
                out_mlflow = file_handle['data'][:]
                return out_mlflow

    local_model, out_local = _get_local_out()

    out_mlflow = _get_mlflow_out(local_model=local_model)

    assert np.allclose(out_local, out_mlflow)
