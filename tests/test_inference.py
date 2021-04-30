import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import pathlib
import h5py


def test_ephys_inference(tmp_path):

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
        pathlib.Path(__file__).parent.absolute(),
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
        pathlib.Path(__file__).parent.absolute(),
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

    jobdir = tmp_path

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_infer = os.path.join(jobdir, "inferrence.json")
    json_obj = JsonSaver(inferrence_param)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    inferrence_obj = ClassLoader(path_infer)
    inferrence_class = inferrence_obj.find_and_build()(path_infer,
                                                       data_generator)

    inferrence_class.run()

    with h5py.File(inferrence_param["output_file"], 'r') as file_handle:
        local_size = file_handle['data'].shape

    # We check we get 100 frames out

    assert local_size[0] == 100
