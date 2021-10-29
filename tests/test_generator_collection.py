from deepinterpolation.generic import JsonSaver, ClassLoader
import os
import pathlib


def test_generator_tif_creation(tmp_path):

    generator_param = {}

    # We are reusing the data generator for training here.
    generator_param["type"] = "generator"
    generator_param["name"] = "SingleTifGenerator"
    generator_param["pre_post_frame"] = 30
    generator_param["pre_post_omission"] = 0
    generator_param[
        "steps_per_epoch"
    ] = -1

    generator_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ophys_tiny_761605196.tif",
    )

    generator_param["batch_size"] = 5
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = 99
    generator_param[
        "randomize"
    ] = 0

    path_generator = os.path.join(tmp_path, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    assert len(data_generator) == 7


def test_generator_ephys_creation(tmp_path):
    generator_param = {}

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

    generator_param["batch_size"] = 10
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = -1
    generator_param[
        "randomize"
    ] = 0

    path_generator = os.path.join(tmp_path, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    assert len(data_generator) == 4996
