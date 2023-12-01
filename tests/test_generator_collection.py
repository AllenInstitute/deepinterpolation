from deepinterpolation.generator_collection import SingleTifGenerator, EphysGenerator
import os
import pathlib


def test_generator_tif_creation():

    generator_param = {}

    # We are reusing the data generator for training here.
    generator_param["pre_frame"] = 30
    generator_param["post_frame"] = 30
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

    data_generator = SingleTifGenerator(generator_param)

    assert len(data_generator) == 8


def test_generator_ephys_creation():
    generator_param = {}

    generator_param["pre_frame"] = 30
    generator_param["post_frame"] = 30
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

    data_generator = EphysGenerator(generator_param)

    assert len(data_generator) == 4993
