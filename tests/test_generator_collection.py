from deepinterpolation.generic import JsonSaver, ClassLoader
import deepinterpolation
import os
import pathlib
import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock

def test_generator_tif_creation(tmp_path):

    generator_param = {}

    # We are reusing the data generator for training here.
    generator_param["type"] = "generator"
    generator_param["name"] = "SingleTifGenerator"
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

    path_generator = os.path.join(tmp_path, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    assert len(data_generator) == 8


def test_generator_ephys_creation(tmp_path):
    generator_param = {}

    generator_param["type"] = "generator"
    generator_param["name"] = "EphysGenerator"
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

    path_generator = os.path.join(tmp_path, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    assert len(data_generator) == 4993


class TestOphysGenerator:
    def create_json(self,
                    tmp_path,
                    gpu_cache_full=False,
                    normalize_cache=True,
                    ):
        generator_param = {}

        generator_param["type"] = "generator"
        generator_param["name"] = "OphysGenerator"
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
            "suite2p_motion_output_mini.h5",
        )

        generator_param["batch_size"] = 4
        generator_param["start_frame"] = 0
        generator_param["end_frame"] = -1
        generator_param[
            "randomize"
        ] = 0
        generator_param["gpu_cache_full"] = gpu_cache_full
        generator_param["normalize_cache"] = normalize_cache
        generator_param["use_mixed_float_16"] = True

        path_generator = os.path.join(tmp_path, "generator.json")
        json_obj = JsonSaver(generator_param)
        json_obj.save_json(path_generator)
        return path_generator

    def test__find_and_build__creates_generator_with_correct_number_of_batches(self, tmp_path):
        path_generator = self.create_json(tmp_path)
        generator_obj = ClassLoader(path_generator)
        data_generator = generator_obj.find_and_build()(path_generator)
        assert len(data_generator) == 9

    @pytest.mark.parametrize("gpu_cache_full", [True, False])
    @pytest.mark.parametrize("normalize_cache", [True, False])
    def test__movie_data__returns_movie(self, 
                                        tmp_path,
                                        gpu_cache_full,
                                        normalize_cache,
                                        ):
        path_generator = self.create_json(
            tmp_path, gpu_cache_full, normalize_cache)
        generator_obj = ClassLoader(path_generator)
        data_generator = generator_obj.find_and_build()(path_generator)
        data_generator._normalize = MagicMock()
        data_generator._gpu_available = MagicMock()
        data_generator._gpu_available.return_value = True
        movie_data = data_generator.movie_data
        assert movie_data.shape == (100, 512, 512)
        if gpu_cache_full:
            assert isinstance(movie_data, tf.Tensor)
            assert movie_data.dtype=='float32'
        else: 
            assert isinstance(movie_data, np.ndarray)
            if normalize_cache:
                assert movie_data.dtype=='float32'
            else:
                assert movie_data.dtype=='int16'
        if normalize_cache:
            assert data_generator._normalize.assert_called_once
            

