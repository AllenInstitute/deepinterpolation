from deepinterpolation.generic import JsonSaver, ClassLoader
import h5py
import os
import pathlib
import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock, patch

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
    
    @pytest.fixture()
    def ophys_movie(self, tmp_path):
        data = np.arange(80, dtype='int16').reshape(20, 2, 2)
        outpath = os.path.join(tmp_path, "ophys_movie.h5")

        with h5py.File(outpath, "w") as f:
            f.create_dataset("data", data=data)
        yield outpath

    def create_json(self,
                    tmp_path,
                    gpu_cache_full,
                    normalize_cache,
                    ophys_movie,
                    ):
        generator_param = {}

        generator_param["type"] = "generator"
        generator_param["name"] = "OphysGenerator"
        generator_param["pre_frame"] = 2
        generator_param["post_frame"] = 2
        generator_param["pre_post_omission"] = 1
        generator_param[
            "steps_per_epoch"
        ] = -1

        generator_param["train_path"] = ophys_movie

        generator_param["batch_size"] = 3
        generator_param["start_frame"] = 0
        generator_param["end_frame"] = -1
        generator_param[
            "randomize"
        ] = 0
        generator_param["gpu_cache_full"] = gpu_cache_full
        generator_param["normalize_cache"] = normalize_cache

        path_generator = os.path.join(tmp_path, "generator.json")
        json_obj = JsonSaver(generator_param)
        json_obj.save_json(path_generator)
        return path_generator


    @pytest.mark.parametrize("gpu_cache_full", [True, False])
    @pytest.mark.parametrize("normalize_cache", [True, False])
    def test__find_and_build__gpu_available_creates_generator_with_correct_batches( # noqa
        self, tmp_path, gpu_cache_full, normalize_cache, ophys_movie):
        path_generator = self.create_json(
            tmp_path, gpu_cache_full, normalize_cache, ophys_movie)
        generator_obj = ClassLoader(path_generator)
        with patch("tensorflow.test.is_gpu_available") as mock_is_available:
            mock_is_available.return_value = True
            data_generator = generator_obj.find_and_build()(path_generator)
        data = np.arange(80, dtype='float32').reshape(20,2,2)
        data = (data - data.mean()) / data.std()
        batch_size = 3
        nb_datasets = len(data_generator)
        test_batch_indices = [0, 3, nb_datasets-1]
        for i in test_batch_indices:
            expected_batch_indices = np.vstack([[0, 1, 5, 6],
                                                [1, 2, 6, 7],
                                                [2, 3, 7, 8]])
            expected_batch_indices += i*batch_size
            expected_batch = data[expected_batch_indices]
            expected_batch = np.moveaxis(expected_batch, 1, -1)
            expected_batch = np.expand_dims(expected_batch, -1)
            obtained_batch = data_generator[i][0].numpy()
            np.testing.assert_array_equal(obtained_batch, expected_batch)
            assert obtained_batch.dtype == 'float32'
        

    @pytest.mark.parametrize("gpu_cache_full", [True, False])
    @pytest.mark.parametrize("normalize_cache", [True, False])
    def test__movie_data__returns_movie(self, 
                                        tmp_path,
                                        gpu_cache_full,
                                        normalize_cache,
                                        ophys_movie,
                                        ):
        path_generator = self.create_json(
            tmp_path, gpu_cache_full, normalize_cache, ophys_movie)
        generator_obj = ClassLoader(path_generator)
        data_generator = generator_obj.find_and_build()(path_generator)
        data_generator._normalize = MagicMock()

        # Setting this to be true for testing compatibility with non-gpu sys
        # This produces the expected behavior since tf.Tensor will use the
        # CPU instead
        data_generator._gpu_available = MagicMock()
        data_generator._gpu_available.return_value = True

        movie_data = data_generator.movie_data
        assert movie_data.shape == (20, 2, 2)
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
