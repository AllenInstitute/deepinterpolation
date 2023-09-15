import glob
import json
import logging
import math
import os
from pathlib import Path
from typing import Tuple, Union, Optional, Dict

import h5py
import nibabel as nib
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tifffile

from deepinterpolation.generic import JsonLoader

logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


class MaxRetryException(Exception):
    # This is helper class for EmGenerator
    pass


class DeepGenerator(keras.utils.Sequence):
    """
    This class instantiante the basic Generator Sequence object
    from which all Deep Interpolation generator should be generated.

    Parameters:
    json_path: a path to the json file used to parametrize the generator

    Returns:
    None
    """

    def __init__(self, json_path):
        local_json_loader = JsonLoader(json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data
        self.local_mean = 1
        self.local_std = 1

    def get_input_size(self):
        """
        This function returns the input size of the
        generator, excluding the batching dimension

        Parameters:
        None

        Returns:
        tuple: list of integer size of input array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[0]

        return local_obj.shape[1:]

    def get_output_size(self):
        """
        This function returns the output size of
        the generator, excluding the batching dimension

        Parameters:
        None

        Returns:
        tuple: list of integer size of output array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[1]

        return local_obj.shape[1:]

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return [np.array([]), np.array([])]

    def __get_norm_parameters__(self, idx):
        """
        This function returns the normalization parameters
        of the generator. This can potentially be different
        for each data sample

        Parameters:
        idx index of the sample

        Returns:
        local_mean
        local_std
        """
        local_mean = self.local_mean
        local_std = self.local_std

        return local_mean, local_std

    def _normalize(
        self, arr: Union[np.ndarray, tf.Tensor], mean: float, std: float
    ) -> Union[np.ndarray, tf.Tensor]:
        """Normalize input

        Parameters:
        arr: Union[np.ndarray, tf.Tensor]
            Input array to be normalized
        mean: float
        std: float

        Returns:
        normalized_arr: Union[np.ndarray, tf.Tensor]
        """
        return (arr - mean) / std


class CollectorGenerator(DeepGenerator):
    """This class allows to create a generator of generators
    for the purpose of training across multiple files
    All generators must have idendical batch size and input,
    output size but can be different length
    """

    def __init__(self, generator_list):
        self.generator_list = generator_list
        self.nb_generator = len(self.generator_list)
        self.batch_size = self.generator_list[0].batch_size
        self.steps_per_epoch = self.generator_list[0].steps_per_epoch
        self.epoch_index = 0

        self.assign_indexes()
        self.shuffle_indexes()

    def __len__(self):
        "Denotes the total number of batches"
        total_len = 0
        for local_generator in self.generator_list:
            total_len = total_len + local_generator.__len__()

        return total_len

    def assign_indexes(self):
        self.list_samples = []
        current_count = 0

        for generator_index, local_generator in enumerate(self.generator_list):
            local_len = local_generator.__len__()
            for index in np.arange(0, local_len):
                self.list_samples.append({"generator": generator_index, "index": index})
                current_count = current_count + 1

    def shuffle_indexes(self):
        np.random.shuffle(self.list_samples)

    def __getitem__(self, index):
        # Generate indexes of the batch
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        local_index = self.list_samples[index]

        local_generator = self.generator_list[local_index["generator"]]
        local_generator_index = local_index["index"]

        input_full, output_full = local_generator.__getitem__(local_generator_index)

        return input_full, output_full

    def on_epoch_end(self):
        if self.steps_per_epoch * (self.epoch_index + 2) <= self.__len__():
            self.epoch_index = self.epoch_index + 1
        else:
            # if we reach the end of the data, we roll over
            self.epoch_index = 0


class FmriGenerator(DeepGenerator):
    def __init__(self, json_path):
        super().__init__(json_path)

        self.raw_data_file = self.json_data["train_path"]
        self.batch_size = self.json_data["batch_size"]
        self.pre_post_x = self.json_data["pre_post_x"]
        self.pre_post_y = self.json_data["pre_post_y"]
        self.pre_post_z = self.json_data["pre_post_z"]
        self.pre_post_t = self.json_data["pre_post_t"]

        self.start_frame = self.json_data["start_frame"]
        self.end_frame = self.json_data["end_frame"]
        self.total_nb_block = self.json_data["total_nb_block"]
        self.steps_per_epoch = self.json_data["steps_per_epoch"]

        if "center_omission_size" in self.json_data.keys():
            self.center_omission_size = self.json_data["center_omission_size"]
        else:
            self.center_omission_size = 1

        if "single_voxel_output_single" in self.json_data.keys():
            self.single_voxel_output_single = self.json_data[
                "single_voxel_output_single"
            ]
        else:
            self.single_voxel_output_single = True

        if "initialize_list" in self.json_data.keys():
            self.initialize_list = self.json_data["initialize_list"]
        else:
            self.initialize_list = 1

        # We load the entire data as it fits into memory
        self.raw_data = nib.load(self.raw_data_file).get_fdata()
        self.data_shape = self.raw_data.shape

        middle_vol = np.round(np.array(self.data_shape) / 2).astype("int")
        range_middle = np.round(np.array(self.data_shape) / 4).astype("int")

        # We take the middle of the volume
        # and time for range estimation to avoid edge effects
        local_center_data = self.raw_data[
            middle_vol[0] - range_middle[0] : middle_vol[0] + range_middle[0],
            middle_vol[1] - range_middle[1] : middle_vol[1] + range_middle[1],
            middle_vol[2] - range_middle[2] : middle_vol[2] + range_middle[2],
            middle_vol[3] - range_middle[3] : middle_vol[3] + range_middle[3],
        ]
        self.local_mean = np.mean(local_center_data.flatten())
        self.local_std = np.std(local_center_data.flatten())
        self.epoch_index = 0

        if self.initialize_list == 1:
            self.x_list = []
            self.y_list = []
            self.z_list = []
            self.t_list = []

            filling_array = np.zeros(self.data_shape, dtype=bool)

            for index, value in enumerate(range(self.total_nb_block)):
                retake = True
                while retake:
                    x_local, y_local, z_local, t_local = self.get_random_xyzt()
                    retake = False
                    if filling_array[x_local, y_local, z_local, t_local]:
                        retake = True

                filling_array[x_local, y_local, z_local, t_local] = True

                self.x_list.append(x_local)
                self.y_list.append(y_local)
                self.z_list.append(z_local)
                self.t_list.append(t_local)

    def get_random_xyzt(self):
        x_center = np.random.randint(0, self.data_shape[0])
        y_center = np.random.randint(0, self.data_shape[1])
        z_center = np.random.randint(0, self.data_shape[2])
        t_center = np.random.randint(self.start_frame, self.end_frame)

        return x_center, y_center, z_center, t_center

    def __len__(self):
        "Denotes the total number of batches"
        return int(np.floor(float(len(self.x_list) / self.batch_size)))

    def on_epoch_end(self):
        if self.steps_per_epoch * (self.epoch_index + 2) <= self.__len__():
            self.epoch_index = self.epoch_index + 1
        else:
            # if we reach the end of the data, we roll over
            self.epoch_index = 0

    def __getitem__(self, index):
        # This is to ensure we are going through the
        # entire data when steps_per_epoch<self.__len__
        index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        indexes = np.arange(index * self.batch_size, (index + 1) * self.batch_size)

        input_full = np.zeros(
            [
                self.batch_size,
                self.pre_post_x * 2 + 1,
                self.pre_post_y * 2 + 1,
                self.pre_post_z * 2 + 1,
                self.pre_post_t * 2 + 1,
            ],
            dtype="float32",
        )

        if self.single_voxel_output_single:
            output_full = np.zeros([self.batch_size, 1, 1, 1, 1], dtype="float32")
        else:
            output_full = np.zeros(
                [
                    self.batch_size,
                    self.pre_post_x * 2 + 1,
                    self.pre_post_y * 2 + 1,
                    self.pre_post_z * 2 + 1,
                    1,
                ],
                dtype="float32",
            )

        for batch_index, sample_index in enumerate(indexes):

            local_x = self.x_list[sample_index]
            local_y = self.y_list[sample_index]
            local_z = self.z_list[sample_index]
            local_t = self.t_list[sample_index]

            input, output = self.__data_generation__(local_x, local_y, local_z, local_t)

            input_full[batch_index, :, :, :, :] = input
            output_full[batch_index, :, :, :, :] = output

        return input_full, output_full

    def __data_generation__(self, local_x, local_y, local_z, local_t):
        "Generates data containing batch_size samples"

        input_full = np.zeros(
            [
                1,
                self.pre_post_x * 2 + 1,
                self.pre_post_y * 2 + 1,
                self.pre_post_z * 2 + 1,
                self.pre_post_t * 2 + 1,
            ],
            dtype="float32",
        )

        if self.single_voxel_output_single:
            output_full = np.zeros([1, 1, 1, 1, 1], dtype="float32")
        else:
            output_full = np.zeros(
                [
                    1,
                    self.pre_post_x * 2 + 1,
                    self.pre_post_y * 2 + 1,
                    self.pre_post_z * 2 + 1,
                    1,
                ],
                dtype="float32",
            )

        # We cap the x axis when touching the limit of the volume
        if local_x - self.pre_post_x < 0:
            pre_x = local_x
        else:
            pre_x = self.pre_post_x
        if local_x + self.pre_post_x > self.data_shape[0] - 1:
            post_x = self.data_shape[0] - 1 - local_x
        else:
            post_x = self.pre_post_x

        # We cap the y axis when touching the limit of the volume
        if local_y - self.pre_post_y < 0:
            pre_y = local_y
        else:
            pre_y = self.pre_post_y
        if local_y + self.pre_post_y > self.data_shape[1] - 1:
            post_y = self.data_shape[1] - 1 - local_y
        else:
            post_y = self.pre_post_y

        # We cap the z axis when touching the limit of the volume
        if local_z - self.pre_post_z < 0:
            pre_z = local_z
        else:
            pre_z = self.pre_post_z
        if local_z + self.pre_post_z > self.data_shape[2] - 1:
            post_z = self.data_shape[2] - 1 - local_z
        else:
            post_z = self.pre_post_z

        # We cap the t axis when touching the limit of the volume
        if local_t - self.pre_post_t < 0:
            pre_t = local_t
        else:
            pre_t = self.pre_post_t
        if local_t + self.pre_post_t > self.data_shape[3] - 1:
            post_t = self.data_shape[3] - 1 - local_t
        else:
            post_t = self.pre_post_t

        input_full[
            0,
            (self.pre_post_x - pre_x) : (self.pre_post_x + post_x + 1),
            (self.pre_post_y - pre_y) : (self.pre_post_y + post_y + 1),
            (self.pre_post_z - pre_z) : (self.pre_post_z + post_z + 1),
            (self.pre_post_t - pre_t) : (self.pre_post_t + post_t + 1),
        ] = self.raw_data[
            (local_x - pre_x) : (local_x + post_x + 1),
            (local_y - pre_y) : (local_y + post_y + 1),
            (local_z - pre_z) : (local_z + post_z + 1),
            (local_t - pre_t) : (local_t + post_t + 1),
        ]
        if self.single_voxel_output_single:
            output_full[0, 0, 0, 0, 0] = input_full[
                0, self.pre_post_x, self.pre_post_y, self.pre_post_z, self.pre_post_t
            ]
        else:
            output_full[0, :, :, :, 0] = input_full[0, :, :, :, self.pre_post_t]

        input_full[
            0, self.pre_post_x, self.pre_post_y, self.pre_post_z, self.pre_post_t
        ] = 0

        if self.center_omission_size > 1:
            local_hole = self.center_omission_size - 1
            input_full[
                0,
                (self.pre_post_x - local_hole) : (self.pre_post_x + local_hole),
                (self.pre_post_y - local_hole) : (self.pre_post_y + local_hole),
                (self.pre_post_z - local_hole) : (self.pre_post_z + local_hole),
                self.pre_post_t,
            ] = 0

        input_full = (input_full.astype("float32") - self.local_mean) / self.local_std
        output_full = (output_full.astype("float32") - self.local_mean) / self.local_std

        return input_full, output_full


class SequentialGenerator(DeepGenerator):
    """This generator stores shared code across generators that have a
    continous temporal direction upon which start_frame, end_frame,
    pre_frame,... are used to to generate a list of samples. It is an
    intermediary class that is meant to be extended with details of
    how datasets are loaded."""

    def __init__(self, json_path):
        "Initialization"
        super().__init__(json_path)

        # We first store the relevant parameters
        if "pre_post_frame" in self.json_data.keys():
            self.pre_frame = self.json_data["pre_post_frame"]
            self.post_frame = self.json_data["pre_post_frame"]
        else:
            self.pre_frame = self.json_data["pre_frame"]
            self.post_frame = self.json_data["post_frame"]

        if "total_samples" in self.json_data.keys():
            self.total_samples = self.json_data["total_samples"]
        else:
            self.total_samples = -1

        if "randomize" in self.json_data.keys():
            self.randomize = self.json_data["randomize"]
        else:
            self.randomize = True

        if "pre_post_omission" in self.json_data.keys():
            self.pre_post_omission = self.json_data["pre_post_omission"]
        else:
            self.pre_post_omission = 0

        # load parameters that are related to training jobs
        self.batch_size = self.json_data["batch_size"]
        self.steps_per_epoch = self.json_data["steps_per_epoch"]

        # Loading limit parameters
        self.start_frame = self.json_data["start_frame"]
        self.end_frame = self.json_data["end_frame"]

        # start_frame starts at 0
        # end_frame is compatible with negative frames. -1 is the last
        # frame.

        # We initialize the epoch counter
        self.epoch_index = 0

    def _update_end_frame(self, total_frame_per_movie):
        """Update end_frame based on the total number of frames available.
        This allows for truncating the end of the movie when end_frame is
        negative."""

        # This is to handle selecting the end of the movie
        if self.end_frame < 0:
            self.end_frame = total_frame_per_movie + self.end_frame
        elif total_frame_per_movie <= self.end_frame:
            self.end_frame = total_frame_per_movie - 1

    def _calculate_list_samples(self, total_frame_per_movie):

        # We first cut if start and end frames are too close to the edges.
        self.start_sample = np.max(
            [self.pre_frame + self.pre_post_omission, self.start_frame]
        )
        self.end_sample = np.min(
            [
                self.end_frame,
                total_frame_per_movie - 1 - self.post_frame - self.pre_post_omission,
            ]
        )

        if (self.end_sample - self.start_sample + 1) < self.batch_size:
            raise Exception(
                "Not enough frames to construct one "
                + str(self.batch_size)
                + " frame(s) batch between "
                + str(self.start_sample)
                + " and "
                + str(self.end_sample)
                + " frame number."
            )

        # +1 to make sure end_samples is included
        self.list_samples = np.arange(self.start_sample, self.end_sample + 1)

        if self.randomize:
            np.random.shuffle(self.list_samples)

        # We cut the number of samples if asked to
        if self.total_samples > 0 and self.total_samples < len(self.list_samples):
            self.list_samples = self.list_samples[0 : self.total_samples]

    def on_epoch_end(self):
        """We only increase index if steps_per_epoch is set to positive value.
        -1 will force the generator to not iterate at the end of each epoch."""
        if self.steps_per_epoch > 0:
            if self.steps_per_epoch * (self.epoch_index + 2) <= self.__len__():
                self.epoch_index = self.epoch_index + 1
            else:
                # if we reach the end of the data, we roll over
                self.epoch_index = 0

    def __len__(self):
        "Denotes the total number of batches"
        return math.ceil(len(self.list_samples) / self.batch_size)

    def generate_batch_indexes(self, index):
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        start_ind = index * self.batch_size
        end_ind = (index + 1) * self.batch_size
        if end_ind < self.list_samples.shape[0]:
            indexes = np.arange(start_ind, end_ind)
            shuffle_indexes = self.list_samples[indexes]
        else:
            shuffle_indexes = self.list_samples[start_ind:]
        return shuffle_indexes


class EphysGenerator(SequentialGenerator):
    """This generator is used when dealing with a single dat file storing a
    continous raw neuropixel recording as a (time, 384, 2) int16 array."""

    def __init__(self, json_path):
        "Initialization"
        super().__init__(json_path)

        self.raw_data_file = self.json_data["train_path"]
        self.nb_probes = 384

        self.raw_data = np.memmap(self.raw_data_file, dtype="int16")
        self.total_frame_per_movie = int(self.raw_data.size / self.nb_probes)

        self._update_end_frame(self.total_frame_per_movie)
        self._calculate_list_samples(self.total_frame_per_movie)

        # We calculate the mean and std of the data
        average_nb_samples = 200000

        shape = (self.total_frame_per_movie, int(self.nb_probes / 2), 2)
        # load it with the correct shape
        self.raw_data = np.memmap(self.raw_data_file, dtype="int16", shape=shape)

        local_data = self.raw_data[0:average_nb_samples, :, :].flatten()
        local_data = local_data.astype("float32")
        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)

        shape = (self.total_frame_per_movie, int(self.nb_probes / 2), 2)

        # load it with the correct shape
        self.raw_data = np.memmap(self.raw_data_file, dtype="int16", shape=shape)

    def __getitem__(self, index):
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        shuffle_indexes = self.generate_batch_indexes(index)
        local_batch_size = shuffle_indexes.shape[0]
        input_full = np.zeros(
            [
                local_batch_size,
                int(self.nb_probes),
                2,
                self.pre_frame + self.post_frame,
            ],
            dtype="float32",
        )
        output_full = np.zeros(
            [local_batch_size, int(self.nb_probes), 2, 1], dtype="float32"
        )

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        "Generates data containing batch_size samples"

        # We reorganize to follow true geometry of probe for convolution
        input_full = np.zeros(
            [1, self.nb_probes, 2, self.pre_frame + self.post_frame], dtype="float32"
        )
        output_full = np.zeros([1, self.nb_probes, 2, 1], dtype="float32")

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        input_index = input_index[input_index != index_frame]

        for index_padding in np.arange(self.pre_post_omission + 1):
            input_index = input_index[input_index != index_frame - index_padding]
            input_index = input_index[input_index != index_frame + index_padding]

        data_img_input = self.raw_data[input_index, :, :]
        data_img_output = self.raw_data[index_frame, :, :]

        data_img_input = np.swapaxes(data_img_input, 1, 2)
        data_img_input = np.swapaxes(data_img_input, 0, 2)

        data_img_input = (
            data_img_input.astype("float32") - self.local_mean
        ) / self.local_std
        data_img_output = (
            data_img_output.astype("float32") - self.local_mean
        ) / self.local_std

        # alternating filling with zeros padding
        even = np.arange(0, self.nb_probes, 2)
        odd = even + 1

        input_full[0, even, 0, :] = data_img_input[:, 0, :]
        input_full[0, odd, 1, :] = data_img_input[:, 1, :]

        output_full[0, even, 0, 0] = data_img_output[:, 0]
        output_full[0, odd, 1, 0] = data_img_output[:, 1]

        return input_full, output_full


class MultiContinuousTifGenerator(SequentialGenerator):
    """This generator is used when dealing with a continuous movie split
    in multiple blocks of frames each within individual tif files as is
    typically generated by ScanImage. The provided path will be a folder
    storing all tif files. The filenames must be sorted alphabetically
    to ensure continuity"""

    def __init__(self, json_path):
        "Initialization"
        super().__init__(json_path)

        # For backward compatibility
        if "train_path" in self.json_data.keys():
            self.raw_data_file = self.json_data["train_path"]
        else:
            self.raw_data_file = self.json_data["movie_path"]

        self.list_tif_files = glob.glob(os.path.join(self.raw_data_file, "*.tif"))

        # We sort the list to make sure we are alphabetical
        self.list_tif_files.sort()

        self.list_raw_data = []
        self.total_frame_per_movie = 0
        self.list_bounds = [0]
        for indiv_tif in self.list_tif_files:
            with tifffile.TiffFile(indiv_tif) as tif:
                local_raw_data = tif.asarray()
                self.list_raw_data.append(local_raw_data)
                self.total_frame_per_movie += local_raw_data.shape[0]
                self.list_bounds.append(self.total_frame_per_movie)

        self.list_bounds = np.array(self.list_bounds)

        self._update_end_frame(self.total_frame_per_movie)
        self._calculate_list_samples(self.total_frame_per_movie)

        average_nb_samples = 1000

        local_data = self.get_raw_frames_from_list(np.arange(0, average_nb_samples))
        local_data = local_data.astype("float32").flatten()
        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)
        self.epoch_index = 0

    def get_list_frame_and_index(self, frame_index):
        list_index = np.where((self.list_bounds - frame_index) <= 0)[0][-1]
        start_index = self.list_bounds[list_index]
        index_in_movie = frame_index - start_index

        return list_index, index_in_movie

    def get_raw_frames_from_list(self, frame_indexes):
        if np.size(frame_indexes) == 1:
            list_index, index_in_movie = self.get_list_frame_and_index(frame_indexes)
            data_img_input = self.list_raw_data[list_index][index_in_movie, :, :]
        else:
            data_img_input = np.zeros(
                [
                    len(frame_indexes),
                    self.list_raw_data[0].shape[1],
                    self.list_raw_data[0].shape[2],
                ]
            )

            for index, indiv_frame in enumerate(frame_indexes):
                list_index, index_in_movie = self.get_list_frame_and_index(indiv_frame)

                data_img_input[index, :, :] = self.list_raw_data[list_index][
                    index_in_movie, :, :
                ]

        return data_img_input

    def __getitem__(self, index):
        shuffle_indexes = self.generate_batch_indexes(index)

        input_full = np.zeros(
            [
                self.batch_size,
                512,
                512,
                self.pre_frame + self.post_frame,
            ],
            dtype="float32",
        )
        output_full = np.zeros(
            [self.batch_size, 512, 512, 1],
            dtype="float32",
        )

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        "Generates data containing batch_size samples"

        # X : (n_samples, *dim, n_channels)
        input_full = np.zeros(
            [1, 512, 512, self.pre_frame + self.post_frame],
            dtype="float32",
        )
        output_full = np.zeros([1, 512, 512, 1], dtype="float32")

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        input_index = input_index[input_index != index_frame]

        for index_padding in np.arange(self.pre_post_omission + 1):
            input_index = input_index[input_index != index_frame - index_padding]
            input_index = input_index[input_index != index_frame + index_padding]

        data_img_input = self.get_raw_frames_from_list(input_index)
        data_img_output = self.get_raw_frames_from_list(index_frame)

        data_img_input = np.swapaxes(data_img_input, 1, 2)
        data_img_input = np.swapaxes(data_img_input, 0, 2)

        img_in_shape = data_img_input.shape
        img_out_shape = data_img_output.shape

        data_img_input = (
            data_img_input.astype("float32") - self.local_mean
        ) / self.local_std
        data_img_output = (
            data_img_output.astype("float32") - self.local_mean
        ) / self.local_std
        input_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_input
        output_full[0, : img_out_shape[0], : img_out_shape[1], 0] = data_img_output

        return input_full, output_full


class SingleTifGenerator(SequentialGenerator):
    """This generator is used when dealing with a single tif file storing a
    continous movie recording. Each frame can be arbitrary (x,y) size but
    should be consistent through training. a maximum of 1000 frames are pulled
    from the beginning of the movie to estimate mean and std."""

    def __init__(self, json_path):
        "Initialization"
        super().__init__(json_path)

        self.raw_data_file = self.json_data["train_path"]

        with tifffile.TiffFile(self.raw_data_file) as tif:
            self.raw_data = tif.asarray()

        self.total_frame_per_movie = self.raw_data.shape[0]

        self._update_end_frame(self.total_frame_per_movie)
        self._calculate_list_samples(self.total_frame_per_movie)

        average_nb_samples = np.min([self.total_frame_per_movie, 1000])
        local_data = self.raw_data[0:average_nb_samples, :, :].flatten()
        local_data = local_data.astype("float32")
        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)
        self.epoch_index = 0

    def __getitem__(self, index):
        shuffle_indexes = self.generate_batch_indexes(index)

        input_full = np.zeros(
            [
                self.batch_size,
                self.raw_data.shape[1],
                self.raw_data.shape[2],
                self.pre_frame + self.post_frame,
            ],
            dtype="float32",
        )
        output_full = np.zeros(
            [self.batch_size, self.raw_data.shape[1], self.raw_data.shape[2], 1],
            dtype="float32",
        )

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        "Generates data containing batch_size samples"

        # X : (n_samples, *dim, n_channels)

        input_full = np.zeros(
            [
                1,
                self.raw_data.shape[1],
                self.raw_data.shape[2],
                self.pre_frame + self.post_frame,
            ],
            dtype="float32",
        )
        output_full = np.zeros(
            [1, self.raw_data.shape[1], self.raw_data.shape[2], 1], dtype="float32"
        )

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        input_index = input_index[input_index != index_frame]

        for index_padding in np.arange(self.pre_post_omission + 1):
            input_index = input_index[input_index != index_frame - index_padding]
            input_index = input_index[input_index != index_frame + index_padding]

        data_img_input = self.raw_data[input_index, :, :]
        data_img_output = self.raw_data[index_frame, :, :]

        data_img_input = np.swapaxes(data_img_input, 1, 2)
        data_img_input = np.swapaxes(data_img_input, 0, 2)

        img_in_shape = data_img_input.shape
        img_out_shape = data_img_output.shape

        data_img_input = (
            data_img_input.astype("float32") - self.local_mean
        ) / self.local_std
        data_img_output = (
            data_img_output.astype("float32") - self.local_mean
        ) / self.local_std
        input_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_input
        output_full[0, : img_out_shape[0], : img_out_shape[1], 0] = data_img_output

        return input_full, output_full


class OphysGenerator(SequentialGenerator):
    """This generator is used when dealing with a single hdf5 file storing a
    continous movie recording into a 'data' field as [time, x, y]. Each
    frame is expected to be smaller than (512,512)."""

    def __init__(self, json_path: Union[str, Path]):
        "Initialization"
        super().__init__(json_path)

        # For backward compatibility
        if "train_path" in self.json_data.keys():
            self.raw_data_file = self.json_data["train_path"]
        else:
            self.raw_data_file = self.json_data["movie_path"]

        self.batch_size = self.json_data["batch_size"]
        movie_obj_point = h5py.File(self.raw_data_file, "r")
        raw_data = movie_obj_point["data"]
        self.movie_dim = raw_data.shape[1:]
        self.total_frame_per_movie = int(raw_data.shape[0])

        self._update_end_frame(self.total_frame_per_movie)
        self._calculate_list_samples(self.total_frame_per_movie)
        self.movie_statistics_sample_size = self.json_data.get(
            "movie_statistics_sample_size", 100
        )
        average_nb_samples = np.min(
            [int(raw_data.shape[0]), self.movie_statistics_sample_size]
        )

        # For backward compatibility
        if "cache_data" in self.json_data.keys():
            self.cache_data = self.json_data["cache_data"]
        else:
            self.cache_data = False

        if self.cache_data:
            logger.info("Caching hdf5 file... \n")
            if self.end_frame > 0:
                self.raw_data = raw_data[
                    0 : np.min(
                        [
                            self.total_frame_per_movie,
                            self.end_frame
                            + self.post_frame
                            + self.pre_post_omission
                            + 1,
                        ]
                    ),
                    :,
                    :,
                ]
            else:
                self.raw_data = raw_data[:, :, :]
            local_data = self.raw_data[0:average_nb_samples, :, :].flatten()
        else:
            local_data = raw_data[0:average_nb_samples, :, :].flatten()
            self.cache_data = False

        local_data = local_data.astype("float32")

        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)

        if self.cache_data:
            self.raw_data = (
                self.raw_data.astype("float") - self.local_mean
            ) / self.local_std

        movie_obj_point.close()

    def __getitem__(self, index: int):
        shuffle_indexes = self.generate_batch_indexes(index)
        local_batch_size = shuffle_indexes.shape[0]
        input_full = np.zeros(
            [
                local_batch_size,
                self.movie_dim[0],
                self.movie_dim[1],
                self.pre_frame + self.post_frame,
            ],
            dtype="float32",
        )

        output_full = np.zeros(
            [local_batch_size, self.movie_dim[0], self.movie_dim[1], 1], dtype="float32"
        )

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)
            X_shape = X.shape
            Y_shape = Y.shape

            input_full[batch_index, : X_shape[0], : X_shape[1], :] = X
            output_full[batch_index, : Y_shape[0], : Y_shape[1], 0] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame: int):
        "Generates data containing batch_size samples"

        if self.cache_data:
            movie_obj = self.raw_data
        else:
            movie_obj_point = h5py.File(self.raw_data_file, "r")
            movie_obj = movie_obj_point["data"]

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        input_index = input_index[input_index != index_frame]

        for index_padding in np.arange(self.pre_post_omission + 1):
            input_index = input_index[input_index != index_frame - index_padding]
            input_index = input_index[input_index != index_frame + index_padding]

        # If data was cached we do not need to normalize. this was done
        # at once to minimize compute
        if self.cache_data:
            data_img_input = movie_obj[input_index, :, :]
            data_img_output = movie_obj[index_frame, :, :]
        else:
            data_img_input = (
                movie_obj[input_index, :, :].astype("float") - self.local_mean
            ) / self.local_std
            data_img_output = (
                movie_obj[index_frame, :, :].astype("float") - self.local_mean
            ) / self.local_std

        data_img_input = np.swapaxes(data_img_input, 1, 2)
        data_img_input = np.swapaxes(data_img_input, 0, 2)

        if not self.cache_data:
            movie_obj_point.close()

        return data_img_input, data_img_output


class InferenceOphysGenerator(SequentialGenerator):
    """This generator is a modified version of OphysGenerator that has
    been optimized when used for inference given randomize==False.
    """

    def __init__(self, json_path: Union[str, Path]):
        "Initialization"
        super().__init__(json_path)
        self._gpu_available = tf.test.is_gpu_available()
        self._movie_data = None
        if self._gpu_available:
            self._batch_tensor_index = (
                None  # Current index cached as a batch tensor on the GPU
            )
            self._batch_tensor = None

        # For backward compatibility
        self.cache_data = self.json_data.get("cache_data", False)
        self.raw_data_file = self.json_data.get(
            "train_path", self.json_data.get("movie_path")
        )
        if self._gpu_available:
            self.gpu_cache_full = self.json_data.get("gpu_cache_full", False)
        else:
            self.gpu_cache_full = False
        self.normalize_cache = self.json_data.get("normalize_cache", False)
        self.batch_size = self.json_data["batch_size"]
        self.movie_statistics_sample_size = self.json_data.get(
            "movie_statistics_sample_size", 100
        )

        self.total_frame_per_movie = int(self.movie_data.shape[0])

        self._update_end_frame(self.total_frame_per_movie)
        self._calculate_list_samples(self.total_frame_per_movie)

    @property
    def movie_data(self) -> Union[np.ndarray, tf.Tensor]:
        if self._movie_data is None:
            with h5py.File(self.raw_data_file, "r") as movie_obj:
                end_ind = self.end_frame + self.post_frame + self.pre_post_omission + 1
                total_frame_per_movie = movie_obj["data"].shape[0]
                if self.end_frame > 0 and total_frame_per_movie > end_ind:
                    movie_data = movie_obj["data"][:end_ind]
                else:
                    movie_data = movie_obj["data"][()]
            average_nb_samples = np.min(
                [movie_data.shape[0], self.movie_statistics_sample_size]
            )
            local_data = movie_data[:average_nb_samples]
            local_data = local_data.astype("float32")
            self.local_mean = local_data.mean()
            self.local_std = local_data.std()
            if self.gpu_cache_full:
                logger.info("Caching full movie onto GPU")
                # tf.convert_to_tensor copies the object onto GPU if it's available
                movie_data = tf.convert_to_tensor(movie_data, dtype="float")
            if self.normalize_cache or self.gpu_cache_full:
                if not self.gpu_cache_full:
                    movie_data = movie_data.astype("float32")
                movie_data = self._normalize(
                    movie_data, self.local_mean, self.local_std
                )
            self._movie_data = movie_data
        return self._movie_data

    def __get_batch_tensor(self, index: int, batch_indices: np.ndarray):
        """Slices minimum movie required to generate batch for a given batch
        index and caches it onto the GPU. If a previous batch is cached, a
        new batch is generated by only copying the difference in frames.
        """
        start_ind = index * self.batch_size
        local_batch_size = batch_indices.shape[0]
        end_ind = (
            start_ind
            + local_batch_size
            + self.pre_frame
            + self.post_frame
            + 2 * self.pre_post_omission
        )
        if self._batch_tensor_index == index - 1:
            # cache the difference in frames between the current index and prev index
            # tf.convert_to_tensor copies the object onto GPU if it's available
            batch_frames = tf.convert_to_tensor(
                self.movie_data[end_ind - local_batch_size : end_ind], dtype="float"
            )
            if not self.normalize_cache:
                batch_frames = self._normalize(
                    batch_frames, self.local_mean, self.local_std
                )
            self._batch_tensor = tf.concat(
                [self._batch_tensor[self.batch_size :], batch_frames], 0
            )
            self._batch_tensor_index = index

        elif self._batch_tensor_index == index:
            return self._batch_tensor

        else:
            # cache the minimum movie required to generate a batch
            # tf.convert_to_tensor copies the object onto GPU if it's available
            self._batch_tensor = tf.convert_to_tensor(
                self.movie_data[start_ind:end_ind], dtype="float"
            )
            self._batch_tensor_index = index
            if not self.normalize_cache:
                self._batch_tensor = self._normalize(
                    self._batch_tensor, self.local_mean, self.local_std
                )
        return self._batch_tensor

    def __getitem__(self, index: int):
        # Get indices for target frames and their inputs
        batch_indices = self.generate_batch_indexes(index)
        if self._gpu_available and not self.gpu_cache_full:
            data_tensor = self.__get_batch_tensor(index, batch_indices)
            batch_indices = batch_indices - index * self.batch_size
            input_indices = np.vstack(
                [
                    self.__get_sample_input_indices(frame_index)
                    for frame_index in batch_indices
                ]
            )
        else:
            data_tensor = self.movie_data
            input_indices = np.vstack(
                [
                    self.__get_sample_input_indices(frame_index)
                    for frame_index in batch_indices
                ]
            )

        # Slice movie with indices to generate input and outputs
        if self._gpu_available:
            input_full = tf.gather(data_tensor, input_indices)
            output_full = tf.gather(data_tensor, batch_indices)
            input_full = tf.transpose(input_full, perm=[0, 2, 3, 1])
            output_full = tf.expand_dims(output_full, -1)
            # dims (sample, x, y, frames)
        else:
            input_full = self.movie_data[input_indices].astype("float32")
            output_full = self.movie_data[batch_indices].astype("float32")
            if not self.normalize_cache:
                input_full = self._normalize(
                    input_full, self.local_mean, self.local_std
                )
                output_full = self._normalize(
                    output_full, self.local_mean, self.local_std
                )
            input_full = np.moveaxis(input_full, 1, -1)
            output_full = np.expand_dims(output_full, -1)
        return input_full, output_full

    def __get_sample_input_indices(self, index_frame: int):
        input_index_left = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame - self.pre_post_omission,
        )
        input_index_right = np.arange(
            index_frame + self.pre_post_omission + 1,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        input_index = np.concatenate([input_index_left, input_index_right])
        return input_index


class MovieJSONGenerator(DeepGenerator):
    """This generator is used when dealing with a large number of hdf5 files
    referenced into a json file with pre-computed mean and std value. The json
    file is passed to the generator in place of the movie file themselves. Each
    frame is expected to be smaller than (512,512).
    Each individual hdf5 movie is recorded into a 'data' field
    as [time, x, y]. The json files is pre-calculated and have the following
    fields (replace <...> appropriately):
    {"<id>": {"path": <string path to the hdf5 file>,
    "frames": <[int frame1, int frame2,...]>,
    "mean": <float value>,
    "std": <float_value>}}"""

    def __init__(
        self,
        json_path: Union[str, Path],
        movs: Optional[Dict[int, np.ndarray]] = None
    ):
        """
        Parameters
        ----------
        json_path
        movs
            Maps ophys experiment id to movie, in order to read movie from
            RAM rather than disk, to speed up training
        """
        super().__init__(json_path)

        if movs is None:
            movs = {}

        self.sample_data_path_json = self.json_data.get(
            "train_path", self.json_data.get("data_path")
        )
        self.batch_size = self.json_data.get("batch_size", 16)
        self.steps_per_epoch = self.json_data.get("steps_per_epoch")
        self.epoch_index = 0
        self.randomize = self.json_data.get("randomize", True)
        self.seed = self.json_data.get("seed", 1234)
        self._movs = movs

        # For backward compatibility
        if "pre_post_frame" in self.json_data.keys():
            self.pre_frame = self.json_data["pre_post_frame"]
            self.post_frame = self.json_data["pre_post_frame"]
        else:
            self.pre_frame = self.json_data["pre_frame"]
            self.post_frame = self.json_data["post_frame"]

        # For backward compatibility
        if "pre_post_omission" in self.json_data.keys():
            self.pre_post_omission = self.json_data["pre_post_omission"]
        else:
            self.pre_post_omission = 0

        with open(self.sample_data_path_json, "r") as json_handle:
            self.frame_data_location = json.load(json_handle)

        self.lims_id = list(self.frame_data_location.keys())

        self.shuffled_data_list = []
        for ophys_experiment_id in self.lims_id:
            n_frames = len(self.frame_data_location[ophys_experiment_id]["frames"])
            for i_frame in range(n_frames):
                self.shuffled_data_list.append((ophys_experiment_id, i_frame))

        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.shuffled_data_list)
        self._make_index_to_frames()

    def on_epoch_end(self):
        # We only increase index if steps_per_epoch
        # is set to positive value. -1 will force the generator
        # to not iterate at the end of each epoch
        if self.steps_per_epoch > 0:
            self.epoch_index = self.epoch_index + 1

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        if self.steps_per_epoch is not None and self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        n_all_data = len(self.shuffled_data_list)
        if (index + 1) * self.batch_size > n_all_data:
            indexes = np.arange(index * self.batch_size, n_all_data)
        else:
            indexes = np.arange(index * self.batch_size, (index + 1) * self.batch_size)

        if len(self.lims_id) == 1 and self._movs:
            # 2D slicing of cached movie if only one movie exists
            output_indices = [self.shuffled_data_list[index][1] for index in indexes]
            input_indices = np.vstack(
                [
                    self.__get_sample_input_indices(frame_index)
                    for frame_index in output_indices
                ]
            )
            input_full = self._movs[self.lims_id[0]][input_indices]
            output_full = self._movs[self.lims_id[0]][output_indices]
            input_full = np.moveaxis(input_full, 1, -1)
            output_full = np.expand_dims(output_full, -1)

        else:
            actual_batch_size = len(indexes)
            input_full = np.zeros(
                [actual_batch_size, 512, 512, self.pre_frame + self.post_frame]
            )
            output_full = np.zeros([actual_batch_size, 512, 512, 1])

            for batch_index, frame_index in enumerate(indexes):
                X, Y = self.__data_generation__(frame_index)

                input_full[batch_index, :, :, :] = X
                output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __get_norm_parameters__(self, index_frame: int) -> Tuple[float, float]:
        local_lims, local_img = self.shuffled_data_list[index_frame]
        local_mean = self.frame_data_location[local_lims]["mean"]
        local_std = self.frame_data_location[local_lims]["std"]

        return local_mean, local_std

    def __get_sample_input_indices(self, index_frame: int):
        input_index_left = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame - self.pre_post_omission,
        )
        input_index_right = np.arange(
            index_frame + self.pre_post_omission + 1,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        input_index = np.concatenate([input_index_left, input_index_right])
        return input_index

    def _make_index_to_frames(self):
        """
        Construct a lookup that goes from video_index, img_index
        to an index of input and outputframes
        """
        self.frame_lookup = dict()
        for ophys_experiment_id in self.lims_id:
            local_frame_data = self.frame_data_location[ophys_experiment_id]
            for img_index in range(len(local_frame_data["frames"])):
                output_frame = local_frame_data["frames"][img_index]

                input_index_left = np.arange(
                    output_frame - self.pre_frame - self.pre_post_omission,
                    output_frame - self.pre_post_omission,
                )

                input_index_right = np.arange(
                    output_frame + self.pre_post_omission + 1,
                    output_frame + self.post_frame + self.pre_post_omission + 1,
                )

                input_index = np.concatenate([input_index_left, input_index_right])

                self.frame_lookup[(ophys_experiment_id, img_index)] = {
                    "output_frame": output_frame,
                    "input_index": input_index,
                }

    def _data_from_indexes(
        self, video_index: int, img_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate X, Y dataset given a video_index and img_index"""

        index_dict = self.frame_lookup[(video_index, img_index)]
        input_index = index_dict["input_index"]
        output_frame = index_dict["output_frame"]

        data_img_input = None
        data_img_output = None

        if data_img_input is None:
            motion_path = self.frame_data_location[video_index]["path"]

            if self._movs:
                data_img_input = self._movs[video_index][input_index]
                data_img_output = self._movs[video_index][output_frame]
            else:
                with h5py.File(motion_path, "r") as movie_obj:
                    data_img_input = movie_obj["data"][input_index]
                    data_img_output = movie_obj["data"][output_frame]

        local_frame_data = self.frame_data_location[video_index]
        local_mean = local_frame_data["mean"]
        local_std = local_frame_data["std"]

        input_full = np.zeros([1, 512, 512, len(input_index)])
        output_full = np.zeros([1, 512, 512, 1])

        data_img_input = np.moveaxis(data_img_input, 0, -1)

        img_in_shape = data_img_input.shape
        img_out_shape = data_img_output.shape

        data_img_input = (data_img_input.astype("float") - local_mean) / local_std
        data_img_output = (data_img_output.astype("float") - local_mean) / local_std
        input_full[0, : img_in_shape[0], : img_in_shape[1]] = data_img_input
        output_full[0, : img_out_shape[0], : img_out_shape[1], 0] = data_img_output

        return input_full, output_full

    def __data_generation__(self, index_frame: int) -> Tuple[np.ndarray, np.ndarray]:
        "Generates data containing batch_size samples"

        # X : (n_samples, *dim, n_channels)
        try:
            local_lims, local_img = self.shuffled_data_list[index_frame]

            return self._data_from_indexes(local_lims, local_img)

        except Exception as e:
            logger.error(f"Exception raised on lims_id: {self.lims_id}")
            logger.error(e)
            raise

    def __len__(self):
        "Denotes the total number of batches"
        n_frames = len(self.shuffled_data_list)
        return int(np.ceil(n_frames / self.batch_size))
