# Class to generate data for training
import numpy as np
import json
import h5py
import os
import tensorflow.keras as keras
from deepinterpolation.generic import JsonLoader
import tifffile
import nibabel as nib
import s3fs
import glob
import logging

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


class OnePGenerator(DeepGenerator):
    """
    This generator deliver data provided from an hdf5 file made
    from one photon miniscope data.

    Parameters:
    str: json_path: path to the json parameter file

    Returns:
    None
    """

    def __init__(self, json_path):

        super().__init__(json_path)

        self.raw_data_file = self.json_data["train_path"]

        # For backward compatibility
        if "pre_post_frame" in self.json_data.keys():
            self.pre_frame = self.json_data["pre_post_frame"]
            self.post_frame = self.json_data["pre_post_frame"]
        else:
            self.pre_frame = self.json_data["pre_frame"]
            self.post_frame = self.json_data["post_frame"]

        self.start_frame = self.json_data["start_frame"]

        # This is compatible with negative frames
        self.end_frame = self.json_data["end_frame"]

        self.raw_data = h5py.File(self.raw_data_file, "r")["1"]
        self.movie_size = self.raw_data.shape

        if self.end_frame < 0:
            self.img_per_movie = (
                int(self.raw_data.shape[0]) + 1 +
                self.end_frame - self.start_frame
            )
        else:
            self.img_per_movie = self.end_frame + 1 - self.start_frame

        self.local_raw_data = self.raw_data[:, :, :]

        list_nan = ~(np.isfinite(self.local_raw_data))
        self.local_raw_data[list_nan] = 0

        average_nb_samples = 50

        self.list_samples = np.arange(
            self.pre_frame + self.start_frame,
            self.start_frame + self.img_per_movie - self.post_frame,
        )

        np.random.shuffle(self.list_samples)
        local_data = self.local_raw_data[0:average_nb_samples, :, :].flatten()

        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)

        self.batch_size = self.json_data["batch_size"]

    def __len__(self):
        "Denotes the total number of batches"
        return int(np.floor(float(len(self.list_samples)) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        if (index + 1) * self.batch_size > self.img_per_movie:
            indexes = np.arange(index * self.batch_size, self.img_per_movie)
        else:
            indexes = np.arange(index * self.batch_size,
                                (index + 1) * self.batch_size)

        shuffle_indexes = self.list_samples[indexes]
        input_full = np.zeros(
            [
                self.batch_size,
                self.movie_size[1],
                self.movie_size[2],
                self.pre_frame + self.post_frame,
            ]
        )
        output_full = np.zeros(
            [self.batch_size, self.movie_size[1], self.movie_size[2], 1]
        )

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        "Generates data containing batch_size samples"
        # local_raw_data = h5py.File(self.raw_data_file, 'r')['1']

        input_full = np.zeros(
            [1,
             self.movie_size[1],
             self.movie_size[2],
             self.pre_frame + self.post_frame]
        )
        output_full = np.zeros([1,
                                self.movie_size[1],
                                self.movie_size[2], 1])

        input_index = np.arange(
            index_frame - self.pre_frame,
            index_frame + self.post_frame + 1
        )
        input_index = input_index[input_index != index_frame]

        data_img_input = self.local_raw_data[input_index, :, :]
        data_img_output = self.local_raw_data[index_frame, :, :]

        data_img_input = np.swapaxes(data_img_input, 1, 2)
        data_img_input = np.swapaxes(data_img_input, 0, 2)

        img_in_shape = data_img_input.shape
        img_out_shape = data_img_output.shape

        data_img_input = (
            data_img_input.astype("float") - self.local_mean
        ) / self.local_std
        data_img_output = (
            data_img_output.astype("float") - self.local_mean
        ) / self.local_std
        input_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_input
        output_full[0, : img_out_shape[0],
                    : img_out_shape[1], 0] = data_img_output

        return input_full, output_full


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
                self.list_samples.append(
                    {"generator": generator_index, "index": index})
                current_count = current_count + 1

    def shuffle_indexes(self):
        np.random.shuffle(self.list_samples)

    def __getitem__(self, index):
        # Generate indexes of the batch

        local_index = self.list_samples[index]

        local_generator = self.generator_list[local_index["generator"]]
        local_generator_index = local_index["index"]

        input_full, output_full = local_generator.__getitem__(
            local_generator_index)

        return input_full, output_full


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
            middle_vol[0] - range_middle[0]: middle_vol[0] + range_middle[0],
            middle_vol[1] - range_middle[1]: middle_vol[1] + range_middle[1],
            middle_vol[2] - range_middle[2]: middle_vol[2] + range_middle[2],
            middle_vol[3] - range_middle[3]: middle_vol[3] + range_middle[3],
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
                print(index)
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
        if self.steps_per_epoch * (self.epoch_index + 2) < self.__len__():
            self.epoch_index = self.epoch_index + 1
        else:
            # if we reach the end of the data, we roll over
            self.epoch_index = 0

    def __getitem__(self, index):
        # This is to ensure we are going through the
        # entire data when steps_per_epoch<self.__len__
        index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        indexes = np.arange(index * self.batch_size,
                            (index + 1) * self.batch_size)

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
            output_full = np.zeros(
                [self.batch_size, 1, 1, 1, 1], dtype="float32")
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

            input, output = self.__data_generation__(
                local_x, local_y, local_z, local_t)

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
            (self.pre_post_x - pre_x): (self.pre_post_x + post_x + 1),
            (self.pre_post_y - pre_y): (self.pre_post_y + post_y + 1),
            (self.pre_post_z - pre_z): (self.pre_post_z + post_z + 1),
            (self.pre_post_t - pre_t): (self.pre_post_t + post_t + 1),
        ] = self.raw_data[
            (local_x - pre_x): (local_x + post_x + 1),
            (local_y - pre_y): (local_y + post_y + 1),
            (local_z - pre_z): (local_z + post_z + 1),
            (local_t - pre_t): (local_t + post_t + 1),
        ]
        if self.single_voxel_output_single:
            output_full[0, 0, 0, 0, 0] = input_full[
                0, self.pre_post_x, self.pre_post_y,
                self.pre_post_z, self.pre_post_t
            ]
        else:
            output_full[0, :, :, :, 0] = input_full[0,
                                                    :, :, :, self.pre_post_t]

        input_full[
            0, self.pre_post_x, self.pre_post_y,
            self.pre_post_z, self.pre_post_t
        ] = 0

        if self.center_omission_size > 1:
            local_hole = self.center_omission_size - 1
            input_full[
                0,
                (self.pre_post_x - local_hole): (self.pre_post_x + local_hole),
                (self.pre_post_y - local_hole): (self.pre_post_y + local_hole),
                (self.pre_post_z - local_hole): (self.pre_post_z + local_hole),
                self.pre_post_t,
            ] = 0

        input_full = (input_full.astype("float32") -
                      self.local_mean) / self.local_std
        output_full = (output_full.astype("float32") -
                       self.local_mean) / self.local_std

        return input_full, output_full


class EphysGenerator(DeepGenerator):
    "Generates data for Keras"

    def __init__(self, json_path):
        "Initialization"
        super().__init__(json_path)

        self.raw_data_file = self.json_data["train_path"]
        self.batch_size = self.json_data["batch_size"]

        if "pre_post_frame" in self.json_data.keys():
            self.pre_frame = self.json_data["pre_post_frame"]
            self.post_frame = self.json_data["pre_post_frame"]
        else:
            self.pre_frame = self.json_data["pre_frame"]
            self.post_frame = self.json_data["post_frame"]

        self.pre_post_omission = self.json_data["pre_post_omission"]
        self.start_frame = self.json_data["start_frame"]
        self.steps_per_epoch = self.json_data["steps_per_epoch"]

        # This is used to limit the total number of samples
        # -1 means to take all and is the default fall back

        if "total_samples" in self.json_data.keys():
            self.total_samples = self.json_data["total_samples"]
        else:
            self.total_samples = -1

        # This is compatible with negative frames
        self.end_frame = self.json_data["end_frame"]

        self.nb_probes = 384

        self.raw_data = np.memmap(self.raw_data_file, dtype="int16")

        if self.end_frame < 0:
            self.img_per_movie = (
                int(self.raw_data.size / self.nb_probes)
                + 1
                + self.end_frame
                - self.start_frame
                - self.post_frame
                - self.pre_post_omission
            )
        elif int(self.raw_data.size / self.nb_probes) < self.end_frame:
            self.img_per_movie = (
                int(self.raw_data.size / self.nb_probes)
                - self.start_frame
                - self.post_frame
                - self.pre_post_omission
            )
        else:
            self.img_per_movie = self.end_frame + 1 - self.start_frame

        self.total_frame_per_movie = int(self.raw_data.size / self.nb_probes)

        average_nb_samples = 200000

        shape = (self.total_frame_per_movie, int(self.nb_probes / 2), 2)
        # load it with the correct shape
        self.raw_data = np.memmap(
            self.raw_data_file, dtype="int16", shape=shape)

        # Older reshape code, to remove when stable
        # Reshape in number of traces
        # self.raw_data = np.reshape(self.raw_data
        #   , (self.total_frame_per_movie,
        #   self.nb_probes))

        # Reshape following probes location
        # self.raw_data = np.reshape(self.raw_data, (self.total_frame_per_movie
        #                                           int(self.nb_probes/2), 2)

        local_data = self.raw_data[0:average_nb_samples, :, :].flatten()
        local_data = local_data.astype("float32")
        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)
        self.epoch_index = 0
        self.list_samples = np.arange(
            self.start_frame, self.start_frame + self.img_per_movie
        )
        if "randomize" in self.json_data.keys():
            if self.json_data["randomize"] == 1:
                np.random.shuffle(self.list_samples)

        # We cut the number of samples if asked to
        if (self.total_samples > 0
                and self.total_samples < len(self.list_samples)):
            self.list_samples = self.list_samples[0: self.total_samples]

    def __len__(self):
        "Denotes the total number of batches"
        return int(np.floor(float(len(self.list_samples)) / self.batch_size))

    def on_epoch_end(self):
        # We only increase index if steps_per_epoch
        # is set to positive value. -1 will force the generator
        # to not iterate at the end of each epoch
        if self.steps_per_epoch > 0:
            if self.steps_per_epoch * (self.epoch_index + 2) < self.__len__():
                self.epoch_index = self.epoch_index + 1
            else:
                # if we reach the end of the data, we roll over
                self.epoch_index = 0

    def __getitem__(self, index):
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        if (index + 1) * self.batch_size > self.total_frame_per_movie:
            indexes = np.arange(index * self.batch_size, self.img_per_movie)
        else:
            indexes = np.arange(index * self.batch_size,
                                (index + 1) * self.batch_size)

        shuffle_indexes = self.list_samples[indexes]
        input_full = np.zeros(
            [self.batch_size, int(self.nb_probes), 2,
             self.pre_frame + self.post_frame],
            dtype="float32",
        )
        output_full = np.zeros(
            [self.batch_size, int(self.nb_probes), 2, 1], dtype="float32"
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
            [1, self.nb_probes, 2,
             self.pre_frame + self.post_frame], dtype="float32"
        )
        output_full = np.zeros([1, self.nb_probes, 2, 1], dtype="float32")

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        input_index = input_index[input_index != index_frame]

        for index_padding in np.arange(self.pre_post_omission + 1):
            input_index = input_index[input_index !=
                                      index_frame - index_padding]
            input_index = input_index[input_index !=
                                      index_frame + index_padding]

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


class MultiContinuousTifGenerator(DeepGenerator):
    """This generator is used when dealing with a continuous movie split
    in multiple blocks of frames each within individual tif files as is
    typically generated by ScanImage.
    The filenames will be sorted alphabetically to ensure continuity"""

    def __init__(self, json_path):
        "Initialization"
        super().__init__(json_path)

        # For backward compatibility
        if "train_path" in self.json_data.keys():
            self.raw_data_file = self.json_data["train_path"]
        else:
            self.raw_data_file = self.json_data["movie_path"]

        self.batch_size = self.json_data["batch_size"]
        self.pre_frame = self.json_data["pre_frame"]
        self.post_frame = self.json_data["post_frame"]

        # For backward compatibility
        if "pre_post_omission" in self.json_data.keys():
            self.pre_post_omission = self.json_data["pre_post_omission"]
        else:
            self.pre_post_omission = 0

        self.start_frame = self.json_data["start_frame"]
        self.steps_per_epoch = self.json_data["steps_per_epoch"]

        if "randomize" in self.json_data.keys():
            self.randomize = self.json_data["randomize"]
        else:
            self.randomize = 1

        # This is used to limit the total number of samples
        # -1 means to take all and is the default fall back
        if "total_samples" in self.json_data.keys():
            self.total_samples = self.json_data["total_samples"]
        else:
            self.total_samples = -1

        # This is compatible with negative frames
        self.end_frame = self.json_data["end_frame"]

        self.list_tif_files = glob.glob(
            os.path.join(self.raw_data_file, '*.tif'))

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

        if self.end_frame < 0:
            self.img_per_movie = (
                self.total_frame_per_movie + 1
                + self.end_frame - self.start_frame
            )
        elif self.total_frame_per_movie < self.end_frame:
            self.img_per_movie = self.total_frame_per_movie
            + 1 - self.start_frame
        else:
            self.img_per_movie = self.end_frame + 1 - self.start_frame

        average_nb_samples = 1000

        local_data = self.get_raw_frames_from_list(
            np.arange(0, average_nb_samples))
        local_data = local_data.astype("float32").flatten()
        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)
        self.epoch_index = 0

        self.list_samples = np.arange(
            self.pre_frame + self.pre_post_omission + self.start_frame,
            self.start_frame
            + self.img_per_movie
            - self.post_frame
            - self.pre_post_omission,
        )

        if self.randomize:
            np.random.shuffle(self.list_samples)

        # We cut the number of samples if asked to
        if (self.total_samples > 0
                and self.total_samples < len(self.list_samples)):
            self.list_samples = self.list_samples[0: self.total_samples]

    def __len__(self):
        "Denotes the total number of batches"
        return int(np.floor(float(len(self.list_samples)) / self.batch_size))

    def on_epoch_end(self):
        # We only increase index if steps_per_epoch is set
        # to positive value. -1 will force the generator
        # to not iterate at the end of each epoch
        if self.steps_per_epoch > 0:
            if self.steps_per_epoch * (self.epoch_index + 2) < self.__len__():
                self.epoch_index = self.epoch_index + 1
            else:
                # if we reach the end of the data, we roll over
                self.epoch_index = 0

    def get_list_frame_and_index(self, frame_index):
        list_index = np.where((self.list_bounds-frame_index) <= 0)[0][-1]
        start_index = self.list_bounds[list_index]
        index_in_movie = frame_index-start_index

        return list_index, index_in_movie

    def get_raw_frames_from_list(self, frame_indexes):
        if np.size(frame_indexes) == 1:
            list_index, index_in_movie = self.get_list_frame_and_index(
                frame_indexes)
            data_img_input = self.list_raw_data[list_index][
                index_in_movie, :, :]
        else:
            data_img_input = np.zeros([len(frame_indexes),
                                       self.list_raw_data[0].shape[1],
                                       self.list_raw_data[0].shape[2]])

            for index, indiv_frame in enumerate(frame_indexes):
                list_index, index_in_movie = self.get_list_frame_and_index(
                    indiv_frame)

                data_img_input[index, :, :] = self.list_raw_data[list_index][
                    index_in_movie, :, :]

        return data_img_input

    def __getitem__(self, index):
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        indexes = np.arange(index * self.batch_size,
                            (index + 1) * self.batch_size)

        shuffle_indexes = self.list_samples[indexes]

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
            [self.batch_size, 512,
             512, 1],
            dtype="float32",
        )

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        # X : (n_samples, *dim, n_channels)
        "Generates data containing batch_size samples"

        input_full = np.zeros(
            [
                1,
                512,
                512,
                self.pre_frame + self.post_frame
            ],
            dtype="float32",
        )
        output_full = np.zeros(
            [1, 512,
             512, 1], dtype="float32"
        )

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        input_index = input_index[input_index != index_frame]

        for index_padding in np.arange(self.pre_post_omission + 1):
            input_index = input_index[input_index !=
                                      index_frame - index_padding]
            input_index = input_index[input_index !=
                                      index_frame + index_padding]

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
        output_full[0, : img_out_shape[0],
                    : img_out_shape[1], 0] = data_img_output

        return input_full, output_full


class SingleTifGenerator(DeepGenerator):
    """This generator is used when dealing with a single tif file storing a 
    continous movie recording."""

    def __init__(self, json_path):
        "Initialization"
        super().__init__(json_path)

        self.raw_data_file = self.json_data["train_path"]
        self.batch_size = self.json_data["batch_size"]

        if "pre_post_frame" in self.json_data.keys():
            self.pre_frame = self.json_data["pre_post_frame"]
            self.post_frame = self.json_data["pre_post_frame"]
        else:
            self.pre_frame = self.json_data["pre_frame"]
            self.post_frame = self.json_data["post_frame"]

        self.pre_post_omission = self.json_data["pre_post_omission"]
        self.start_frame = self.json_data["start_frame"]
        self.steps_per_epoch = self.json_data["steps_per_epoch"]

        if "randomize" in self.json_data.keys():
            self.randomize = self.json_data["randomize"]
        else:
            self.randomize = 1

        # This is used to limit the total number of samples
        # -1 means to take all and is the default fall back
        if "total_samples" in self.json_data.keys():
            self.total_samples = self.json_data["total_samples"]
        else:
            self.total_samples = -1

        # This is compatible with negative frames
        self.end_frame = self.json_data["end_frame"]

        with tifffile.TiffFile(self.raw_data_file) as tif:
            self.raw_data = tif.asarray()

        self.total_frame_per_movie = self.raw_data.shape[0]

        if self.end_frame < 0:
            self.img_per_movie = (
                self.total_frame_per_movie + 1
                + self.end_frame - self.start_frame
            )
        elif self.total_frame_per_movie < self.end_frame:
            self.img_per_movie = self.total_frame_per_movie \
                + 1 - self.start_frame
        else:
            self.img_per_movie = self.end_frame + 1 - self.start_frame

        average_nb_samples = 1000

        local_data = self.raw_data[0:average_nb_samples, :, :].flatten()
        local_data = local_data.astype("float32")
        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)
        self.epoch_index = 0

        self.list_samples = np.arange(
            self.pre_frame + self.pre_post_omission + self.start_frame,
            self.start_frame
            + self.img_per_movie
            - self.post_frame
            - self.pre_post_omission,
        )

        if self.randomize:
            np.random.shuffle(self.list_samples)

        # We cut the number of samples if asked to
        if (self.total_samples > 0
                and self.total_samples < len(self.list_samples)):
            self.list_samples = self.list_samples[0: self.total_samples]

    def __len__(self):
        "Denotes the total number of batches"
        return int(np.floor(float(len(self.list_samples)) / self.batch_size))

    def on_epoch_end(self):
        # We only increase index if steps_per_epoch is set
        # to positive value. -1 will force the generator
        # to not iterate at the end of each epoch
        if self.steps_per_epoch > 0:
            if self.steps_per_epoch * (self.epoch_index + 2) < self.__len__():
                self.epoch_index = self.epoch_index + 1
            else:
                # if we reach the end of the data, we roll over
                self.epoch_index = 0

    def __getitem__(self, index):
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        indexes = np.arange(index * self.batch_size,
                            (index + 1) * self.batch_size)

        shuffle_indexes = self.list_samples[indexes]

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
            [self.batch_size, self.raw_data.shape[1],
             self.raw_data.shape[2], 1],
            dtype="float32",
        )

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        # X : (n_samples, *dim, n_channels)
        "Generates data containing batch_size samples"

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
            [1, self.raw_data.shape[1],
             self.raw_data.shape[2], 1], dtype="float32"
        )

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        input_index = input_index[input_index != index_frame]

        for index_padding in np.arange(self.pre_post_omission + 1):
            input_index = input_index[input_index !=
                                      index_frame - index_padding]
            input_index = input_index[input_index !=
                                      index_frame + index_padding]

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
        output_full[0, : img_out_shape[0],
                    : img_out_shape[1], 0] = data_img_output

        return input_full, output_full


class OphysGenerator(DeepGenerator):
    """This generator is used when dealing with a single hdf5 file storing a 
    continous movie recording into a 'data' field as [time, x, y]"""


    def __init__(self, json_path):
        "Initialization"
        super().__init__(json_path)

        if "from_s3" in self.json_data.keys():
            self.from_s3 = self.json_data["from_s3"]
        else:
            self.from_s3 = False

        # For backward compatibility
        if "train_path" in self.json_data.keys():
            self.raw_data_file = self.json_data["train_path"]
        else:
            self.raw_data_file = self.json_data["movie_path"]

        self.batch_size = self.json_data["batch_size"]

        # For backward compatibility
        if "pre_post_frame" in self.json_data.keys():
            self.pre_frame = self.json_data["pre_post_frame"]
            self.post_frame = self.json_data["pre_post_frame"]
        else:
            self.pre_frame = self.json_data["pre_frame"]
            self.post_frame = self.json_data["post_frame"]

        self.steps_per_epoch = self.json_data["steps_per_epoch"]
        self.start_frame = self.json_data["start_frame"]
        self.epoch_index = 0

        # This is compatible with negative frames
        self.end_frame = self.json_data["end_frame"]

        # This is used to limit the total number of samples
        # -1 means to take all and is the default fall back
        if "total_samples" in self.json_data.keys():
            self.total_samples = self.json_data["total_samples"]
        else:
            self.total_samples = -1

        if self.from_s3:
            s3_filesystem = s3fs.S3FileSystem()
            raw_data = h5py.File(
                s3_filesystem.open(self.raw_data_file, "rb"), "r")["data"]
        else:
            raw_data = h5py.File(self.raw_data_file, "r")["data"]

        self.total_frame_per_movie = int(raw_data.shape[0])

        if self.end_frame < 0:
            self.img_per_movie = (
                self.total_frame_per_movie
                + 1
                + self.end_frame
                - self.start_frame
                - self.post_frame
            )
        elif self.total_frame_per_movie < self.end_frame:
            self.img_per_movie = (
                self.total_frame_per_movie - self.start_frame - self.post_frame
            )
        else:
            self.img_per_movie = self.end_frame + 1 - self.start_frame

        average_nb_samples = 1000

        local_data = raw_data[0:average_nb_samples, :, :].flatten()
        local_data = local_data.astype("float32")

        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)

        self.list_samples = np.arange(
            self.start_frame, self.start_frame + self.img_per_movie
        )

        if "randomize" in self.json_data.keys():
            self.randomize = self.json_data["randomize"]
        else:
            self.randomize = 1

        if self.randomize:
            np.random.shuffle(self.list_samples)

        # We cut the number of samples if asked to
        if (self.total_samples > 0
                and self.total_samples < len(self.list_samples)):
            self.list_samples = self.list_samples[0: self.total_samples]

    def __len__(self):
        "Denotes the total number of batches"
        return int(np.floor(float(len(self.list_samples)) / self.batch_size))

    def on_epoch_end(self):
        # We only increase index if steps_per_epoch
        # is set to positive value. -1 will force the generator
        # to not iterate at the end of each epoch
        if self.steps_per_epoch > 0:
            self.epoch_index = self.epoch_index + 1

    def __getitem__(self, index):
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        if (index + 1) * self.batch_size > self.total_frame_per_movie:
            indexes = np.arange(index * self.batch_size, self.img_per_movie)
        else:
            indexes = np.arange(index * self.batch_size,
                                (index + 1) * self.batch_size)

        shuffle_indexes = self.list_samples[indexes]

        input_full = np.zeros(
            [self.batch_size, 512, 512, self.pre_frame + self.post_frame],
            dtype="float32",
        )

        output_full = np.zeros([self.batch_size, 512, 512, 1], dtype="float32")

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def __data_generation__(self, index_frame):
        "Generates data containing batch_size samples"

        if self.from_s3:
            s3_filesystem = s3fs.S3FileSystem()
            movie_obj = h5py.File(s3_filesystem.open(
                self.raw_data_file, "rb"), "r")
        else:
            movie_obj = h5py.File(self.raw_data_file, "r")

        input_full = np.zeros([1, 512, 512, self.pre_frame + self.post_frame])
        output_full = np.zeros([1, 512, 512, 1])
        input_index = np.arange(
            index_frame - self.pre_frame, index_frame + self.post_frame + 1,
        )
        input_index = input_index[input_index != index_frame]

        data_img_input = movie_obj["data"][input_index, :, :]
        data_img_output = movie_obj["data"][index_frame, :, :]

        data_img_input = np.swapaxes(data_img_input, 1, 2)
        data_img_input = np.swapaxes(data_img_input, 0, 2)

        img_in_shape = data_img_input.shape
        img_out_shape = data_img_output.shape

        data_img_input = (
            data_img_input.astype("float") - self.local_mean
        ) / self.local_std
        data_img_output = (
            data_img_output.astype("float") - self.local_mean
        ) / self.local_std

        input_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_input
        output_full[0, : img_out_shape[0],
                    : img_out_shape[1], 0] = data_img_output
        movie_obj.close()

        return input_full, output_full


class MovieJSONGenerator(DeepGenerator):
    """This generator is used when dealing with a large number of hdf5 files
    referenced into a json file with pre-computed mean and std value. The json
    file is passed to the generator in place of the movie file themselves.  
    Each individual hdf5 movie is recorded into a 'data' field 
    as [time, x, y]. The json files is pre-calculated and have the following
    fields (replace <...> appropriately): 
    {"<id>": {"path": <string path to the hdf5 file>, 
    "frames": <[int frame1, int frame2,...]>, 
    "mean": <float value>,
    "std": <float_value>}"""

    def __init__(self, json_path):
        "Initialization"
        super().__init__(json_path)

        self.sample_data_path_json = self.json_data["train_path"]
        self.batch_size = self.json_data["batch_size"]
        self.steps_per_epoch = self.json_data["steps_per_epoch"]
        self.epoch_index = 0

        # For backward compatibility
        if "pre_post_frame" in self.json_data.keys():
            self.pre_frame = self.json_data["pre_post_frame"]
            self.post_frame = self.json_data["pre_post_frame"]
        else:
            self.pre_frame = self.json_data["pre_frame"]
            self.post_frame = self.json_data["post_frame"]

        with open(self.sample_data_path_json, "r") as json_handle:
            self.frame_data_location = json.load(json_handle)

        self.lims_id = list(self.frame_data_location.keys())
        self.nb_lims = len(self.lims_id)
        self.img_per_movie = len(
            self.frame_data_location[self.lims_id[0]]["frames"])

    def __len__(self):
        "Denotes the total number of batches"
        return int(np.ceil(float(self.nb_lims
                                 * self.img_per_movie) / self.batch_size))

    def on_epoch_end(self):
        # We only increase index if steps_per_epoch
        # is set to positive value. -1 will force the generator
        # to not iterate at the end of each epoch
        if self.steps_per_epoch > 0:
            self.epoch_index = self.epoch_index + 1

    def __getitem__(self, index):
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index

        # Generate indexes of the batch
        if (index + 1) * self.batch_size > self.nb_lims * self.img_per_movie:
            indexes = np.arange(
                index * self.batch_size, self.nb_lims * self.img_per_movie
            )
        else:
            indexes = np.arange(index * self.batch_size,
                                (index + 1) * self.batch_size)

        input_full = np.zeros(
            [self.batch_size, 512, 512, self.pre_frame + self.post_frame]
        )
        output_full = np.zeros([self.batch_size, 512, 512, 1])

        for batch_index, frame_index in enumerate(indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full

    def get_lims_id_sample_from_index(self, index):
        local_img = int(np.floor(index / self.nb_lims))

        local_lims_index = int(index - self.nb_lims * local_img)

        local_lims = self.lims_id[local_lims_index]

        return local_lims, local_img

    def __get_norm_parameters__(self, index_frame):
        local_lims, local_img = self.get_lims_id_sample_from_index(index_frame)
        local_mean = self.frame_data_location[local_lims]["mean"]
        local_std = self.frame_data_location[local_lims]["std"]

        return local_mean, local_std

    def __data_generation__(self, index_frame):
        # X : (n_samples, *dim, n_channels)
        "Generates data containing batch_size samples"
        try:
            local_lims, local_img = self.get_lims_id_sample_from_index(
                index_frame)

            # Initialization
            local_path = self.frame_data_location[local_lims]["path"]

            _filenames = ["motion_corrected_video.h5", "concat_31Hz_0.h5"]
            motion_path = []
            for _filename in _filenames:
                _filepath = os.path.join(local_path, "processed", _filename)
                if os.path.exists(_filepath) and not os.path.islink(
                    _filepath
                ):  # Path exists and is not symbolic
                    motion_path = _filepath
                    break

            movie_obj = h5py.File(motion_path, "r")

            local_frame_data = self.frame_data_location[local_lims]
            output_frame = local_frame_data["frames"][local_img]
            local_mean = local_frame_data["mean"]
            local_std = local_frame_data["std"]

            input_full = np.zeros(
                [1, 512, 512, self.pre_frame + self.post_frame])
            output_full = np.zeros([1, 512, 512, 1])
            input_index = np.arange(
                output_frame - self.pre_frame, output_frame
                + self.post_frame + 1,
            )
            input_index = input_index[input_index != output_frame]

            data_img_input = movie_obj["data"][input_index, :, :]
            data_img_output = movie_obj["data"][output_frame, :, :]

            data_img_input = np.swapaxes(data_img_input, 1, 2)
            data_img_input = np.swapaxes(data_img_input, 0, 2)

            img_in_shape = data_img_input.shape
            img_out_shape = data_img_output.shape

            data_img_input = (data_img_input.astype(
                "float") - local_mean) / local_std
            data_img_output = (data_img_output.astype(
                "float") - local_mean) / local_std
            input_full[0, : img_in_shape[0],
                       : img_in_shape[1], :] = data_img_input
            output_full[0, : img_out_shape[0],
                        : img_out_shape[1], 0] = data_img_output
            movie_obj.close()

            return input_full, output_full
        except Exception:
            print("Issues with " + str(self.lims_id))
