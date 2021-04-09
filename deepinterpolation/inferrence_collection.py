import h5py
import numpy as np
from deepinterpolation.generic import JsonLoader
from tensorflow.keras.models import load_model
import deepinterpolation.loss_collection as lc


class fmri_inferrence:
    # This inferrence is specific to fMRI which is raster scanning
    # for denoising

    def __init__(self, inferrence_json_path, generator_obj):
        self.inferrence_json_path = inferrence_json_path
        self.generator_obj = generator_obj

        local_json_loader = JsonLoader(inferrence_json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data
        self.output_file = self.json_data["output_file"]
        self.model_path = self.json_data["model_path"]

        # This is used when output is a full volume to select only the center
        # currently only set to true. Future implementation could make smarter
        # scanning of the volume and leverage more
        # than just the center pixel
        if "single_voxel_output_single" in self.json_data.keys():
            self.single_voxel_output_single = self.json_data[
                "single_voxel_output_single"
            ]
        else:
            self.single_voxel_output_single = True

        self.model_path = self.json_data["model_path"]

        self.model = load_model(self.model_path)
        self.input_data_size = self.generator_obj.data_shape

    def run(self):
        chunk_size = list(self.generator_obj.data_shape)

        # Time is where we chunk the h5 file
        chunk_size[-1] = 1

        with h5py.File(self.output_file, "w") as file_handle:
            dset_out = file_handle.create_dataset(
                "data",
                shape=tuple(self.generator_obj.data_shape),
                chunks=tuple(chunk_size),
                dtype="float32",
            )
            # This was used to alter the volume infered and reduce
            # computation time
            # np.array([20])
            all_z_values = np.arange(0, self.input_data_size[2])
            all_y_values = np.arange(0, self.input_data_size[1])

            input_full = np.zeros(
                [
                    all_y_values.shape[0]
                    * all_z_values.shape[0]
                    * self.input_data_size[3],
                    self.generator_obj.pre_post_x * 2 + 1,
                    self.generator_obj.pre_post_y * 2 + 1,
                    self.generator_obj.pre_post_z * 2 + 1,
                    self.generator_obj.pre_post_t * 2 + 1,
                ],
                dtype="float32",
            )

            # We are looping across the volume
            for local_x in np.arange(0, self.input_data_size[0]):
                print("x=" + str(local_x))
                for index_y, local_y in enumerate(all_y_values):
                    print("y=" + str(local_y))
                    for index_z, local_z in enumerate(all_z_values):
                        for local_t in np.arange(0, self.input_data_size[3]):
                            (
                                input_full[
                                    local_t
                                    + index_z * self.input_data_size[3]
                                    + index_y
                                    * self.input_data_size[3]
                                    * all_z_values.shape[0],
                                    :,
                                    :,
                                    :,
                                    :,
                                ],
                                output_tmp,
                            ) = self.generator_obj.__data_generation__(
                                local_x, local_y, local_z, local_t
                            )

                predictions_data = self.model.predict(input_full)
                corrected_data = (
                    predictions_data * self.generator_obj.local_std
                    + self.generator_obj.local_mean
                )
                for index_y, local_y in enumerate(all_y_values):
                    for index_z, local_z in enumerate(all_z_values):
                        for local_t in np.arange(0, self.input_data_size[3]):
                            dset_out[
                                local_x, local_y, local_z, local_t
                            ] = corrected_data[
                                local_t
                                + index_z * self.input_data_size[3]
                                + index_y
                                * self.input_data_size[3]
                                * all_z_values.shape[0],
                                self.generator_obj.pre_post_x,
                                self.generator_obj.pre_post_y,
                                self.generator_obj.pre_post_z,
                                :,
                            ]


class core_inferrence:
    # This is the generic inferrence class
    def __init__(self, inferrence_json_path, generator_obj):
        self.inferrence_json_path = inferrence_json_path
        self.generator_obj = generator_obj

        local_json_loader = JsonLoader(inferrence_json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data

        self.output_file = self.json_data["output_file"]
        self.model_path = self.json_data["model_path"]

        if "save_raw" in self.json_data.keys():
            self.save_raw = self.json_data["save_raw"]
        else:
            self.save_raw = False

        if "rescale" in self.json_data.keys():
            self.rescale = self.json_data["rescale"]
        else:
            self.rescale = True

        self.batch_size = self.generator_obj.batch_size
        self.nb_datasets = len(self.generator_obj)
        self.indiv_shape = self.generator_obj.get_output_size()

        self.model = load_model(
            self.model_path,
            custom_objects={
                "annealed_loss": lc.loss_selector("annealed_loss")},
        )

    def run(self):
        final_shape = [self.nb_datasets * self.batch_size]
        final_shape.extend(self.indiv_shape)

        chunk_size = [1]
        chunk_size.extend(self.indiv_shape)

        dset_out = self.model.predict(self.generator_obj,
                                      max_queue_size=32)

        with h5py.File(self.output_file, "w") as file_handle:
            file_handle.create_dataset("data", data=dset_out)

        # min value, max value, path
        return dset_out.min(), dset_out.max(), self.output_file

        #    if self.save_raw:
        #        raw_out = file_handle.create_dataset(
        #            "raw",
        #            shape=tuple(final_shape),
        #            chunks=tuple(chunk_size),
        #            dtype="float32",
        #        )

        #    for index_dataset in np.arange(0, self.nb_datasets, 1):
        #        local_data = self.generator_obj.__getitem__(index_dataset)

        #        predictions_data = self.model.predict(local_data[0],
        #                                              workers=self.workers,
        #                                              use_multiprocessing=self.use_multiprocessing,
        #                                              max_queue_size=32)

        #        local_mean, local_std = \
        #            self.generator_obj.__get_norm_parameters__(index_dataset)
        #        local_size = predictions_data.shape[0]

        #        if self.rescale:
        #            corrected_data = predictions_data * local_std + local_mean
        #        else:
        #            corrected_data = predictions_data

        #        istart = index_dataset * self.batch_size
        #        iend = istart + local_size

        #        if self.save_raw:
        #            if self.rescale:
        #                corrected_raw = local_data[1] * local_std + local_mean
        #            else:
        #                corrected_raw = local_data[1]
        #            raw_out[istart: iend, :, ] = corrected_raw
        #        dset_out[istart: iend, :, ] = corrected_data
