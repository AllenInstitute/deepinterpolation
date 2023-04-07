import warnings

import h5py
import logging
import numpy as np
from deepinterpolation.generic import JsonLoader
from deepinterpolation.generator_collection import DeepGenerator
from tensorflow.keras.models import load_model
import deepinterpolation.loss_collection as lc
from tqdm.auto import tqdm
import tensorflow as tf
from typing import Union
import multiprocessing
from multiprocessing.managers import DictProxy, AcquirerProxy
from deepinterpolation.multiprocessing_utils import winnow_process_list
from pathlib import Path
tf.compat.v1.logging.set_verbosity(
    tf.compat.v1.logging.ERROR)

logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)
class fmri_inferrence:
    # This inferrence is specific to fMRI which is raster scanning for
    # denoising

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

        if "output_datatype" in self.json_data.keys():
            self.output_datatype = self.json_data["output_datatype"]
        else:
            self.output_datatype = 'float32'

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
                dtype=self.output_datatype,
            )
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
                dtype=self.output_datatype,
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

def _load_model(json_data: dict) -> tf.keras.Model:
    try:
        local_model_path = __get_local_model_path(json_data)
        model = __load_local_model(path=local_model_path)
    except KeyError:
        model = __load_model_from_mlflow(json_data)
    return model

def _rescale(data: np.ndarray,
             std: float,
             mean: float,
    ) -> np.ndarray:
    """Rescale the inference output. Since the inference is
    predicting normalized data, rescaling can denormalize
    
    Parameters
    -----------
    data : np.ndarray
        output from inference
    std : float
        standard deviation calculated from the movie prior
        to denoising
    mean : float
        mean calculated from the movie prior to denoising

    Returns
    -------
    np.ndarray : rescaled movie
    """
    return data * std + mean


def core_inference_worker(
                        json_data: dict,
                        input_lookup: dict,
                        rescale: bool,
                        save_raw: bool,
                        output_dict: DictProxy,
                        output_lock: AcquirerProxy):

    """ Helper function to define Worker for processing inference.
        Used with multiprocessing.process 
    """
    model = _load_model(json_data)
    local_output = {}
    for dataset_index in input_lookup:
        local_lookup = input_lookup[dataset_index]
        local_data = local_lookup['local_data']
        local_mean = local_lookup['local_mean']
        local_std = local_lookup['local_std']

        predictions_data = model.predict_on_batch(local_data[0])

        if rescale:
            corrected_data = _rescale(predictions_data, local_std, local_mean)
        else:
            corrected_data = predictions_data

        corrected_raw = None
        if save_raw:
            if rescale:
                corrected_raw = _rescale(local_data[1], local_std, local_mean)
            else:
                corrected_raw = local_data[1]

        local_output[dataset_index] = {'corrected_raw': corrected_raw,
                                       'corrected_data': corrected_data}

    with output_lock:
        k_list = list(local_output.keys())
        for k in k_list:
            output_dict[k] = local_output.pop(k)

def __get_local_model_path(json_data: dict) -> str:
    try:
        model_path = json_data['model_path']
        warnings.warn('Loading model from model_path will be deprecated '
                      'in a future release')
    except KeyError:
        model_path = json_data['model_source']['local_path']
    return model_path

def __load_local_model(path: str) -> tf.keras.Model:
    model = load_model(
        path,
        custom_objects={
            "annealed_loss": lc.loss_selector("annealed_loss")},
    )
    return model

def __load_model_from_mlflow(json_data: dict) -> tf.keras.Model:
    import mlflow
    mlflow_registry_params = \
        json_data['model_source']['mlflow_registry']

    model_name = mlflow_registry_params['model_name']
    model_version = mlflow_registry_params.get('model_version')
    model_stage = mlflow_registry_params.get('model_stage')

    mlflow.set_tracking_uri(mlflow_registry_params['tracking_uri'])

    if model_version is not None:
        model_uri = f"models:/{model_name}/{model_version}"
    elif model_stage:
        model_uri = f"models:/{model_name}/{model_stage}"
    else:
        # Gets the latest version without any stage
        model_uri = f"models:/{model_name}/None"

    model = mlflow.keras.load_model(
        model_uri=model_uri
    )

    return model


class core_inferrence:
    # This is the generic inferrence class
    def __init__(self, inferrence_json_path: Union[str, Path],
                  generator_obj: DeepGenerator):
        self.model = None
        self.inferrence_json_path = inferrence_json_path
        self.generator_obj = generator_obj

        local_json_loader = JsonLoader(inferrence_json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data

        self.output_file = self.json_data["output_file"]
        self.output_dataset_name = "data"
        # The following settings are used to keep backward compatilibity
        # when not using the CLI. We expect to remove when all uses
        # are migrated to the CLI.
        if "save_raw" in self.json_data.keys():
            self.save_raw = self.json_data["save_raw"]
            self.raw_dataset_name = "raw"
        else:
            self.save_raw = False
            self.raw_dataset_name = None

        if "rescale" in self.json_data.keys():
            self.rescale = self.json_data["rescale"]
        else:
            self.rescale = True

        if "output_datatype" in self.json_data.keys():
            self.output_datatype = self.json_data["output_datatype"]
        else:
            self.output_datatype = 'float32'

        if "output_padding" in self.json_data.keys():
            self.output_padding = self.json_data["output_padding"]
        else:
            self.output_padding = False

        if "nb_workers" in self.json_data.keys():
            self.workers = self.json_data["nb_workers"]
        else:
            self.workers = 16

        self.batch_size = self.generator_obj.batch_size
        self.nb_datasets = len(self.generator_obj)
        self.indiv_shape = self.generator_obj.get_output_size()
        self._create_h5_datasets(self.output_dataset_name, self.raw_dataset_name)


    def _get_first_output_index(self) -> int:
        """
        Get first output index depending if output_padding is enabled.
        If output padding is enabled, the output is padded with
        blank frames such that it's the same size as the original movie.
        Hence the first output index is equal pre_frame + pre_post_omission

        Returns
        -------
        first_output_index: int
        """
        if self.output_padding:
            first_output_index = self.generator_obj.start_sample - \
                self.generator_obj.start_frame
        else:
            first_output_index = 0
        return first_output_index


    def _create_h5_datasets(self,
                            output_dataset_name: str,
                            raw_dataset_name: str,
                            ):
        """Create datasets on output h5 file.

        Parameters
        ----------
        output_dataset_name : str
            name of the key of the output dataset stored in h5
        raw_dataset_name : str
            name of the key of the raw dataset stored in h5
        """
        nb_frames_inference = self.generator_obj.list_samples.shape[0]
        if self.output_padding:
            final_shape = [nb_frames_inference + self._get_first_output_index()]

        else:
            final_shape = [nb_frames_inference]

        final_shape.extend(self.indiv_shape[:-1])
        chunk_size = [1]
        chunk_size.extend(self.indiv_shape[:-1])

        with h5py.File(self.output_file, "w") as file_handle:
            file_handle.create_dataset(
                output_dataset_name,
                shape=tuple(final_shape),
                chunks=tuple(chunk_size),
                dtype=self.output_datatype,
            )
            if self.save_raw:
                file_handle.create_dataset(
                    raw_dataset_name,
                    shape=tuple(final_shape),
                    chunks=tuple(chunk_size),
                    dtype=self.output_datatype,
                )
        logger.info(f"Created empty HDF5 file {self.output_file}")


    def _write_output_to_file(self,
                              index_dataset: int,
                              corrected_data: np.ndarray,
                              corrected_raw: np.ndarray = None,
                              ):
        """Write denoised and raw outputs onto respective datasets on
        h5 file
        """
        local_size = corrected_data.shape[0]
        start = self._get_first_output_index() + index_dataset * self.batch_size
        end = start + local_size
        with h5py.File(self.output_file, "a") as file_handle:
            dset_out = file_handle[self.output_dataset_name]
            dset_out[start:end] = np.squeeze(corrected_data, -1)

            if self.save_raw and corrected_raw is not None:
                raw_out = file_handle[self.raw_dataset_name]
                raw_out[start:end] = np.squeeze(corrected_raw, -1)


    def run(self):
        self.model = _load_model(self.json_data)
        print(f"DEBUG: local mean: {self.generator_obj.local_mean}, local_std{self.generator_obj.local_std}")
        for epoch_index, index_dataset in enumerate(tqdm(np.arange(self.nb_datasets))):
            local_data = self.generator_obj[index_dataset]
            # We overwrite epoch_index to allow the last unfilled epoch
            self.generator_obj.epoch_index = epoch_index
            local_mean, local_std = \
                    self.generator_obj.__get_norm_parameters__(index_dataset)  
            predictions_data = self.model.predict_on_batch(local_data[0])
            if self.rescale:
                corrected_data = _rescale(predictions_data, local_std, local_mean)
            else:
                corrected_data = predictions_data

            if self.save_raw:
                if self.rescale:
                    corrected_raw = _rescale(local_data[1], local_std, local_mean)
                else:
                    corrected_raw = local_data[1]
            else:
                corrected_raw = None

            self._write_output_to_file(index_dataset, corrected_data, corrected_raw)


    def run_multiprocessing(self):
        with multiprocessing.Manager() as mgr:
            output_lock = mgr.Lock()
            output_dict = mgr.dict()
            process_list = []

            for epoch_index, index_dataset in enumerate(tqdm(np.arange(self.nb_datasets))):
                local_data = self.generator_obj[index_dataset]

                # We overwrite epoch_index to allow the last unfilled epoch
                self.generator_obj.epoch_index = epoch_index
                local_mean, local_std = \
                        self.generator_obj.__get_norm_parameters__(index_dataset)  
                
                this_batch = {
                    index_dataset: {
                        'local_data': local_data,
                        'local_mean': local_mean,
                        'local_std': local_std,
                    }
                }

                process = multiprocessing.Process(
                            target=core_inference_worker,
                            args=(self.json_data,
                                this_batch,
                                self.rescale,
                                self.save_raw,
                                output_dict,
                                output_lock))
                process.start()
                process_list.append(process)

                while len(process_list) >= self.workers:
                    process_list = winnow_process_list(process_list)


                if len(output_dict) >= max(1, self.nb_datasets//8):
                    with output_lock:
                        index_list = list(output_dict.keys())
                        for dataset_index in index_list:
                            dataset = output_dict.pop(dataset_index)      
                            if self.save_raw:
                                if dataset['corrected_raw'] is not None:
                                    corrected_raw = dataset['corrected_raw']
                            else:
                                corrected_raw = None
                            corrected_data = dataset['corrected_data']
                            self._write_output_to_file(dataset_index, corrected_data, corrected_raw)


            logger.info('processing last datasets')
            for p in process_list:
                p.join()

            if output_dict.keys():
                dataset_index = output_dict.keys()[0]
                dataset = output_dict[dataset_index]
                if self.save_raw:
                    if dataset['corrected_raw'] is not None:
                        corrected_raw = dataset['corrected_raw']
                else:
                    corrected_raw = None
                corrected_data = dataset['corrected_data']
                self._write_output_to_file(dataset_index, corrected_data, corrected_raw)