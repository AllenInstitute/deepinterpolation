import logging
import math
import os
import warnings
from contextlib import nullcontext
from typing import List, Union, Tuple

import matplotlib.pylab as plt
import numpy as np
import tensorflow
from packaging import version
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop

import deepinterpolation.loss_collection as lc
from deepinterpolation.generic import JsonLoader


def create_decay_callback(initial_learning_rate, epochs_drop):
    """This is a helper function to return a configured
    learning rate decay callback. It uses passed parameters
    to define the behavior of the decay
    """

    def step_decay(epoch):
        """learning decay callback"""

        drop = 0.5
        decrease_pow = math.floor((1 + epoch) / epochs_drop)
        lrate = initial_learning_rate * math.pow(drop, decrease_pow)

        return lrate

    return step_decay


class core_trainer:
    # This is the generic trainer class
    # auto_compile can be set to False when doing an
    # hyperparameter search to allow modification of the model
    def __init__(
        self,
        network_obj,
        trainer_json_path,
        auto_compile=True,
    ):

        self.network_obj = network_obj

        json_obj = JsonLoader(trainer_json_path)

        # the following line is to be backward compatible in case
        # new parameter logics are added.
        json_obj.set_default("apply_learning_decay", 0)

        json_data = json_obj.json_data
        self.output_dir = json_data["output_dir"]
        self.run_uid = json_data["run_uid"]
        self.model_string = json_data["model_string"]
        self.steps_per_epoch = json_data["steps_per_epoch"]
        self.loss_type = json_data["loss"]
        self.nb_gpus = json_data["nb_gpus"]
        self.period_save = json_data["period_save"]
        self.learning_rate = json_data["learning_rate"]
        self.verbose = json_data.get("verbose", "auto")
        self.json_data = json_data
        self._logger = logging.getLogger(__name__)
        self._val_losses = []
        self._auto_compile = auto_compile
        self._validation_data = None

        if "checkpoints_dir" in json_data.keys():
            self.checkpoints_dir = json_data["checkpoints_dir"]
        else:
            self.checkpoints_dir = self.output_dir

        if "use_multiprocessing" in json_data.keys():
            self.use_multiprocessing = json_data["use_multiprocessing"]
        else:
            self.use_multiprocessing = True

        if "caching_validation" in json_data.keys():
            self.caching_validation = json_data["caching_validation"]
        else:
            self.caching_validation = False

        if "nb_workers" in json_data.keys():
            self.workers = json_data["nb_workers"]
        else:
            self.workers = 16

        # These parameters are related to setting up the
        # behavior of learning rates
        self.apply_learning_decay = json_data["apply_learning_decay"]

        if self.apply_learning_decay == 1:
            self.initial_learning_rate = json_data["initial_learning_rate"]
            self.epochs_drop = json_data["epochs_drop"]

        self.nb_times_through_data = json_data["nb_times_through_data"]

    @property
    def output_model_file_path(self):
        return os.path.join(
            self.output_dir, self.run_uid + "_" + self.model_string + "_model.h5"
        )

    def initialize_callbacks(
            self,
            model: Model,
            test_data: Union[tensorflow.keras.utils.Sequence, Tuple]
    ):

        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            self.run_uid + "_" + self.model_string + "-{epoch:04d}-{loss:.4f}.h5",
        )
        checkpoint = CustomModelCheckpoint(
            checkpoint_path,
            monitor="loss",
            verbose=self.verbose,
            save_best_only=True,
            mode="min",
            period=self.period_save,
        )

        validation_callback = ValidationCallback(
            model=model,
            trainer=self,
            model_checkpoint_callback=checkpoint,
            test_data=test_data,
            workers=self.workers,
            use_multiprocessing=self.use_multiprocessing,
            verbose=self.verbose,
            logger=self._logger
        )

        callbacks_list = [validation_callback]

        if self.apply_learning_decay == 1:
            step_decay_callback = create_decay_callback(
                self.initial_learning_rate, self.epochs_drop
            )

            lrate = LearningRateScheduler(step_decay_callback)
            callbacks_list.append(lrate)

        return callbacks_list

    def _get_n_epochs(self, data_generator):
        """Calculate number of epochs to run"""
        if self.steps_per_epoch > 0:
            epochs = self.nb_times_through_data * int(
                np.floor(len(data_generator) / self.steps_per_epoch)
            )
        else:
            epochs = self.nb_times_through_data * int(len(data_generator))
        return epochs

    def initialize_network(
            self,
            train_generator: tensorflow.keras.utils.Sequence
    ):
        local_size = train_generator.get_input_size()

        input_img = Input(shape=local_size)
        return Model(input_img, self.network_obj(input_img))

    @staticmethod
    def cache_validation(
            test_generator: tensorflow.keras.utils.Sequence):
        # This is used to remove IO duplication,
        # leverage memory for validation and
        # avoid deadlocks that happens when
        # using keras.utils.Sequence as validation datasets

        input_example = test_generator[0]
        nb_object = int(len(test_generator))

        input_shape = list(input_example[0].shape)
        nb_samples = input_shape[0]

        input_shape[0] = input_shape[0] * nb_object

        output_shape = list(input_example[1].shape)
        output_shape[0] = output_shape[0] * nb_object

        cache_input = np.zeros(shape=input_shape, dtype=input_example[0].dtype)
        cache_output = np.zeros(shape=output_shape, dtype=input_example[1].dtype)

        for local_index in range(len(test_generator)):
            local_data = test_generator[local_index]
            cache_input[
                local_index * nb_samples : (local_index + 1) * nb_samples, :
            ] = local_data[0]
            cache_output[
                local_index * nb_samples : (local_index + 1) * nb_samples, :
            ] = local_data[1]

        return cache_input, cache_output

    def run(
            self,
            train_generator: tensorflow.keras.utils.Sequence,
            test_generator: tensorflow.keras.utils.Sequence):
        # we first cache the validation data
        if self.caching_validation:
            validation_data = \
                self.cache_validation(test_generator=test_generator)
        else:
            validation_data = None

        mirrored_strategy = tensorflow.distribute.MirroredStrategy()
        with mirrored_strategy.scope() if self.nb_gpus > 1 else nullcontext():
            if self._auto_compile:
                model = self.initialize_network(
                    train_generator=train_generator)

            callbacks = self.initialize_callbacks(
                model=model,
                test_data=(validation_data if validation_data is not None
                           else test_generator)
            )

            if self._auto_compile:
                model.compile(
                    loss=lc.loss_selector(self.loss_type),
                    optimizer=RMSprop(learning_rate=self.learning_rate)
                )

        steps_per_epoch = self.steps_per_epoch if self.steps_per_epoch > 0 \
            else None

        model.fit(
            x=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self._get_n_epochs(data_generator=train_generator),
            max_queue_size=32,
            workers=self.workers,
            shuffle=False,
            use_multiprocessing=self.use_multiprocessing,
            callbacks=callbacks,
            initial_epoch=0,
            verbose=self.verbose
        )

        self._logger.info("finetuning finished - finalizing output model")
        self.finalize(
            model=model,
            train_generator=train_generator
        )

    def finalize(
            self,
            model: Model,
            train_generator: tensorflow.keras.utils.Sequence
    ):
        draw_plot = True

        if "loss" in model.history.history.keys():
            loss = model.history.history["loss"]
            # save losses

            save_loss_path = os.path.join(
                self.checkpoints_dir,
                self.run_uid + "_" + self.model_string + "_loss.npy",
            )
            np.save(save_loss_path, loss)
        else:
            self._logger.warning("Loss data was not present")
            draw_plot = False

        if self._val_losses:
            val_loss = self._val_losses

            save_val_loss_path = os.path.join(
                self.checkpoints_dir,
                self.run_uid + "_" + self.model_string + "_val_loss.npy",
            )
            np.save(save_val_loss_path, val_loss)
        else:
            self._logger.warning("Val. loss data was not present")
            draw_plot = False

        # save model
        model.save(self.output_model_file_path)

        self._logger.info("Saved model to disk")

        if draw_plot:
            h = plt.figure()
            plt.plot(loss, label="loss " + self.run_uid)
            plt.plot(val_loss, label="val_loss " + self.run_uid)

            if self.steps_per_epoch > 0:
                plt.xlabel(
                    "number of epochs ("
                    + str(train_generator.batch_size * self.steps_per_epoch)
                    + " samples/epochs)"
                )
            else:
                plt.xlabel(
                    "number of epochs ("
                    + str(train_generator.batch_size * len(train_generator))
                    + " samples/epochs)"
                )

            plt.ylabel("training loss")
            plt.legend()
            save_hist_path = os.path.join(
                self.checkpoints_dir,
                self.run_uid + "_" + self.model_string + "_losses.png",
            )
            plt.savefig(save_hist_path)
            plt.close(h)

    def update_val_loss(self, loss: float):
        """Update the val loss after calling `Model.evaluate`"""
        self._val_losses.append(loss)

    @property
    def val_loss(self) -> List[float]:
        """Validation loss"""
        return self._val_losses


# This is a helper class to fix an issue in tensorflow 2.0.
# the on_epoch_end callback from sequences is not called.
class OnEpochEnd(tensorflow.keras.callbacks.Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback()


class CustomModelCheckpoint(ModelCheckpoint):
    """This class repurposes `ModelCheckpoint` to work with model.evaluate.
    It currently only works with `model.fit`"""
    def on_test_end(self, logs=None):
        """`on_test_end` is not called in `ModelCheckpoint`.
        We define it here"""
        super().on_epoch_end(epoch=self._current_epoch, logs=logs)


class ValidationCallback(tensorflow.keras.callbacks.Callback):
    """This class defines `on_epoch_begin` and calls
    `ModelCheckpoint.on_epoch_begin` callback every epoch during validation,
    since otherwise it wouldn't be called (only called in .fit, not .evaluate)
    """
    def __init__(
            self,
            model: tensorflow.keras.Model,
            trainer: core_trainer,
            model_checkpoint_callback: CustomModelCheckpoint,
            test_data: tensorflow.keras.utils.Sequence,
            workers: int,
            use_multiprocessing: bool,
            verbose: int,
            logger: logging.Logger
    ):
        self._model = model
        self._model_checkpoint_callback = model_checkpoint_callback
        self._test_data = test_data
        self._workers = workers
        self._use_multiprocessing = use_multiprocessing
        self._verbose = verbose
        self._logger = logger
        self._trainer = trainer

    def on_epoch_begin(self, epoch, logs=None):
        self._logger.info(f'Epoch {epoch+1} train')
        self._model_checkpoint_callback.on_epoch_begin(epoch=epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self._logger.info(f'Epoch {epoch+1} validation')

        if isinstance(self._test_data, tuple):
            x, y = self._test_data
        else:
            x = self._test_data
            y = None
        loss = self._model.evaluate(
            x=x,
            y=y,
            max_queue_size=32,
            workers=self._workers,
            use_multiprocessing=self._use_multiprocessing,
            callbacks=[self._model_checkpoint_callback],
            verbose=self._verbose
        )
        self._trainer.update_val_loss(loss=loss)


class transfer_trainer(core_trainer):
    # This class is used to fine-tune a pre-trained model with additional data

    def __init__(
        self,
        trainer_json_path,
        auto_compile=True,
    ):

        super().__init__(
            network_obj=None,
            trainer_json_path=trainer_json_path,
            auto_compile=auto_compile
        )

        # For transfer learning, knowing the
        # baseline validation loss is important
        # this is expensive so we only do it when asked
        # Default is set to true here to match older behavior
        self.measure_baseline_loss = self.json_data.get(
            "measure_baseline_loss", True)

    def run(self,
            train_generator: tensorflow.keras.utils.Sequence,
            test_generator: tensorflow.keras.utils.Sequence):
        model = self.initialize_network()

        if self.measure_baseline_loss:
            self.baseline_val_loss = model.evaluate(
                x=test_generator,
                max_queue_size=32,
                workers=self.workers,
                use_multiprocessing=self.use_multiprocessing,
            )
        super().run(
            train_generator=train_generator,
            test_generator=test_generator
        )

    @property
    def output_model_file_path(self):
        return os.path.join(
            self.output_dir, self.run_uid + "_" + self.model_string +
            "_transfer_model.h5"
        )

    def initialize_network(self, **kwargs):
        return self.__load_model()

    def finalize(self, **kwargs):
        # save init losses

        if self.measure_baseline_loss:
            save_loss_path = os.path.join(
                self.checkpoints_dir,
                self.run_uid + "_" + self.model_string + "init_val_loss.npy",
            )
            np.save(save_loss_path, self.baseline_val_loss)

        super().finalize(**kwargs)

    def __load_model(self):
        try:
            local_model_path = self.__get_local_model_path()
            model = load_model(
                filepath=local_model_path,
                custom_objects={
                    "annealed_loss": lc.loss_selector("annealed_loss")},
            )
        except KeyError:
            model = self.__load_model_from_mlflow()
        return model

    def __get_local_model_path(self):
        try:
            model_path = self.json_data["model_path"]
            warnings.warn(
                "Loading model from model_path will be deprecated "
                "in a future release"
            )
        except KeyError:
            model_path = self.json_data["model_source"]["local_path"]
        return model_path

    def __load_model_from_mlflow(self):
        import mlflow

        mlflow_registry_params = self.json_data["model_source"]["mlflow_registry"]

        model_name = mlflow_registry_params["model_name"]
        model_version = mlflow_registry_params.get("model_version")
        model_stage = mlflow_registry_params.get("model_stage")

        mlflow.set_tracking_uri(mlflow_registry_params["tracking_uri"])

        if model_version is not None:
            model_uri = f"models:/{model_name}/{model_version}"
        elif model_stage:
            model_uri = f"models:/{model_name}/{model_stage}"
        else:
            # Gets the latest version without any stage
            model_uri = f"models:/{model_name}/None"

        return mlflow.keras.load_model(model_uri=model_uri)
