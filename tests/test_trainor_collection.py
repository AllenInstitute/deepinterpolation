import os
import pathlib
from deepinterpolation.generator_collection import EphysGenerator
from deepinterpolation.trainor_collection import core_trainer
from deepinterpolation.network_collection import unet_single_ephys_1024

def test_ephys_training(tmp_path):

    # Initialize meta-parameters objects
    training_param = {}
    generator_param = {}
    network_param = {}
    generator_test_param = {}

    steps_per_epoch = 2

    generator_test_param[
        "pre_post_frame"
    ] = 30  # Number of frame provided before and after the predicted frame
    generator_test_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )
    generator_test_param["batch_size"] = 10
    generator_test_param["start_frame"] = 30
    generator_test_param["end_frame"] = 60
    generator_test_param[
        "pre_post_omission"
    ] = 1  # Number of frame omitted before and after the predicted frame
    generator_test_param[
        "steps_per_epoch"
    ] = -1

    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["pre_post_frame"] = 30
    generator_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )
    generator_param["batch_size"] = 10
    generator_param["start_frame"] = 1030
    generator_param["end_frame"] = 1060
    generator_param["pre_post_omission"] = 1

    # Those are parameters used for the network topology
    network_param["type"] = "network"
    network_param[
        "name"
    ] = "unet_single_ephys_1024"  # Name of network topology in the collection

    # Those are parameters used for the training process
    training_param["run_uid"] = 'tmp'
    training_param["batch_size"] = generator_test_param["batch_size"]
    training_param["steps_per_epoch"] = steps_per_epoch
    training_param[
        "period_save"
    ] = 25
    training_param["nb_gpus"] = 0
    training_param["apply_learning_decay"] = 0
    training_param[
        "nb_times_through_data"
    ] = 1
    training_param["learning_rate"] = 0.0001
    training_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
    training_param["loss"] = "mean_absolute_error"
    training_param[
        "nb_workers"
    ] = 1

    training_param["model_string"] = (
        network_param["name"]
        + "-"
        + training_param["loss"]
    )
    jobdir = tmp_path

    training_param["output_dir"] = os.fspath(jobdir)

    network_obj = unet_single_ephys_1024({})
    train_generator = EphysGenerator(generator_param)
    test_generator = EphysGenerator(generator_test_param)

    training_class = core_trainer(train_generator, test_generator, 
                                  network_obj, training_param)

    training_class.run()

    # Finalize and save output of the training.
    training_class.finalize()

    # Validation is a bit random due to initilization. We check that you get
    # reasonable number
    assert training_class.model_train.history["val_loss"][-1] < 10
