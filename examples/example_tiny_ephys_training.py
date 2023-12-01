import os
from deepinterpolation.generator_collection import EphysGenerator
from deepinterpolation.trainor_collection import core_trainer
from deepinterpolation.network_collection import unet_single_ephys_1024
import datetime
import pathlib

if __name__ == '__main__':
    # This is used for record-keeping
    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M")

    # Initialize meta-parameters objects
    training_param = {}
    generator_param = {}
    network_param = {}
    generator_test_param = {}

    # An epoch is defined as the number of batches pulled from the dataset.
    # Because our datasets are VERY large. Often, we cannot
    # go through the entirity of the data so we define an epoch slightly
    # differently than is usual.
    steps_per_epoch = 10

    # Those are parameters used for the Validation test generator.
    # Here the test is done on the beginning of the data but
    # this can be a separate file
    generator_test_param[
        "pre_post_frame"
    ] = 30  # Number of frame provided before and after the predicted frame
    generator_test_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )
    generator_test_param["batch_size"] = 100
    generator_test_param["start_frame"] = 0
    generator_test_param["end_frame"] = 1999
    generator_test_param[
        "pre_post_omission"
    ] = 1  # Number of frame omitted before and after the predicted frame
    generator_test_param[
        "steps_per_epoch"
    ] = -1
    # No step necessary for testing as epochs are not relevant.
    # -1 deactivate it.

    # Those are parameters used for the main data generator
    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["pre_post_frame"] = 30
    generator_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )
    generator_param["batch_size"] = 100
    generator_param["start_frame"] = 2000
    generator_param["end_frame"] = 7099
    generator_param["pre_post_omission"] = 1

    # Those are parameters used for the training process
    training_param["run_uid"] = run_uid
    training_param["batch_size"] = generator_test_param["batch_size"]
    training_param["steps_per_epoch"] = steps_per_epoch
    training_param[
        "period_save"
    ] = 25
    # network model is potentially saved during training
    # between a regular nb epochs
    training_param["nb_gpus"] = 0
    training_param["apply_learning_decay"] = 0
    training_param[
        "nb_times_through_data"
    ] = 1
    # if you want to cycle through the entire data.
    # Two many iterations will cause noise overfitting
    training_param["learning_rate"] = 0.0001
    training_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
    training_param["loss"] = "mean_absolute_error"
    training_param[
        "nb_workers"
    ] = 16
    # this is to enable multiple threads for data generator loading.
    # Useful when this is slower than training

    training_param["model_string"] = (
        "unet_single_ephys_1024_"
        + training_param["loss"]
        + "_"
        + training_param["run_uid"]
    )

    # Where do you store ongoing training progress
    jobdir = os.path.join(
        ".", training_param["model_string"] + "_" + run_uid,
    )
    training_param["output_dir"] = jobdir

    try:
        os.mkdir(jobdir)
    except Exception:
        print("folder already exists")

    # We find the generator obj in the collection using the json file
    train_generator = EphysGenerator(generator_param)
    test_generator = EphysGenerator(generator_test_param)

    # We build the training object.
    training_class = core_trainer(
        train_generator, test_generator, unet_single_ephys_1024({}), training_param
    )

    # Start training. This can take very long time.
    training_class.run()

    # Finalize and save output of the training.
    training_class.finalize()
