import os
import pathlib
from deepinterpolation.cli.fine_tuning import FineTuning

if __name__ == '__main__':
    # Initialize meta-parameters objects
    finetuning_params = {}
    generator_param = {}
    generator_test_param = {}

    # Those are parameters used for the Validation test generator.
    # Here the test is done on the beginning of the data but
    # this can be a separate file
    generator_test_param["name"] = "EphysGenerator"  # Name of object
    # in the collection
    generator_test_param["pre_frame"] = 30
    generator_test_param["post_frame"] = 30
    generator_test_param["data_path"] = os.path.join(
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

    # Those are parameters used for the main data generator
    generator_param["name"] = "EphysGenerator"
    generator_param["pre_frame"] = 30
    generator_param["post_frame"] = 30
    generator_param["data_path"] = os.path.join(
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
    finetuning_params["name"] = "transfer_trainer"

    # Change this path to any model you wish to improve
    filename = \
        "2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5"
    local_path = \
        os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            "..",
            "sample_data",
            filename
        )
    finetuning_params["model_source"] = {
        "local_path": local_path
    }

    # An epoch is defined as the number of batches pulled from the dataset.
    # Because our datasets are VERY large. Often, we cannot
    # go through the entirety of the data so we define an epoch
    # slightly differently than is usual.
    steps_per_epoch = 10
    finetuning_params["steps_per_epoch"] = steps_per_epoch
    finetuning_params[
        "period_save"
    ] = 25
    # network model is potentially saved during training between a regular
    # nb epochs

    finetuning_params["learning_rate"] = 0.0001
    finetuning_params["loss"] = "mean_squared_error"
    finetuning_params["output_dir"] = '/Users/jeromel/Desktop/test/'

    args = {
        "finetuning_params": finetuning_params,
        "generator_params": generator_param,
        "test_generator_params": generator_test_param,
        "output_full_args": True
    }

    finetuning_obj = FineTuning(input_data=args, args=[])
    finetuning_obj.run()
