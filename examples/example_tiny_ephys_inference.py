import os
from deepinterpolation.inference_collection import core_inference
from deepinterpolation.generator_collection import EphysGenerator
import pathlib

if __name__ == '__main__':
    generator_param = {}
    inference_param = {}

    # We are reusing the data generator for training here. Some parameters
    # like steps_per_epoch are irrelevant but currently needs to be provided
    generator_param["pre_post_frame"] = 30
    generator_param["pre_post_omission"] = 1
    generator_param[
        "steps_per_epoch"
    ] = -1
    # No steps necessary for inference as epochs are not relevant.
    # -1 deactivate it.

    generator_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )

    generator_param["batch_size"] = 100
    generator_param["start_frame"] = 100
    generator_param["end_frame"] = 200  # -1 to go until the end.
    generator_param[
        "randomize"
    ] = 0
    # This is important to keep the order and avoid the
    # randomization used during training

    # Replace this path to where you stored your model
    inference_param[
        "model_path"
    ] = r"./sample_data/2020_02_29_15_28_unet_single_ephys_1024_mean_" \
        + r"squared_error-1050.h5"

    # Replace this path to where you want to store your output file
    inference_param[
        "output_file"
    ] = "./ephys_tiny_continuous_deep_interpolation.h5"

    jobdir = "./"

    try:
        os.mkdir(jobdir)
    except Exception:
        print("folder already exists")

    generator_obj = EphysGenerator(generator_param)

    inference_class = core_inference(inference_param, generator_obj)

    # Except this to be slow on a laptop without GPU. Inference needs
    # parallelization to be effective.
    inference_class.run()
