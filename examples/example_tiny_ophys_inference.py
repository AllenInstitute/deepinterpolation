import os
from deepinterpolation.generator_collection import SingleTifGenerator
from deepinterpolation.inference_collection import core_inference
import pathlib

if __name__ == '__main__':
    generator_param = {}
    inference_param = {}

    # We are reusing the data generator for training here.
    generator_param["pre_post_frame"] = 30
    generator_param["pre_post_omission"] = 0
    generator_param[
        "steps_per_epoch"
    ] = -1
    # No steps necessary for inference as epochs are not relevant.
    # -1 deactivate it.

    generator_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ophys_tiny_761605196.tif",
    )

    generator_param["batch_size"] = 5
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = 99  # -1 to go until the end.
    generator_param[
        "randomize"
    ] = 0
    # This is important to keep the order
    # and avoid the randomization used during training

    # Replace this path to where you stored your model
    inference_param[
        "model_path"
    ] = r"/Users/jerome.lecoq/Dropbox/DO NOT DELETE/Deepinterpolation-" \
        + r"Models/deep_interpolation_ai93_v1_1/2019_09_11_23_32_unet_" \
        + r"single_1024_mean_absolute_error_Ai93-0450.h5"

    # Replace this path to where you want to store your output file
    inference_param[
        "output_file"
    ] = "./ophys_tiny_continuous_deep_interpolation.h5"

    jobdir = "./"

    try:
        os.mkdir(jobdir)
    except Exception:
        print("folder already exists")

    data_generator = SingleTifGenerator(generator_param)

    inference_class = core_inference(inference_param, data_generator)

    # Except this to be slow on a laptop without GPU. Inference needs
    # parallelization to be effective.
    inference_class.run()
