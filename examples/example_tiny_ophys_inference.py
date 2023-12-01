import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import pathlib

if __name__ == '__main__':
    generator_param = {}
    inference_param = {}

    # We are reusing the data generator for training here.
    generator_param["type"] = "generator"
    generator_param["name"] = "SingleTifGenerator"
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

    inference_param["type"] = "inference"
    inference_param["name"] = "core_inference"

    # Replace this path to where you stored your model
    inference_param[
        "model_path"
    ] = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects\
        /Deep2P/repos/public/deepinterpolation/examples/unet_single_1024_\
        mean_absolute_error_2020_11_12_21_33_2020_11_12_21_33/2020_11_\
        12_21_33_unet_single_1024_mean_absolute_error_2020_11_12_21_33_\
        model.h5"

    # Replace this path to where you want to store your output file
    inference_param[
        "output_file"
    ] = "/Users/jeromel/test/ophys_tiny_continuous_deep_interpolation.h5"

    jobdir = "/Users/jeromel/test/"

    try:
        os.mkdir(jobdir)
    except Exception:
        print("folder already exists")

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_infer = os.path.join(jobdir, "inference.json")
    json_obj = JsonSaver(inference_param)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    inference_obj = ClassLoader(path_infer)
    inference_class = inference_obj.find_and_build()(path_infer,
                                                       data_generator)

    # Except this to be slow on a laptop without GPU. Inference needs
    # parallelization to be effective.
    inference_class.run()
