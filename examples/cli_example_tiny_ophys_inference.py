import os
import pathlib
from deepinterpolation.cli.inference import Inference

# Initialize meta-parameters objects
generator_param = {}
inference_param = {}

# We are reusing the data generator for training here.
generator_param["name"] = "SingleTifGenerator"
generator_param["pre_post_frame"] = 30
generator_param["pre_post_omission"] = 0

generator_param["data_path"] = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "..",
    "sample_data",
    "ophys_tiny_761605196.tif",
)
# Note the CLI has changed train_path to data_path to take into account
# the use of generators for inference

generator_param["batch_size"] = 1  # Increasing this number will facilitate
# parallelisation of inference but will cost more memory.
generator_param["start_frame"] = 0
generator_param["end_frame"] = 99  # -1 to go until the end.

# This is the name of the underlying inference class called
inference_param["name"] = "core_inferrence"

# Replace this path to where you stored your model
local_path = '/Users/jeromel/Desktop/test/\
2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5'

inference_param["model_source"] = {
    "local_path": local_path
}

# Replace this path to where you want to store your output file
inference_param[
    "output_file"
] = "/Users/jeromel/Desktop/test/ophys_tiny_continuous_deep_interpolation.h5"

# This option is to add blank frames at the onset and end of the output
# movie if some output frames are missing input frames to go through
# the model. This could be present at the start and end of the movie.
inference_param["output_padding"] = False

# this is an optional parameter to bring back output data to a given
# precision. Read the CLI documentation for more details.
# this is available through 'python -m deepinterpolation.cli.inference --help'
inference_param["output_datatype"] = 'uint16'

args = {
    "generator_params": generator_param,
    "inference_params": inference_param,
    "output_full_args": True
}

inference_obj = Inference(input_data=args, args=[])
inference_obj.run()
