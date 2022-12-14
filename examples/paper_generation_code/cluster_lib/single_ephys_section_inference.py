import h5py
import os
import numpy as np
import sys
import getopt
from deepinterpolation.generic import JsonSaver, ClassLoader


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # model will be trained on GPU 1


def main(argv):
    opts, args = getopt.getopt(
        argv,
        [],
        [
            "movie_path=",
            "frame_start=",
            "frame_end=",
            "output_file=",
            "model_file=",
            "batch_size=",
            "pre_post_frame=",
            "pre_post_omission=",
        ],
    )

    for opt, arg in opts:
        if opt == "--movie_path":
            movie_path = arg
        if opt == "--frame_start":
            input_frames_start = np.int(arg)
        if opt == "--frame_end":
            input_frames_end = np.int(arg)
        if opt == "--output_file":
            output_file = arg
        if opt == "--model_file":
            model_path = arg
        if opt == "--batch_size":
            batch_size = int(arg)
        if opt == "--pre_post_frame":
            pre_post_frame = int(arg)
        if opt == "--pre_post_omission":
            pre_post_omission = int(arg)

    generator_param = {}
    inference_param = {}

    generator_param["type"] = "generator"
    generator_param["name"] = "EphysGenerator"
    generator_param["pre_post_frame"] = pre_post_frame
    generator_param["pre_post_omission"] = pre_post_omission
    generator_param["batch_size"] = batch_size
    generator_param["start_frame"] = input_frames_start
    generator_param["end_frame"] = input_frames_end
    generator_param["randomize"] = 0
    generator_param["steps_per_epoch"] = -1

    generator_param["train_path"] = movie_path

    inference_param["type"] = "inference"
    inference_param["name"] = "core_inference"
    inference_param["model_path"] = model_path
    inference_param["output_file"] = output_file

    path_generator = output_file + ".generator.json"
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_infer = output_file + ".inference.json"
    json_obj = JsonSaver(inference_param)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    inference_obj = ClassLoader(path_infer)
    inference_class = inference_obj.find_and_build()(path_infer,
                                                       data_generator)

    inference_class.run()

    # to notify process is finished
    finish_file = h5py.File(output_file + ".done", "w")
    finish_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
