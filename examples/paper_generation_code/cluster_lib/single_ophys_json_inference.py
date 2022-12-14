import h5py
import numpy as np
import sys
import getopt
from deepinterpolation.generic import JsonSaver, ClassLoader


def main(argv):
    opts, args = getopt.getopt(
        argv,
        [],
        [
            "json_path=",
            "output_file=",
            "model_file=",
            "batch_size=",
            "pre_frame=",
            "post_frame=",
        ],
    )

    for opt, arg in opts:
        if opt == "--json_path":
            json_path = arg
        if opt == "--output_file":
            output_file = arg
        if opt == "--model_file":
            model_path = arg
        if opt == "--batch_size":
            batch_size = int(arg)
        if opt == "--pre_frame":
            pre_frame = int(arg)
        if opt == "--post_frame":
            post_frame = int(arg)

    NotDone = True

    generator_param = {}
    inference_param = {}

    generator_param["type"] = "generator"
    generator_param["name"] = "MovieJSONGenerator"
    generator_param["pre_frame"] = pre_frame
    generator_param["post_frame"] = post_frame
    generator_param["batch_size"] = batch_size
    generator_param["train_path"] = json_path

    # This parameter is not used in this context but is needed
    generator_param["steps_per_epoch"] = 10

    inference_param["type"] = "inference"
    inference_param["name"] = "core_inference"
    inference_param["model_path"] = model_path
    inference_param["output_file"] = output_file
    inference_param["save_raw"] = True
    inference_param["rescale"] = False

    while NotDone:
        path_generator = output_file + ".generator.json"
        json_obj = JsonSaver(generator_param)
        json_obj.save_json(path_generator)

        path_infer = output_file + ".inference.json"
        json_obj = JsonSaver(inference_param)
        json_obj.save_json(path_infer)

        generator_obj = ClassLoader(path_generator)
        data_generator = generator_obj.find_and_build()(path_generator)

        inference_obj = ClassLoader(path_infer)
        inference_class = inference_obj.find_and_build()(path_infer, data_generator)

        inference_class.run()
        NotDone = False

    # to notify process is finished
    finish_file = h5py.File(output_file + ".done", "w")
    finish_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
