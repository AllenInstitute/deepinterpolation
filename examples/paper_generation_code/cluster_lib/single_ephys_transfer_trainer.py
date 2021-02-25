import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
import getopt
import numpy as np
import sys

def main(argv):
    opts, args = getopt.getopt(
        argv,
        [],
        [
            "movie_path=",
            "train_frame_start=",
            "train_frame_end=",
            "train_total_samples=",
            "val_frame_start=",
            "val_frame_end=",
            "val_total_samples=",
            "output_path=",
            "model_file=",
            "batch_size=",
            "pre_post_frame=",
            "pre_post_omission=",
            "loss=",
        ],
    )

    # default
    train_frame_start = 20000
    train_frame_end = -1
    train_total_samples = 10000000
    val_frame_start = 0
    val_frame_end = 19999
    val_total_samples = -1
    batch_size = 100
    pre_post_frame = 30
    pre_post_omission = 1
    loss = 'mean_squared_error'

    for opt, arg in opts:
        if opt == "--movie_path":
            movie_path = arg
        if opt == "--train_frame_start":
            train_frame_start = np.int(arg)
        if opt == "--train_frame_end":
            train_frame_end = np.int(arg)
        if opt == "--train_total_samples":
            train_total_samples = np.int(arg)
        if opt == "--val_frame_start":
            val_frame_start = np.int(arg)
        if opt == "--val_frame_end":
            val_frame_end = np.int(arg)
        if opt == "--val_total_samples":
            val_total_samples = np.int(arg)
        if opt == "--batch_size":
            batch_size = np.int(arg)
        if opt == "--output_path":
            output_path = arg
        if opt == "--model_file":
            model_file = arg
        if opt == "--batch_size":
            batch_size = np.int(arg)
        if opt == "--pre_post_frame":
            pre_post_frame = np.int(arg)
        if opt == "--pre_post_omission":
            pre_post_omission = np.int(arg)
        if opt == "--loss":
            loss = arg

    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M")

    training_param = {}

    network_param = {}
    generator_test_param = {}

    generator_param = {}
    generator_param["type"] = "generator"
    generator_param["name"] = "EphysGenerator"
    generator_param["pre_post_frame"] = pre_post_frame
    generator_param["train_path"] = movie_path
    generator_param["batch_size"] = batch_size  # 100
    generator_param["start_frame"] = train_frame_start  # 20000
    generator_param["end_frame"] = train_frame_end  # -1
    generator_param["pre_post_omission"] = pre_post_omission  # 1
    generator_param["randomize"] = 1
    generator_param["steps_per_epoch"] = 100
    generator_param["total_samples"] = train_total_samples  # 10000000

    generator_test_param["type"] = "generator"
    generator_test_param["name"] = "EphysGenerator"
    generator_test_param["pre_post_frame"] = pre_post_frame
    generator_test_param[
        "train_path"
    ] = movie_path
    generator_test_param["batch_size"] = batch_size
    generator_test_param["start_frame"] = val_frame_start  # 0
    generator_test_param["end_frame"] = val_frame_end  # 19999
    generator_test_param["pre_post_omission"] = pre_post_omission
    generator_test_param["randomize"] = 1
    generator_test_param["steps_per_epoch"] = -1
    generator_test_param["total_samples"] = val_total_samples  # -1

    training_param["type"] = "trainer"
    training_param["name"] = "transfer_trainer"
    training_param[
        "model_path"
    ] = model_file

    training_param["run_uid"] = run_uid
    training_param["batch_size"] = generator_test_param["batch_size"]
    training_param["steps_per_epoch"] = generator_param["steps_per_epoch"]
    training_param["period_save"] = 25
    training_param["nb_gpus"] = 1
    training_param["nb_times_through_data"] = 1
    training_param["learning_rate"] = 0.0001
    training_param["apply_learning_decay"] = 1
    training_param["initial_learning_rate"] = 0.0001
    training_param["epochs_drop"] = 1000
    training_param["loss"] = loss
    training_param["model_string"] = network_param["name"] + \
        "_" + training_param["loss"]

    training_param["output_dir"] = output_path

    try:
        os.mkdir(output_path)
    except Exception:
        print("folder already exists")

    path_training = os.path.join(output_path, "training.json")
    json_obj = JsonSaver(training_param)
    json_obj.save_json(path_training)

    path_generator = os.path.join(
        output_path, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)
    generator_obj = ClassLoader(path_generator)
    train_generator = generator_obj.find_and_build()(path_generator)

    path_test_generator = os.path.join(output_path, "test_generator.json")
    json_obj = JsonSaver(generator_test_param)
    json_obj.save_json(path_test_generator)

    generator_test_obj = ClassLoader(path_test_generator)

    trainer_obj = ClassLoader(path_training)

    test_generator = generator_test_obj.find_and_build()(path_test_generator)

    training_class = trainer_obj.find_and_build()(
        train_generator, test_generator, path_training
    )

    training_class.run()

    training_class.finalize()

if __name__ == "__main__":
    main(sys.argv[1:])