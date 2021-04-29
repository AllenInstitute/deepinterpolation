import deepinterpolation as de
import sys
from shutil import copyfile
import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
from typing import Any, Dict
import pathlib
import sys
import getopt
import numpy as np


def main(argv):
    opts, args = getopt.getopt(
        argv, [], ["movie_path=", "input_data_generator=", "raw_model_path=",
                   "output_path_training=", "fine_tune=",
                   'batch_size=', 'training_loss=', 'learning_rate=',
                   'steps_per_epoch=', 'save_raw=', 'pre_frame',
                   'post_frame', 'total_samples_for_training',
                   'total_samples_for_validation', 'multi_gpus',
                   'nb_times_training_through_data', 'frame_start_validation',
                   'frame_end_validation', 'frame_start_training',
                   'frame_end_training', 'frame_start_inferrence',
                   'frame_end_inferrence', 'output_file_inferrence'
                   ],
    )

    # Default values
    batch_size = 5
    save_raw = False
    fine_tune = False
    training_loss = "mean_squared_error"
    learning_rate = 0.0001
    steps_per_epoch = 200
    pre_frame = 30
    post_frame = 30
    total_samples_for_training = 50000
    total_samples_for_validation = 1000
    multi_gpus = True
    nb_times_training_through_data = 1
    frame_start_validation = 0
    frame_end_validation = 6000
    frame_start_training = 6001
    frame_end_training = -1
    frame_start_inferrence = 0
    frame_end_inferrence = -1
    output_file_inferrence =
    input_data_generator = "OphysGenerator"
    for opt, arg in opts:
        if opt == "--movie_path":
            file_h5_path = arg
        if opt == "--input_data_generator":
            input_data_generator = arg
        if opt == "--raw_model_path":
            raw_model_path = arg
        if opt == "--output_path_training":
            output_path_training = arg
        if opt == "--fine_tune":
            fine_tune = bool(arg)
        if opt == "--batch_size":
            batch_size = int(arg)
        if opt == "--training_loss":
            training_loss = arg
        if opt == "--learning_rate":
            learning_rate = float(arg)
        if opt == "--steps_per_epoch":
            steps_per_epoch = int(arg)
        if opt == "--save_raw":
            save_raw = bool(arg)
        if opt == "--pre_frame":
            pre_frame = int(arg)
        if opt == "--post_frame":
            post_frame = int(arg)
        if opt == "--total_samples_for_training":
            total_samples_for_training = int(arg)
        if opt == "--total_samples_for_validation":
            total_samples_for_validation = int(arg)
        if opt == "--multi_gpus":
            multi_gpus = bool(arg)
        if opt == "--nb_times_training_through_data":
            nb_times_training_through_data = int(arg)
        if opt == "--frame_start_validation":
            frame_start_validation = int(arg)
        if opt == "--frame_end_validation":
            frame_end_validation = int(arg)
        if opt == "--frame_start_training":
            frame_start_training = int(arg)
        if opt == "--frame_end_training":
            frame_end_training = int(arg)
        if opt == "--frame_start_inferrence":
            frame_start_inferrence = int(arg)
        if opt == "--frame_end_inferrence":
            frame_end_inferrence = int(arg)
        if opt == "--output_file_inferrence":
            output_file_inferrence = arg

    # This is used for record-keeping
    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M")

    # Initialize meta-parameters objects
    training_param = {}
    generator_param = {}
    generator_test_param = {}

    # An epoch is defined as the number of batches pulled from the dataset.
    # Because our datasets are VERY large. Often, we cannot
    # go through the entirity of the data so we define an epoch slightly
    # differently than is usual.

    generator_param["type"] = "generator"
    generator_param["name"] = input_data_generator
    generator_param["pre_frame"] = pre_frame
    generator_param["post_frame"] = post_frame
    generator_param["movie_path"] = movie_path
    generator_param["batch_size"] = batch_size
    generator_param["start_frame"] = frame_start_training
    generator_param["end_frame"] = frame_end_training
    generator_param["randomize"] = 1
    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["total_samples"] = total_samples_for_training

    generator_test_param["type"] = "generator"
    generator_test_param["name"] = input_data_generator
    generator_test_param["pre_frame"] = pre_frame
    generator_test_param["post_frame"] = post_frame
    generator_test_param["movie_path"] = movie_path
    generator_test_param["batch_size"] = batch_size
    generator_test_param["start_frame"] = frame_start_validation
    generator_test_param["end_frame"] = frame_end_validation
    generator_test_param["randomize"] = 1
    generator_test_param["steps_per_epoch"] = -1
    generator_test_param["total_samples"] = total_samples_for_validation

    # Those are parameters used for the training process
    training_param["type"] = "trainer"
    training_param["name"] = "transfer_trainer"
    training_param["run_uid"] = run_uid
    training_param["model_path"] = raw_model_path

    training_param["batch_size"] = batch_size
    training_param["steps_per_epoch"] = steps_per_epoch
    training_param[
        "period_save"
    ] = 1
    # network model is potentially saved during training between a regular
    # nb epochs

    training_param["nb_gpus"] = int(multi_gpus)
    training_param["apply_learning_decay"] = 0
    training_param[
        "nb_times_through_data"
    ] = nb_times_training_through_data
    # if you want to cycle through theentire data.
    # Two many iterations will cause noise overfitting
    training_param["learning_rate"] = learning_rate
    training_param["loss"] = training_loss
    training_param[
        "nb_workers"
    ] = 16
    # this is to enable multiple threads for data generator loading.
    # Useful when this is slower than training

    training_param["model_string"] = (
        "transfer" + "_" + training_param["loss"]
        + "_" + training_param["run_uid"]
    )

    # Where do you store ongoing training progress
    jobdir = output_path_training
    training_param["output_dir"] = jobdir

    try:
        os.mkdir(jobdir)
    except Exception as e:
        print("folder already exists")

    # Here we create all json files that are fed to the training.
    # This is used for recording purposes as well as input to the
    # training process
    path_training = os.path.join(jobdir, "training.json")
    json_obj = JsonSaver(training_param)
    json_obj.save_json(path_training)

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_test_generator = os.path.join(jobdir, "test_generator.json")
    json_obj = JsonSaver(generator_test_param)
    json_obj.save_json(path_test_generator)

    # We find the generator obj in the collection using the json file
    generator_obj = ClassLoader(path_generator)
    generator_test_obj = ClassLoader(path_test_generator)

    # We find the training obj in the collection using the json file
    trainer_obj = ClassLoader(path_training)

    # We build the generators object. This will, among other things,
    # calculate normalizing parameters.
    train_generator = generator_obj.find_and_build()(path_generator)
    test_generator = generator_test_obj.find_and_build()(path_test_generator)

    # We build the training object.
    training_class = trainer_obj.find_and_build()(
        train_generator, test_generator, path_training
    )

    # Start training. This can take very long time.
    training_class.run()

    # Finalize and save output of the training.
    training_class.finalize()

    # Now we go into inferrence
    generator_param = {}
    inferrence_param = {}

    generator_param["type"] = "generator"
    generator_param["name"] = input_data_generator
    generator_param["pre_frame"] = pre_frame
    generator_param["post_frame"] = post_frame

    # This is meant to allow compatibility
    # with a generator also used in training
    generator_param["steps_per_epoch"] = steps_per_epoch

    generator_param["batch_size"] = batch_size
    generator_param["start_frame"] = frame_start_inferrence
    generator_param["end_frame"] = frame_end_inferrence
    generator_param["movie_path"] = movie_path
    generator_param["randomize"] = 0

    inferrence_param["type"] = "inferrence"
    inferrence_param["name"] = "core_inferrence"

    if fine_tune:
        inferrence_param["model_path"] = training_class.output_model_file_path
    else:
        inferrence_param["model_path"] = raw_model_path

    inferrence_param["output_file"] = output_file_inferrence
    inferrence_param["save_raw"] = save_raw

    while NotDone:
        path_generator = output_file + ".generator.json"
        json_obj = JsonSaver(generator_param)
        json_obj.save_json(path_generator)

        path_infer = output_file + ".inferrence.json"
        json_obj = JsonSaver(inferrence_param)
        json_obj.save_json(path_infer)

        generator_obj = ClassLoader(path_generator)
        data_generator = generator_obj.find_and_build()(path_generator)

        inferrence_obj = ClassLoader(path_infer)
        inferrence_class = inferrence_obj.find_and_build()(path_infer,
                                                           data_generator)

        inferrence_class.run()


if __name__ == "__main__":
    main(sys.argv[1:])
