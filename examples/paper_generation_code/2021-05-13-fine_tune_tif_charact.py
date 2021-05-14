import sys
import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
import getopt


def main(argv):
    opts, args = getopt.getopt(
        argv, [], ["file_h5_path=", "raw_model_path=", "output_path=", ],
    )

    for opt, arg in opts:
        if opt == "--file_h5_path":
            file_h5_path = arg
        if opt == "--raw_model_path":
            raw_model_path = arg
        if opt == "--output_path":
            output_path = arg

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
    # go through the entirity of the data so we define an epoch
    # slightly differently than is usual.
    steps_per_epoch = 200

    generator_param["type"] = "generator"
    generator_param["name"] = "SingleTifGenerator"
    generator_param["pre_post_frame"] = 30
    generator_param["pre_post_omission"] = 0
    generator_param["train_path"] = file_h5_path
    generator_param["batch_size"] = 5
    generator_param["start_frame"] = 501
    generator_param["end_frame"] = -1
    generator_param["randomize"] = 1
    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["total_samples"] = -1
    generator_param["x_size"] = 512
    generator_param["y_size"] = 512

    generator_test_param["type"] = "generator"
    generator_test_param["name"] = "SingleTifGenerator"
    generator_test_param["pre_post_frame"] = 30
    generator_test_param["pre_post_omission"] = 0
    generator_test_param["train_path"] = file_h5_path
    generator_test_param["batch_size"] = 5
    generator_test_param["start_frame"] = 0
    generator_test_param["end_frame"] = 500
    generator_test_param["randomize"] = 1
    generator_test_param["steps_per_epoch"] = -1
    generator_test_param["total_samples"] = 500
    generator_test_param["x_size"] = 512
    generator_test_param["y_size"] = 512

    # Those are parameters used for the training process
    training_param["type"] = "trainer"
    training_param["name"] = "transfer_trainer"
    training_param["run_uid"] = run_uid
    training_param["model_path"] = raw_model_path

    training_param["batch_size"] = generator_test_param["batch_size"]
    training_param["steps_per_epoch"] = steps_per_epoch
    training_param[
        "period_save"
    ] = 1
    # network model is potentially saved during
    # training between a regular nb epochs
    training_param["nb_gpus"] = 1
    training_param["apply_learning_decay"] = 0
    training_param[
        "nb_times_through_data"
    ] = 3
    # if you want to cycle through the entire data.
    # Two many iterations will cause noise overfitting
    training_param["learning_rate"] = 0.0005
    training_param["loss"] = "mean_squared_error"
    training_param[
        "nb_workers"
    ] = 16
    # this is to enable multiple threads for data generator loading.
    # Useful when this is slower than training

    training_param["model_string"] = (
        "transfer" + "_" +
        training_param["loss"] + "_" + training_param["run_uid"]
    )

    # Where do you store ongoing training progress
    jobdir = output_path
    training_param["output_dir"] = jobdir

    try:
        os.mkdir(jobdir)
    except:
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


if __name__ == "__main__":
    main(sys.argv[1:])
