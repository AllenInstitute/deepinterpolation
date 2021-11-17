import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
import pathlib

if __name__ == '__main__':
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
    # go through the entirity of the data so we define an epoch slightly
    # differently than is usual.
    steps_per_epoch = 5

    # Those are parameters used for the Validation test generator. Here the
    # test is done on the beginning of the data but
    # this can be a separate file
    generator_test_param["type"] = "generator"  # type of collection
    generator_test_param["name"] = "SingleTifGenerator"
    # Name of object in the collection
    generator_test_param[
        "pre_post_frame"
    ] = 30  # Number of frame provided before and after the predicted frame
    generator_test_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ophys_tiny_761605196.tif",
    )
    generator_test_param["batch_size"] = 5
    generator_test_param["start_frame"] = 0
    generator_test_param["end_frame"] = 99
    generator_test_param[
        "pre_post_omission"
    ] = 1  # Number of frame omitted before and after the predicted frame
    generator_test_param[
        "steps_per_epoch"
    ] = -1  
    # No step necessary for testing as epochs are not relevant.
    # -1 deactivate it.

    # Those are parameters used for the main data generator
    generator_param["type"] = "generator"
    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["name"] = "SingleTifGenerator"
    generator_param["pre_post_frame"] = 30
    generator_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ophys_tiny_761605196.tif",
    )
    generator_param["batch_size"] = 5
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = 99
    generator_param["pre_post_omission"] = 0

    # Those are parameters used for the network topology
    network_param["type"] = "network"
    network_param["name"] = "unet_single_1024"  
    # Name of network topology in the collection

    # Those are parameters used for the training process
    training_param["type"] = "trainer"
    training_param["name"] = "core_trainer"
    training_param["run_uid"] = run_uid
    training_param["batch_size"] = generator_test_param["batch_size"]
    training_param["steps_per_epoch"] = steps_per_epoch
    training_param[
        "period_save"
    ] = 25  
    # network model is potentially saved
    # during training between a regular nb epochs
    training_param["nb_gpus"] = 0
    training_param["apply_learning_decay"] = 0
    training_param[
        "nb_times_through_data"
    ] = 1  
    # if you want to cycle through the entire data.
    # Two many iterations will cause noise overfitting
    training_param["learning_rate"] = 0.0001
    training_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
    training_param["loss"] = "mean_absolute_error"
    training_param[
        "nb_workers"
    ] = 16  
    # this is to enable multiple threads for data generator loading.
    # Useful when this is slower than training

    training_param["model_string"] = (
        network_param["name"]
        + "_"
        + training_param["loss"]
        + "_"
        + training_param["run_uid"]
    )

    # Where do you store ongoing training progress
    jobdir = os.path.join(
        "/Users/jeromel/test", training_param["model_string"] + "_" + run_uid,
    )
    training_param["output_dir"] = jobdir

    try:
        os.mkdir(jobdir)
    except Exception:
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

    path_network = os.path.join(jobdir, "network.json")
    json_obj = JsonSaver(network_param)
    json_obj.save_json(path_network)

    # We find the generator obj in the collection using the json file
    generator_obj = ClassLoader(path_generator)
    generator_test_obj = ClassLoader(path_test_generator)

    # We find the network obj in the collection using the json file
    network_obj = ClassLoader(path_network)

    # We find the training obj in the collection using the json file
    trainer_obj = ClassLoader(path_training)

    # We build the generators object. This will, among other things,
    # calculate normalizing parameters.
    train_generator = generator_obj.find_and_build()(path_generator)
    test_generator = generator_test_obj.find_and_build()(path_test_generator)

    # We build the network object. This will, among other things,
    # calculate normalizing parameters.
    network_callback = network_obj.find_and_build()(path_network)

    # We build the training object.
    training_class = trainer_obj.find_and_build()(
        train_generator, test_generator, network_callback, path_training
    )

    # Start training. This can take very long time.
    training_class.run()

    # Finalize and save output of the training.
    training_class.finalize()
