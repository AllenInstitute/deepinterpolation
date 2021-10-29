import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import pathlib


def test_ephys_training(tmp_path):

    # Initialize meta-parameters objects
    training_param = {}
    generator_param = {}
    network_param = {}
    generator_test_param = {}

    steps_per_epoch = 2

    generator_test_param["type"] = "generator"  # type of collection
    generator_test_param["name"] = "EphysGenerator"
    generator_test_param[
        "pre_post_frame"
    ] = 30  # Number of frame provided before and after the predicted frame
    generator_test_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )
    generator_test_param["batch_size"] = 10
    generator_test_param["start_frame"] = 0
    generator_test_param["end_frame"] = 100
    generator_test_param[
        "pre_post_omission"
    ] = 1  # Number of frame omitted before and after the predicted frame
    generator_test_param[
        "steps_per_epoch"
    ] = -1

    generator_param["type"] = "generator"
    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["name"] = "EphysGenerator"
    generator_param["pre_post_frame"] = 30
    generator_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )
    generator_param["batch_size"] = 10
    generator_param["start_frame"] = 1050
    generator_param["end_frame"] = 1200
    generator_param["pre_post_omission"] = 1

    # Those are parameters used for the network topology
    network_param["type"] = "network"
    network_param[
        "name"
    ] = "unet_single_ephys_1024"  # Name of network topology in the collection

    # Those are parameters used for the training process
    training_param["type"] = "trainer"
    training_param["name"] = "core_trainer"
    training_param["run_uid"] = 'tmp'
    training_param["batch_size"] = generator_test_param["batch_size"]
    training_param["steps_per_epoch"] = steps_per_epoch
    training_param[
        "period_save"
    ] = 25
    training_param["nb_gpus"] = 0
    training_param["apply_learning_decay"] = 0
    training_param[
        "nb_times_through_data"
    ] = 1
    training_param["learning_rate"] = 0.0001
    training_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
    training_param["loss"] = "mean_absolute_error"
    training_param[
        "nb_workers"
    ] = 1

    training_param["model_string"] = (
        network_param["name"]
        + "-"
        + training_param["loss"]
    )
    jobdir = tmp_path

    training_param["output_dir"] = os.fspath(jobdir)

    path_training = os.path.join(jobdir, "training.json")
    json_obj = JsonSaver(training_param)
    print(path_training)
    json_obj.save_json(path_training)

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_test_generator = os.path.join(jobdir, "test-generator.json")
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

    train_generator = generator_obj.find_and_build()(path_generator)
    test_generator = generator_test_obj.find_and_build()(path_test_generator)

    network_callback = network_obj.find_and_build()(path_network)

    training_class = trainer_obj.find_and_build()(
        train_generator, test_generator, network_callback, path_training
    )

    training_class.run()

    # Finalize and save output of the training.
    training_class.finalize()

    # Validation is a bit random due to initilization. We check that you get
    # reasonable number
    assert training_class.model_train.history["val_loss"][-1] < 10
