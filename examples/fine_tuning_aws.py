import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--raw_model_path', type=str,
                        default=os.environ['SM_CHANNEL_MODEL'])
    parser.add_argument('--raw_data_path', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val_data_path', type=str,
                        default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--nb_gpus', type=str,
                        default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--output_path', type=str,
                        default=os.environ['SM_MODEL_DIR'])

    # hyperparameters sent by the client are passed as
    # command-line arguments to the script.
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--training_loss', type=str,
                        default="mean_squared_error")
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--pre_frame', type=int, default=30)
    parser.add_argument('--post_frame', type=int, default=30)
    parser.add_argument('--total_samples_for_training',
                        type=int, default=500)
    parser.add_argument('--total_samples_for_validation',
                        type=int, default=100)
    parser.add_argument('--nb_times_training_through_data',
                        type=int, default=1)
    parser.add_argument('--frame_start_validation', type=int, default=0)
    parser.add_argument('--frame_end_validation', type=int, default=6000)
    parser.add_argument('--frame_start_training', type=int, default=6001)
    parser.add_argument('--frame_end_training', type=int, default=-1)
    parser.add_argument('--input_data_generator',
                        type=str, default="OphysGenerator")

    args, _ = parser.parse_known_args()

    # Passing on the parameters
    raw_model_path = args.raw_model_path
    raw_data_path = args.raw_data_path
    val_data_path = args.val_data_path
    output_path = args.output_path
    nb_gpus = args.nb_gpus

    input_data_generator = args.input_data_generator

    batch_size = args.batch_size
    training_loss = args.training_loss
    learning_rate = args.learning_rate
    steps_per_epoch = args.steps_per_epoch
    pre_frame = args.pre_frame
    post_frame = args.post_frame
    total_samples_for_training = args.total_samples_for_training
    total_samples_for_validation = args.total_samples_for_validation
    nb_times_training_through_data = args.nb_times_training_through_data
    frame_start_validation = args.frame_start_validation
    frame_end_validation = args.frame_end_validation
    frame_start_training = args.frame_start_training
    frame_end_training = args.frame_end_training
    frame_start_inferrence = args.frame_start_inferrence
    frame_end_inferrence = args.frame_end_inferrence

    # This is used for record-keeping
    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M")

    # Initialize meta-parameters objects
    training_param = {}
    generator_param = {}
    generator_test_param = {}

    # An epoch is defined as the number of batches pulled from the dataset.
    # Because our datasets are VERY large. Often, we cannot
    # go through the entirity of the data so we define an
    # epoch slightly differently than is usual.

    generator_param["type"] = "generator"
    generator_param["name"] = input_data_generator
    generator_param["pre_frame"] = pre_frame
    generator_param["post_frame"] = post_frame
    generator_param["movie_path"] = raw_data_path
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
    generator_test_param["movie_path"] = val_data_path
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
    # network model is potentially saved during training
    # between a regular nb epochs

    training_param["nb_gpus"] = nb_gpus
    training_param["apply_learning_decay"] = 0
    training_param[
        "nb_times_through_data"
    ] = nb_times_training_through_data
    # if you want to cycle through the entire data.
    # Two many iterations will cause noise overfitting
    training_param["learning_rate"] = learning_rate
    training_param["loss"] = training_loss
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
    training_param["output_dir"] = output_path

    # Here we create all json files that are fed to the training.
    # This is used for recording purposes as well as input to the
    # training process
    path_training = os.path.join(output_path, "training.json")
    json_obj = JsonSaver(training_param)
    json_obj.save_json(path_training)

    path_generator = os.path.join(output_path, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_test_generator = os.path.join(output_path, "test_generator.json")
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
