import deepinterpolation as de
import sys
from shutil import copyfile
import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
from typing import Any, Dict
import tensorflow

tensorflow.compat.v1.disable_eager_execution()
    
now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

training_param = {}
generator_param = {}
network_param = {}
generator_test_param = {}


input_dir = "/home/jeromel/Documents/Projects/Deep2P/input_data/"
train_path = os.path.join(
    input_dir, "2020-06-16-train-semi-large-single-plane-Ai93-norm.json"
)
input_dir = "/home/jeromel/Documents/Projects/Deep2P/input_data/"
val_path = os.path.join(
    input_dir, "2020-06-16-val-semi-large-single-plane-Ai93-norm.json"
)

steps_per_epoch = 500
generator_param["type"] = "generator"
generator_param["name"] = "MovieJSONGenerator"
generator_param["pre_frame"] = 30
generator_param["post_frame"] = 30
generator_param["batch_size"] = 5
generator_param["train_path"] = train_path
generator_param["steps_per_epoch"] = steps_per_epoch
generator_param["total_samples"] = 200000

generator_test_param["type"] = "generator"
generator_test_param["name"] = "MovieJSONGenerator"
generator_test_param["pre_frame"] = 30
generator_test_param["post_frame"] = 30
generator_test_param["batch_size"] = 5
generator_test_param["train_path"] = val_path
generator_test_param["steps_per_epoch"] = -1

network_param["type"] = "network"
network_param["name"] = "unet_1024_search"
network_param["nb_features_scale"] = 32
network_param["network_depth"] = 4
network_param["unet"] = True
network_param["feature_increase_power"] = 2

training_param["type"] = "trainer"
training_param["name"] = "core_trainer"
training_param["run_uid"] = run_uid
training_param["batch_size"] = generator_test_param["batch_size"]
training_param["steps_per_epoch"] = steps_per_epoch
training_param["period_save"] = 5
training_param["nb_gpus"] = 1
training_param["apply_learning_decay"] = 0
training_param["nb_times_through_data"] = 1
training_param["learning_rate"] = 0.0001
training_param["caching_validation"] = False

training_param["loss"] = "mean_squared_error"
training_param["model_string"] = (
    network_param["name"]
    + "_"
    + training_param["loss"]
    + "_pre_"
    + str(generator_param["pre_frame"])
    + "_post_"
    + str(generator_param["post_frame"])
    + "_feat_"
    + str(network_param["nb_features_scale"])
    + "_power_"
    + str(network_param["feature_increase_power"])
    + "_depth_"
    + str(network_param["network_depth"])
    + "_unet_"
    + str(network_param["unet"])
)

jobdir = (
    "/allen/programs/braintv/workgroups/neuralcoding/2p_data/trained_models/"
    + training_param["model_string"]
    + "_"
    + run_uid
)
training_param["tensorboard_path"] = ""


training_param["output_dir"] = jobdir

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

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

generator_obj = ClassLoader(path_generator)
generator_test_obj = ClassLoader(path_test_generator)

network_obj = ClassLoader(path_network)
trainer_obj = ClassLoader(path_training)

train_generator = generator_obj.find_and_build()(path_generator)
test_generator = generator_test_obj.find_and_build()(path_test_generator)

network_callback = network_obj.find_and_build()(path_network)

training_class = trainer_obj.find_and_build()(
    train_generator, test_generator, network_callback, path_training
)

training_class.local_model.summary()

training_class.run()

training_class.finalize()
