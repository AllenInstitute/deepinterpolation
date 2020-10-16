import deepinterpolation as de
import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime


now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")

training_param = {}

generator_param1 = {}
generator_param2 = {}
generator_param3 = {}

network_param = {}
generator_test_param = {}

train_paths = [
    r"/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/767871931_415149_20181024_probeE/continuous.dat2",
    r"/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/768515987_412809_20181025_probeF/continuous.dat2",
    r"/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/771160300_415148_20181031_probeE/continuous.dat2",
]

generator_param_list = []
for indiv_path in train_paths:
    generator_param = {}
    generator_param["type"] = "generator"
    generator_param["name"] = "EphysGenerator"
    generator_param["pre_post_frame"] = 30
    generator_param["train_path"] = indiv_path
    generator_param["batch_size"] = 100
    generator_param["start_frame"] = 20000
    generator_param["end_frame"] = -1
    generator_param["pre_post_omission"] = 1
    generator_param["randomize"] = 1

    generator_param_list.append(generator_param)

generator_test_param["type"] = "generator"
generator_test_param["name"] = "EphysGenerator"
generator_test_param["pre_post_frame"] = 30
generator_test_param[
    "train_path"
] = "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/787025148_416356_20181128_probeC/continuous.dat2"
generator_test_param["batch_size"] = 100
generator_test_param["start_frame"] = 0
generator_test_param["end_frame"] = 19999
generator_test_param["pre_post_omission"] = 1
generator_test_param["randomize"] = 1

network_param["type"] = "network"
network_param["name"] = "unet_single_ephys_1024_with_dropout"

training_param["type"] = "trainer"
training_param["name"] = "core_trainer"
training_param["run_uid"] = run_uid1h
training_param["batch_size"] = generator_test_param["batch_size"]
training_param["steps_per_epoch"] = 100
training_param["period_save"] = 25
training_param["nb_gpus"] = 1
training_param["nb_times_through_data"] = 3
training_param["learning_rate"] = 0.0001
training_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
training_param["apply_learning_decay"] = 1
training_param["initial_learning_rate"] = 0.0001
training_param["epochs_drop"] = 1000
training_param["loss"] = "mean_squared_error"
training_param["model_string"] = network_param["name"] + \
    "_" + training_param["loss"]

jobdir = (
    "/allen/programs/braintv/workgroups/neuralcoding/Neuropixels_Data/neuropixels_10_sessions/778998620_419112_20181114_probeD/trained_models/"
    + training_param["model_string"]
    + "_"
    + run_uid
)

training_param["output_dir"] = jobdir

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

path_training = os.path.join(jobdir, "training.json")
json_obj = JsonSaver(training_param)
json_obj.save_json(path_training)

list_train_generator = []
for local_index, indiv_generator in enumerate(generator_param_list):
    path_generator = os.path.join(
        jobdir, "generator" + str(local_index) + ".json")
    json_obj = JsonSaver(indiv_generator)
    json_obj.save_json(path_generator)
    generator_obj = ClassLoader(path_generator)
    train_generator = generator_obj.find_and_build()(path_generator)
    list_train_generator.append(train_generator)

path_test_generator = os.path.join(jobdir, "test_generator.json")
json_obj = JsonSaver(generator_test_param)
json_obj.save_json(path_test_generator)

path_network = os.path.join(jobdir, "network.json")
json_obj = JsonSaver(network_param)
json_obj.save_json(path_network)

generator_test_obj = ClassLoader(path_test_generator)

network_obj = ClassLoader(path_network)
trainer_obj = ClassLoader(path_training)


test_generator = generator_test_obj.find_and_build()(path_test_generator)

network_callback = network_obj.find_and_build()(path_network)

global_train_generator = de.generator_collection.CollectorGenerator(
    list_train_generator
)

training_class = trainer_obj.find_and_build()(
    global_train_generator, test_generator, network_callback, path_training
)

training_class.run()

training_class.finalize()
