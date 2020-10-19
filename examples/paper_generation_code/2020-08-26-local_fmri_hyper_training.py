import deepinterpolation as de
import sys
from shutil import copyfile
import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
from typing import Any, Dict
from kerastuner.tuners import RandomSearch, BayesianOptimization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import pickle

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")

training_param = {}
generator_param = {}
network_param = {}
generator_test_param = {}

steps_per_epoch = 150

generator_test_param["type"] = "generator"
generator_test_param["name"] = "FmriGenerator"
generator_test_param["pre_post_x"] = 3
generator_test_param["pre_post_y"] = 3
generator_test_param["pre_post_z"] = 3
generator_test_param["pre_post_t"] = 2
generator_test_param['center_omission_size'] = 4

generator_test_param[
    "train_path"
] = "/home/ec2-user/fmri_data/meta_testing/sub-01:ses-perceptionTest01:func:sub-01_ses-perceptionTest01_task-perception_run-01_bold.nii.gz"
generator_test_param["batch_size"] = 1000
generator_test_param["start_frame"] = 5
generator_test_param["end_frame"] = 160
generator_test_param["total_nb_block"] = 50000
generator_test_param["steps_per_epoch"] = steps_per_epoch

# '/home/ec2-user/fmri_data/training'
local_train_path = '/home/ec2-user/fmri_data/meta_training'#'/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/fMRI/studyimagenet/tmp/train'
train_paths = os.listdir(local_train_path)

generator_param_list = []
for indiv_path in train_paths:

    generator_param = {}

    generator_param["type"] = "generator"
    generator_param["name"] = "FmriGenerator"
    generator_param["pre_post_x"] = 3
    generator_param["pre_post_y"] = 3
    generator_param["pre_post_z"] = 3
    generator_param["pre_post_t"] = 2
    generator_param["train_path"] = os.path.join(local_train_path, indiv_path)
    generator_param["batch_size"] = 1000
    generator_param["start_frame"] = 5
    generator_param["end_frame"] = 160
    generator_param["total_nb_block"] = 150000
    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["center_omission_size"] = 4
    generator_param_list.append(generator_param)

network_param["type"] = "network"
network_param["name"] = "fmri_flexible_architecture"

training_param["type"] = "trainer"
training_param["name"] = "core_trainer"
training_param["run_uid"] = run_uid
training_param["batch_size"] = generator_test_param["batch_size"]
training_param["steps_per_epoch"] = steps_per_epoch
training_param["period_save"] = 1000
training_param["nb_gpus"] = 0
training_param["apply_learning_decay"] = 0
training_param["initial_learning_rate"] = 0.0001
training_param["epochs_drop"] = 50
training_param["nb_times_through_data"] = 1
training_param["learning_rate"] = 0.0001

training_param["loss"] = "mean_absolute_error"
training_param["model_string"] = (
    network_param["name"]
    + "_"
    + training_param["loss"]
    + "_"
    + training_param["run_uid"]
)

jobdir = (
    "/home/ec2-user/trained_fmri_models/"    
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
    if local_index == 0:
        indiv_generator["initialize_list"] = 1
    else:
        indiv_generator["initialize_list"] = 0

    path_generator = os.path.join(
        jobdir, "generator" + str(local_index) + ".json")
    json_obj = JsonSaver(indiv_generator)
    json_obj.save_json(path_generator)
    generator_obj = ClassLoader(path_generator)
    train_generator = generator_obj.find_and_build()(path_generator)

    # we don't need to set a random set of points for all 100 or so
    if local_index == 0:
        keep_generator = train_generator
    else:
        train_generator.x_list = keep_generator.x_list
        train_generator.y_list = keep_generator.y_list
        train_generator.z_list = keep_generator.z_list
        train_generator.t_list = keep_generator.t_list

    list_train_generator.append(train_generator)


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

global_train_generator = de.generator_collection.CollectorGenerator(
    list_train_generator
)

test_generator = generator_test_obj.find_and_build()(path_test_generator)

network_callback = network_obj.find_and_build()(path_network)

# We initialize the trainer as usual except without compiling
training_class = trainer_obj.find_and_build()(
    global_train_generator, test_generator, network_callback, path_training, auto_compile=False
)

# We build the hyperparameter training class


def build_model(hp):

    # We allow learning rate to change
    training_class.optimizer = RMSprop(
        lr=hp.Choice('learning_rate', [1e-3, 1e-4]))  # , 1e-4, 1e-5]))

    local_size = training_class.local_generator.get_input_size()

    input_img = Input(shape=local_size)
    training_class.local_model = Model(
        input_img, training_class.network_obj(input_img, hp))

    training_class.compile()

    return training_class.local_model


# This is where we set the searching strategy
tuner = BayesianOptimization(
    build_model,
    objective='val_mae',
    seed=40,
    max_trials=500,
    executions_per_trial=1,
    directory=jobdir)

tuner.search_space_summary()

training_class.cache_validation()

# replacement for model.fit
tuner.search(training_class.local_generator,
             validation_data=training_class.local_test_generator,
             steps_per_epoch=training_class.steps_per_epoch,
             epochs=training_class.epochs,
             max_queue_size=4,  # 32,
             workers=training_class.workers,
             shuffle=False,
             use_multiprocessing=True,
             callbacks=training_class.callbacks_list,
             initial_epoch=0,)

tuner.results_summary()
pickle.dump(tuner, open(os.path.join(jobdir,"result.pkl"),"wb"))

