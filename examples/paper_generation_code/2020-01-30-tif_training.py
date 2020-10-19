import deepinterpolation as de
import sys
from pbstools import PythonJob
from shutil import copyfile
import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
from typing import Any, Dict

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")

training_param = {}
generator_param = {}
network_param = {}
generator_test_param = {}

generator_param['type'] = 'generator'
generator_param['name'] = 'SingleTifGenerator'
generator_param['pre_post_frame'] = 30
generator_param['train_path'] = '/allen/programs/braintv/workgroups/ophysdev/OPhysCore/Natalia/denoising/run2/20191215_raw_noisy.tif'
generator_param['batch_size'] = 5
generator_param['start_frame'] = 2000
generator_param['end_frame'] = -1
generator_param['pre_post_omission'] = 0

generator_test_param['type'] = 'generator'
generator_test_param['name'] = 'SingleTifGenerator'
generator_test_param['pre_post_frame'] = 30
generator_test_param['train_path'] = '/allen/programs/braintv/workgroups/ophysdev/OPhysCore/Natalia/denoising/run2/20191215_raw_noisy.tif'
generator_test_param['batch_size'] = 5
generator_test_param['start_frame'] = 0
generator_test_param['end_frame'] = 1999
generator_test_param['pre_post_omission'] = 0

network_param['type'] = 'network'
network_param['name'] = 'padding_unet_single_1024'

training_param['type'] = 'trainer'
training_param['name'] = 'core_trainer'
training_param['run_uid'] = run_uid
training_param['batch_size'] = generator_test_param['batch_size']
training_param['steps_per_epoch'] = 200
training_param['period_save'] = 25
training_param['nb_gpus'] = 2
training_param['apply_learning_decay'] = 0
training_param['nb_times_through_data'] = 3
training_param['learning_rate'] = 0.001
training_param['pre_post_frame'] = generator_test_param['pre_post_frame']

training_param['loss'] = 'mean_absolute_error'
training_param['model_string'] = network_param['name']+'_'+training_param['loss']

jobdir = r'/allen/programs/braintv/workgroups/ophysdev/OPhysCore/Natalia/denoising/run2/trained_models/'+training_param['model_string']+'_'+run_uid

training_param['output_dir'] = jobdir

try:
    os.mkdir(jobdir)
except: 
    print('folder already exists')

path_training = os.path.join(jobdir, 'training.json')
json_obj = JsonSaver(training_param)
json_obj.save_json(path_training)

path_generator = os.path.join(jobdir, 'generator.json')
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

path_test_generator = os.path.join(jobdir, 'test_generator.json')
json_obj = JsonSaver(generator_test_param)
json_obj.save_json(path_test_generator)

path_network = os.path.join(jobdir, 'network.json')
json_obj = JsonSaver(network_param)
json_obj.save_json(path_network)

generator_obj = ClassLoader(path_generator)
generator_test_obj = ClassLoader(path_test_generator)

network_obj = ClassLoader(path_network)
trainer_obj = ClassLoader(path_training)

train_generator = generator_obj.find_and_build()(path_generator)
test_generator = generator_test_obj.find_and_build()(path_test_generator)

network_callback = network_obj.find_and_build()(path_network)

training_class = trainer_obj.find_and_build()(train_generator, test_generator,
                                              network_callback, path_training)

training_class.run()

training_class.finalize()
