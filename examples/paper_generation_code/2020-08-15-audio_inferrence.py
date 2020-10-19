import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")

generator_param = {}
inferrence_param = {}
steps_per_epoch = 10

generator_param["type"] = "generator"
generator_param["name"] = "WavGenerator"
generator_param["pre_post_frame"] = 5000
generator_param['limit_size'] = 0
generator_param["pre_post_hole"] = 50


generator_param[
    "train_path"
] = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/Weekly Analysis/08-15-2020/testing_sounds/36105__erh__roswell.wav"
generator_param["batch_size"] = 1000
generator_param["start_frame"] = 0
generator_param["end_frame"] = 3700000
generator_param["steps_per_epoch"] = steps_per_epoch

inferrence_param["type"] = "inferrence"
inferrence_param["name"] = "audio_inferrence"
inferrence_param["max_frame"] = 920000
inferrence_param[
    "model_path"
] = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/Weekly Analysis/08-15-2020/sound_denoiser_mean_absolute_error_2020_08_17_07_18_2020_08_17_07_18/2020_08_17_07_18_sound_denoiser_mean_absolute_error_2020_08_17_07_18-0006-0.4910.h5"

inferrence_param[
    "output_file"
] = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/Weekly Analysis/08-15-2020/denoised/36105__erh__roswell_v1.wav"

jobdir = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/Weekly Analysis/08-15-2020/denoised/"

try:
    os.mkdir(jobdir)
except:
    print("folder already exists")

path_generator = os.path.join(jobdir, "generator.json")
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

path_infer = os.path.join(jobdir, "inferrence.json")
json_obj = JsonSaver(inferrence_param)
json_obj.save_json(path_infer)

generator_obj = ClassLoader(path_generator)
data_generator = generator_obj.find_and_build()(path_generator)

inferrence_obj = ClassLoader(path_infer)
inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator)


inferrence_class.run()
