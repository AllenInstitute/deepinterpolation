import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")

generator_param = {}
inferrence_param = {}
steps_per_epoch = 10

generator_param["type"] = "generator"
generator_param["name"] = "FmriGenerator"
generator_param["pre_post_x"] = 3
generator_param["pre_post_y"] = 3
generator_param["pre_post_z"] = 3
generator_param["pre_post_t"] = 1

generator_param[
    "train_path"
] = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/fMRI/studyimagenet/derivatives-preproc-spm-output-sub-02-ses-perceptionTraining01-func-sub-02_ses-perceptionTraining01_task-perception_run-01_bold_preproc.nii"
generator_param["batch_size"] = 100
generator_param["start_frame"] = 0
generator_param["end_frame"] = 100
generator_param["total_nb_block"] = 10
generator_param["steps_per_epoch"] = steps_per_epoch


inferrence_param["type"] = "inferrence"
inferrence_param["name"] = "fmri_inferrence"
inferrence_param[
    "model_path"
] = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/fMRI/trained_fmri_models/fmri_volume_dense_denoiser_mean_absolute_error_2020_08_08_01_05_2020_08_08_01_05/2020_08_08_01_05_fmri_volume_dense_denoiser_mean_absolute_error_2020_08_08_01_05-1640-0.0474.h5"

inferrence_param[
    "output_file"
] = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/fMRI/studyimagenet/denoised/fmri_volume_denoiser_mean_absolute_error_task_full_7.h5"

jobdir = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/fMRI/studyimagenet/denoised"

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
