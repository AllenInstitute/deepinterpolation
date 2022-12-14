import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
path_inference = r'/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/fMRI/studyimagenet/inference'
path_output = r"/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/fMRI/studyimagenet/output_inference"

list_inf_files = os.listdir(path_inference)
for indiv_inference_file in list_inf_files:
    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M")

    generator_param = {}
    inference_param = {}
    steps_per_epoch = 10

    generator_param["type"] = "generator"
    generator_param["name"] = "FmriGenerator"
    generator_param["pre_post_x"] = 3
    generator_param["pre_post_y"] = 3
    generator_param["pre_post_z"] = 3
    generator_param["pre_post_t"] = 2
    generator_param['center_omission_size'] = 4

    generator_param[
        "train_path"
    ] = os.path.join(path_inference, indiv_inference_file)
    generator_param["batch_size"] = 100
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = 100
    generator_param["total_nb_block"] = 10
    generator_param["steps_per_epoch"] = steps_per_epoch

    inference_param["type"] = "inference"
    inference_param["name"] = "fmri_inference"
    inference_param[
        "model_path"
    ] = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/fMRI/trained_fmri_models/fmri_volume_dense_denoiser_mean_absolute_error_2020_08_25_23_54_2020_08_25_23_54/2020_08_25_23_54_fmri_volume_dense_denoiser_mean_absolute_error_2020_08_25_23_54_model.h5"

    inference_param[
        "output_file"
    ] = os.path.join(path_output, "denoised_"+indiv_inference_file)

    jobdir = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/fMRI/studyimagenet/denoised"

    try:
        os.mkdir(jobdir)
    except:
        print("folder already exists")

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_infer = os.path.join(jobdir, "inference.json")
    json_obj = JsonSaver(inference_param)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    inference_obj = ClassLoader(path_infer)
    inference_class = inference_obj.find_and_build()(path_infer, data_generator)

    inference_class.run()
