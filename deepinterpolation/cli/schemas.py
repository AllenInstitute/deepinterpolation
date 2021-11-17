import argschema
import marshmallow as mm
import datetime
from marshmallow import ValidationError
import inspect
from deepinterpolation import network_collection
from deepinterpolation import generator_collection
from deepinterpolation import trainor_collection
from deepinterpolation import inferrence_collection
import logging
from marshmallow.validate import OneOf


def get_list_of_networks():
    """Helper function to get the list of networks architecture available
    currently in the package.
    """
    list_architecture = inspect.getmembers(network_collection,
                                           inspect.isfunction)
    curated_list = [indiv_arch[0] for indiv_arch in list_architecture]
    excluded_list = ["Input", "dot", "load_model"]
    curated_list = [
        indiv_arch
        for i, indiv_arch in enumerate(curated_list)
        if indiv_arch not in excluded_list
    ]

    return curated_list


def get_list_of_generators():
    """Helper function to get the list of generators available
    currently in the package.
    """
    list_generator = inspect.getmembers(generator_collection, inspect.isclass)
    curated_list = [indiv_arch[0] for indiv_arch in list_generator]
    excluded_list = ["MaxRetryException", "JsonLoader",
                     "FmriGenerator", "CollectorGenerator",
                     "DeepGenerator", "SequentialGenerator"]
    # Some generators are not compatible with the CLI yet.

    curated_list = [
        indiv_arch
        for i, indiv_arch in enumerate(curated_list)
        if indiv_arch not in excluded_list
    ]

    return curated_list


def get_list_of_trainors():
    """Helper function to get the list of trainors available
    currently in the package.
    """
    list_trainors = inspect.getmembers(trainor_collection, inspect.isclass)
    curated_list = [indiv_arch[0] for indiv_arch in list_trainors]
    excluded_list = [
        "LearningRateScheduler",
        "JsonLoader",
        "Model",
        "ModelCheckpoint",
        "OnEpochEnd",
        "RMSprop",
    ]
    curated_list = [
        indiv_arch
        for indiv_arch in curated_list
        if indiv_arch not in excluded_list
    ]

    return curated_list


def get_list_of_inferrences():
    """Helper function to get the list of inferrences available
    currently in the package.
    """
    list_infers = inspect.getmembers(inferrence_collection, inspect.isclass)
    curated_list = [indiv_arch[0] for indiv_arch in list_infers]
    excluded_list = ["JsonLoader"]

    curated_list = [
        indiv_arch
        for indiv_arch in curated_list
        if indiv_arch not in excluded_list
    ]

    return curated_list


class GeneratorSchema(argschema.schemas.DefaultSchema):
    """defaults set in this class should be applicable to
    both ophys and ephys, and both the training generator
    and test generator.
    """

    name = argschema.fields.String(
        required=False,
        default="SingleTifGenerator",
        validate=OneOf(get_list_of_generators()),
        description=(
            "The data generator will control how data is read from individual\
            data files. Specify a data generator available in  \
            generator_collection.py, depending on your file data format."
        ),
    )

    pre_frame = argschema.fields.Int(
        required=False,
        default=30,
        description=(
            "Number of frames fed to the DeepInterpolation model before a \
            center frame for interpolation. Omitted frames will not be used \
            to fetch pre_frames. All pre_frame frame(s) will be fetched \
            before pre_omission frame(s)."
        ),
    )

    post_frame = argschema.fields.Int(
        required=False,
        default=30,
        description=(
            "Number of frames fed to the DeepInterpolation model after a \
            center frame for interpolation. Omitted frames will not be used \
            to fetch post_frames. All post_frame frame(s) will be fetch after \
            post_omission frame(s)."
        ),
    )

    pre_post_omission = argschema.fields.Int(
        required=False,
        default=0,
        description="Number of frames omitted before and after a center frame \
            for DeepInterpolation. Omission will be done on both sides of the \
            center frame, ie. twice pre_post_omission are omitted.\
            Omitted frames will not be used to fetch pre_frames and \
            post_frames."
        )

    data_path = argschema.fields.String(
        required=True,
        description="Path to the file containing data used by \
            the generator. Usually this will be a full filepath. In some \
            cases, this can point to a folder (with \
            MultiContinuousTifGenerator)"
    )

    batch_size = argschema.fields.Int(
        required=False,
        default=5,
        description="Batch size provided to the DeepInterpolation model by \
            the generator.",
    )

    start_frame = argschema.fields.Int(
        required=False,
        default=0,
        description="First frame used by the generator. First frame is 0.",
    )

    end_frame = argschema.fields.Int(
        required=False,
        default=-1,
        description=(
            "Last frame used by the generator. -1 defaults to the \
            last available frame. Negative values smaller than -1 \
            increasingly truncates the end of the dataset. 0 is not permitted.\
            Note: if end_frame = 2000, frame 2000 will be included \
            if there is sufficient post_frame frames available after frame \
            2000."
        ),
    )

    randomize = argschema.fields.Boolean(
        required=False,
        default=True,
        description="Whether to shuffle all selected frames in the generator.\
        It is recommended to set to 'True' for training and 'False' for \
        inference."
    )

    total_samples = argschema.fields.Int(
        required=False,
        default=-1,
        description="Total number of frames used between start_frame and \
            end_frame. -1 defaults to all available samples between \
            start_frame and end_frame. If total_samples is larger than the \
            number of available frames, it will automatically be reduced to \
            the maximal number."
    )

    @mm.pre_load
    def generator_specific_settings(self, data, **kwargs):
        # This is for backward compatibility
        if "train_path" in data:
            logging.warning("train_path has been deprecated and is to be \
replaced by data_path as generators can be used for training and inference. \
We are forwarding the value but please update your code.")
            data["data_path"] = data["train_path"]
            del data['train_path']
        if "pre_post_frame" in data:
            logging.warning("pre_post_frame has been deprecated and is to be \
replaced by pre_frame and post_frame. We are forwarding the value but please \
update your code.")
            data["pre_frame"] = data["pre_post_frame"]
            data["post_frame"] = data["pre_post_frame"]
            del data['pre_post_frame']
        return data


class MlflowRegistrySchema(argschema.schemas.DefaultSchema):
    tracking_uri = argschema.fields.String(
        required=True, description="MLflow tracking URI"
    )

    model_name = argschema.fields.String(
        required=True, description="Model name to fetch"
    )

    model_version = argschema.fields.Int(
        required=False,
        description="Model version to fetch. If neither model_version nor "
        "model_stage are provided, will try to fetch the latest "
        "model without a stage",
    )
    model_stage = argschema.fields.String(
        required=False,
        description="Model stage to fetch.If neither model_version nor "
        "model_stage are provided, will try to fetch the latest "
        "model without a stage",
    )


class ModelSourceSchema(argschema.schemas.DefaultSchema):
    mlflow_registry = argschema.fields.Nested(
        MlflowRegistrySchema,
        required=False,
        description="MLflow registry, if the model should be loaded from "
        "mlflow. If this is provided, then local_path should "
        "not be.",
    )
    local_path = argschema.fields.InputFile(
        required=False,
        description="Local path to model source. "
        "If this is provided then mlflow_registry should not be.",
    )

    @mm.validates_schema
    def validate(self, data):
        path_given = "local_path" in data
        mlflow_params_given = "mlflow_registry" in data
        if path_given and mlflow_params_given:
            raise ValidationError(
                "Either local_path or mlflow_registry "
                "should be supplied but not both"
            )

        if not path_given and not mlflow_params_given:
            raise ValidationError(
                "One of local_path or mlflow_registry should be supplied"
            )


class InferenceSchema(argschema.schemas.DefaultSchema):
    name = argschema.fields.String(
        required=False,
        default="core_inferrence",
        validate=OneOf(get_list_of_inferrences()),
        description=(
            "Inferrence class to use. All available classes are visible in \
            the inferrence_collection.py file as part of the \
            deepinterpolation package. More classes can be added to this file \
            to modify details of the inferrence behavior."
        ),
    )

    model_source = argschema.fields.Nested(
        ModelSourceSchema,
        description="Path to model if loading locally, or mlflow registry",
    )

    output_file = argschema.fields.OutputFile(
        required=True,
        description="Path to where the inference output will \
            get written. Inference will be saved in an hdf5 file \
            with a 'data' field",
    )

    save_raw = argschema.fields.Bool(
        required=False,
        default=False,
        description=(
            "Whether to save raw data along with the inference in the output \
            file. It will be saved as a 'raw' field in the hdf5 file. \
            This is useful for evaluation and direct comparison. \
            Output file will take twice hard drive space when set to 'True'."
        ),
    )

    output_datatype = argschema.fields.String(
        required=False,
        default="float32",
        validate=OneOf(['uint32', 'int32', 'uint16', 'int16', 'uint8', 'int8',
                        'float32', 'float16']),
        description=(
            "Output data type for inference. Default is float32. It is \
            important to keep in mind that DeepInterpolation can increase \
            available bit depth due to the increased Signal to Noise (SNR). \
            Smaller data types will save space at the cost of signal \
            resolution. Make sure to turn on 'rescaling' if that impacts the \
            output data range."
        ),
    )

    output_padding = argschema.fields.Bool(
        required=False,
        default=False,
        description=(
            "Whether to pad the output frames. DeepInterpolation requires \
            pre_frame and post_frame frames before and after any given frame \
            for inference. Setting this to 'True' will pad the start and end \
            of the output dataset with blank frames to keep the dataset of \
            the same size."
        ),
    )

    rescale = argschema.fields.Bool(
        required=False,
        default=True,
        description=(
            "Whether to bring back infered data to the original data range. \
            DeepInterpolation networks initially rescale all datasets within \
            -1 to 1 for training."
        ),
    )


class InferenceInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    run_uid = argschema.fields.Str(
        required=False,
        default=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"),
        description="unique identifier",
    )
    inference_params = argschema.fields.Nested(InferenceSchema, default={})
    generator_params = argschema.fields.Nested(GeneratorSchema, default={})
    output_full_args = argschema.fields.Bool(
        required=False,
        default=False,
        description=(
            "whether to output the full set of args to a json. "
            "this will show the args sent to the underlying classes "
            "including defaults."
        ),
    )

    @mm.post_load
    def inference_specific_settings(self, data, **kwargs):
        # This is to force randomize to be off if set by mistake
        if data["generator_params"]["randomize"]:
            logging.info("randomize should be set to False for inference. \
                        Overriding the parameter")
            data["generator_params"]["randomize"] = False

        # To disable rolling samples for inference
        data["generator_params"]["steps_per_epoch"] = -1

        return data


class TrainingSchema(argschema.schemas.DefaultSchema):
    name = argschema.fields.String(
        required=False,
        default="core_trainer",
        validate=OneOf(get_list_of_trainors()),
        description=(
            "Training class to use. All available classes are visible in the \
            trainor_collection.py file as part of the deepinterpolation \
            package. More classes can be added to this file to modify\
            details of the training behavior."
        ),
    )

    output_dir = argschema.fields.OutputDir(
        required=True,
        description="A folder where the training outputs will get written.",
    )

    steps_per_epoch = argschema.fields.Int(
        required=False,
        default=100,
        description="Number of batches per epoch. Here, epochs are not \
            defined in relation to the total number of batches in the dataset \
            as our datasets can be very large and it is beneficial to save \
            models and evaluate validation loss during training. After each \
            epoch a validation loss is computed and a checkpoint model is \
            potentialy saved (see period_save).")

    nb_times_through_data = argschema.fields.Int(
        required=False,
        default=1,
        description="Setting this to more than 1 will make the training use \
            individual batches multiple times during training, thereby \
            increasing your training samples. Larger repetition of the same\
            batches could cause noise overfitting",
    )

    learning_rate = argschema.fields.Float(
        required=False,
        default=0.0001,
        description="base learning rate used by the optimizer",
    )

    loss = argschema.fields.String(
        required=False,
        default="mean_squared_error",
        description="loss function used for training and validation. Loss \
            functions recognized by tensorflow are recognized : \
            https://www.tensorflow.org/api_docs/python/tf/keras/losses. \
            Additional losses can be added to the loss_collection.py file.",
    )

    model_string = argschema.fields.String(
        required=False,
        default="",
        description="Text string used to save the final model file and all\
            intermediary checkpoint models. Filename is constructed from other\
            fields if empty, using <network_name>_<loss>.",
    )

    caching_validation = argschema.fields.Bool(
        required=False,
        default=False,
        description="Whether to cache the validation data in memory \
            for training. On some systems, this could accelerate training as \
            it reduces the need for IO. On some system, the additional memory\
            requirement could cause memory issues.",
    )

    multi_gpus = argschema.fields.Bool(
        required=False,
        default=False,
        description="Set to True to use multi-gpus code when multi-gpus are \
            available and set up on the machine and environment. \
            Single GPU or CPU code is used if set to False.",
    )

    apply_learning_decay = argschema.fields.Bool(
        required=False,
        default=False,
        description="whether to use a learning scheduler during training.\
            If set to True, the learning rate will be halved every \
            <epochs_drop>",
    )

    epochs_drop = argschema.fields.Int(
        required=False,
        default=5,
        description="Number of epochs. Used when apply_learning_decay is \
            set to True. Will halve the learning rate every epoch_drop. \
            One epoch is defined using steps_per_epoch.",
    )

    period_save = argschema.fields.Int(
        required=False,
        default=5,
        description="Period in number of epochs to periodically save model \
            checkpoints.",
    )


class FineTuningSchema(argschema.schemas.DefaultSchema):
    name = argschema.fields.String(
        required=False,
        default="transfer_trainer",
        validate=OneOf(get_list_of_trainors()),
        description=(
            "Training class to use. All available classes are visible in the \
            trainor_collection.py file as part of the deepinterpolation \
            package. More classes can be added to this file to modify\
            details of the training behavior."
        ),
    )

    model_source = argschema.fields.Nested(
        ModelSourceSchema,
        description="Path to model if loading locally, or mlflow registry",
    )

    output_dir = argschema.fields.OutputDir(
        required=True,
        description="A folder where the training outputs will get written.",
    )

    steps_per_epoch = argschema.fields.Int(
        required=False,
        default=100,
        description="Number of batches per epoch. Here, epochs are not \
            defined in relation to the total number of batches in the dataset \
            as our datasets can be very large and it is beneficial to save \
            models and evaluate validation loss during training. After each \
            epoch a validation loss is computed and a checkpoint model is \
            potentialy saved (see period_save).")

    nb_times_through_data = argschema.fields.Int(
        required=False,
        default=1,
        description="Setting this to more than 1 will make the training use \
            individual samples multiple times during training, thereby \
            increasing your training samples. Larger repetition of the same\
            samples could cause noise overfitting",
    )

    learning_rate = argschema.fields.Float(
        required=False,
        default=0.0001,
        description="base learning rate used by the optimizer",
    )

    loss = argschema.fields.String(
        required=False,
        default="mean_squared_error",
        description="loss function used for training and validation. Loss \
            functions recognized by tensorflow are recognized : \
            https://www.tensorflow.org/api_docs/python/tf/keras/losses. \
            Additional losses can be added to the loss_collection.py file.",
    )

    model_string = argschema.fields.String(
        required=False,
        default="",
        description="Text string used to construct the final model filename\
            and all intermediary checkpoint models. Filename is constructed\
            from other fields if empty, using <network_name>_<loss>.",
    )

    caching_validation = argschema.fields.Bool(
        required=False,
        default=False,
        description="Whether to cache the validation data in memory \
            for training. On some system, this could accelerate training as it\
            reduces the need for IO. On some system, the additional memory\
            requirement could cause memory issues.",
    )

    multi_gpus = argschema.fields.Bool(
        required=False,
        default=False,
        description="Set to True to use multi-gpus code when multi-gpus are \
            available and set up on the machine and environment. \
            Single GPU or CPU code is used if set to False.",
    )

    apply_learning_decay = argschema.fields.Bool(
        required=False,
        default=False,
        description="whether to use a learning scheduler during training.\
            If set to True, the learning rate will be halved every \
            <epochs_drop>",
    )

    epochs_drop = argschema.fields.Int(
        required=False,
        default=5,
        description="Number of epochs. Used when apply_learning_decay is \
            set to True. Will half the learning rate every epoch_drop. \
            One epoch is defined using steps_per_epoch.",
    )

    period_save = argschema.fields.Int(
        required=False,
        default=5,
        description="Period in number of epochs to periodically save model \
            checkpoints.",
    )


class NetworkSchema(argschema.schemas.DefaultSchema):
    name = argschema.fields.String(
        required=True,
        default="unet_single_1024",
        validate=OneOf(get_list_of_networks()),
        description=(
            "callback of the neuronal network architecture to build.\
            All available architectures are visible in the \
            network_collection.py file as part of the deepinterpolation \
            package. More architectures callbacks can be added to this file \
            if necessary."
        ),
    )


class TrainingInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    run_uid = argschema.fields.Str(
        required=False,
        default=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"),
        description="unique identifier",
    )
    training_params = argschema.fields.Nested(TrainingSchema, default={})
    generator_params = argschema.fields.Nested(GeneratorSchema, default={})
    test_generator_params = argschema.fields.Nested(GeneratorSchema,
                                                    default={})
    network_params = argschema.fields.Nested(NetworkSchema, default={})
    output_full_args = argschema.fields.Bool(
        required=False,
        default=False,
        description=(
            "whether to output the full set of args to a json. "
            "this will show the args sent to the underlying classes "
            "including defaults."
        ),
    )

    @mm.post_load
    def training_specific_settings(self, data, **kwargs):
        # We forward this parameter to the generator
        data["generator_params"]["steps_per_epoch"] = \
            data["training_params"]["steps_per_epoch"]
        data["test_generator_params"]["steps_per_epoch"] = -1
        return data


class FineTuningInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    run_uid = argschema.fields.Str(
        required=False,
        default=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"),
        description="unique identifier",
    )
    finetuning_params = argschema.fields.Nested(FineTuningSchema, default={})
    generator_params = argschema.fields.Nested(GeneratorSchema, default={})
    test_generator_params = argschema.fields.Nested(GeneratorSchema,
                                                    default={})
    output_full_args = argschema.fields.Bool(
        required=False,
        default=False,
        description=(
            "whether to output the full set of args to a json. "
            "this will show the args sent to the underlying classes "
            "including defaults."
        ),
    )

    @mm.post_load
    def finetuning_specific_settings(self, data, **kwargs):
        # We forward this parameter to the generator
        data["generator_params"]["steps_per_epoch"] = \
            data["finetuning_params"]["steps_per_epoch"]
        data["test_generator_params"]["steps_per_epoch"] = -1

        return data
