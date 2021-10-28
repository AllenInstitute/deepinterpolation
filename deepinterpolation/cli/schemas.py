import argschema
import marshmallow as mm
import datetime
from marshmallow import ValidationError
import inspect
from deepinterpolation import network_collection
from deepinterpolation import generator_collection
from deepinterpolation import trainor_collection
from deepinterpolation import inferrence_collection

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
    excluded_list = ["MaxRetryException", "JsonLoader"]
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
            "Specify a data generator available in the generator_collection.py\
            . Choose according to your data format"
        ),
    )

    pre_post_frame = argschema.fields.Int(
        required=False,
        default=30,
        description=(
            "number of frames considered before and after a frame \
                for interpolation."
        ),
    )
    pre_post_omission = argschema.fields.Int(required=False, default=0,
                                             description="")
    train_path = argschema.fields.InputFile(
        required=True, description="training data input path"
    )
    batch_size = argschema.fields.Int(
        required=False,
        default=5,
        description="batch size provided to model by generator",
    )
    start_frame = argschema.fields.Int(
        required=False,
        default=1000,
        description="first frame for starting the generator.",
    )
    end_frame = argschema.fields.Int(
        required=False,
        default=-1,
        description=(
            "last frame for the generator. -1 defaults to "
            "last frame in input data set."
        ),
    )
    randomize = argschema.fields.Int(
        required=False,
        default=0,
        description="integer as bool. useful in training, not inference.",
    )
    steps_per_epoch = argschema.fields.Int(required=False, default=100,
                                           description="")
    total_samples = argschema.fields.Int(
        required=False,
        default=-1,
        description="-1 defaults to all samples between start_frame\
            and end_frame.",
    )


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
            "Inferrence class to use. All available classes are visible in the \
            inferrence_collection.py file as part of the deepinterpolation \
            package. More classes can be added to this file to modify\
            details of the inferrence behavior."
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
            "Whether to save raw data along with the infered in the output \
            file. This is useful for evaluation and direct comparison. \
            Output file will take twice hard drive space when set to true."
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
        # Commented to allow CLI to work with more than just h5 files
        # data['generator_params']['name'] = "OphysGenerator"
        # To remove when updating CLI with pre/post processing modules
        data["generator_params"]["randomize"] = 0
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


class FineTuningInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    run_uid = argschema.fields.Str(
        required=False,
        default=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"),
        description="unique identifier",
    )
    training_params = argschema.fields.Nested(FineTuningSchema, default={})
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
