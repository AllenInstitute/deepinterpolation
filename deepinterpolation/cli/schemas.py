import argschema
import marshmallow as mm
from marshmallow import ValidationError
import datetime


class GeneratorSchema(argschema.schemas.DefaultSchema):
    """defaults set in this class should be applicable to
    both ophys and ephys, and both the training generator
    and test generator.
    """
    type = argschema.fields.String(
        required=False,
        default="generator",
        description="")
    name = argschema.fields.String(
        required=False,
        default="",
        description="")
    pre_post_frame = argschema.fields.Int(
        required=False,
        default=30,
        description="")
    train_path = argschema.fields.InputFile(
        required=True,
        description="training data input path")
    batch_size = argschema.fields.Int(
        required=False,
        default=5,
        description="")
    start_frame = argschema.fields.Int(
        required=False,
        default=1000,
        description="")
    end_frame = argschema.fields.Int(
        required=False,
        default=-1,
        description="")
    randomize = argschema.fields.Int(
        required=False,
        default=1,
        description="integer as bool?")
    steps_per_epoch = argschema.fields.Int(
        required=False,
        default=100,
        description="")
    total_samples = argschema.fields.Int(
        required=False,
        default=-1,
        description="")


class TrainingSchema(argschema.schemas.DefaultSchema):
    """defaults set in this class should be applicable to
    both ophys and ephys.
    """
    type = argschema.fields.String(
        required=False,
        default="trainer",
        description="")
    name = argschema.fields.String(
        required=False,
        default="transfer_trainer",
        description="")
    model_path = argschema.fields.InputFile(
        required=True,
        description="path to model source for transfer training.")
    run_uid = argschema.fields.Str(
        required=False,
        default=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"),
        description="unique identifier")
    batch_size = argschema.fields.Int(
        required=False,
        default=5,
        description="")
    steps_per_epoch = argschema.fields.Int(
        required=False,
        default=100,
        description="")
    period_save = argschema.fields.Int(
        required=False,
        default=25,
        description="")
    nb_gpus = argschema.fields.Int(
        required=False,
        default=1,
        description="number of GPUs")
    nb_times_through_data = argschema.fields.Int(
        required=False,
        default=3,
        description="number of passes")
    learning_rate = argschema.fields.Float(
        required=False,
        default=0.0005,
        description="")
    initial_learning_rate = argschema.fields.Float(
        required=False,
        default=0.0005,
        description="")
    apply_learning_decay = argschema.fields.Int(
        default=1,
        required=False,
        description="int interpreted as bool?")
    epochs_drop = argschema.fields.Int(
        default=10,
        required=False,
        description="")
    loss = argschema.fields.String(
        required=False,
        default="mean_squared_error",
        description="loss specifier")
    output_dir = argschema.fields.OutputDir(
        required=True,
        description="ouptut destination")
    caching_validation = argschema.fields.Bool(
        require=False,
        default=False,
        description="need this to resolve memory error")

    @mm.post_load
    def set_model_str(self, data, **kwargs):
        data["model_string"] = f"transfer_train__{data['loss']}"
        return data


class TransferTrainerInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    train_path = argschema.fields.InputFile(
        required=True,
        description="training data input path")
    batch_size = argschema.fields.Int(
        required=False,
        default=5,
        description="")
    steps_per_epoch = argschema.fields.Int(
        required=False,
        default=100,
        description="")
    set_defaults = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description="with 'ophys', for example, populates some defaults")
    output_dir = argschema.fields.OutputDir(
        required=True,
        description="ouptut destination")
    training_params = argschema.fields.Nested(
        TrainingSchema,
        default={})
    generator_params = argschema.fields.Nested(
        GeneratorSchema,
        default={})
    generator_test_params = argschema.fields.Nested(
        GeneratorSchema,
        default={})
    output_full_args = argschema.fields.Bool(
        required=False,
        default=False,
        description=("whether to output the full set of args to a json. "
                     "this will show the args sent to the underlying classes "
                     "including defaults."))

    @mm.pre_load
    def set_outdir(self, data, **kwargs):
        data["training_params"]["output_dir"] = data["output_dir"]
        return data

    @mm.pre_load
    def set_train_path(self, data, **kwargs):
        data["generator_params"]["train_path"] = data["train_path"]
        data["generator_test_params"]["train_path"] = data["train_path"]
        return data

    @mm.pre_load
    def set_batch_epoch(self, data, **kwargs):
        """sets a consistent batch_size and steps_per_epoch
        across the three Nested schema
        """
        for k in ["training_params",
                  "generator_params",
                  "generator_test_params"
                  ]:
            for e in ["batch_size", "steps_per_epoch"]:
                data[k][e] = data[e]
        data["generator_test_params"]["steps_per_epoch"] = -1
        return data

    @mm.post_load
    def set_ophys_defaults(self, data, **kwargs):
        """sets ophys-specific defaults
        """
        if "set_defaults" in data:
            if data["set_defaults"] == "ophys":
                data["generator_params"]["name"] = "OphysGenerator"
                data["generator_test_params"]["name"] = "OphysGenerator"
                data["generator_params"]["start_frame"] = 1000
                data["generator_test_params"]["start_frame"] = 0
                data["generator_params"]["end_frame"] = -1
                data["generator_test_params"]["end_frame"] = 1000
        return data


class MlflowSchema(argschema.schemas.DefaultSchema):
    tracking_uri = argschema.fields.String(
        required=True,
        description="MLflow tracking URI"
    )
    model_name = argschema.fields.String(
        required=True,
        description="Model name"
    )
    model_version = argschema.fields.Int(
        required=False,
        description='Model version. Either this needs to be supplied or '
                    'model_stage'
    )
    model_stage = argschema.fields.String(
        required=False,
        description='Model stage. Either this needs to be supplied or '
                    'model_version'
    )

    @mm.validates_schema
    def validate_model_version(self, data):
        model_version_given = 'model_version' in data
        model_stage_given = 'model_stage' in data

        if model_version_given and model_stage_given:
            raise ValidationError('Either model_version or model_stage should '
                                  'be supplied but not both')


class InferenceSchema(argschema.schemas.DefaultSchema):
    type = argschema.fields.String(
        required=False,
        default="inferrence",
        description="")
    name = argschema.fields.String(
        required=False,
        default="core_inferrence",
        description="")
    mlflow_params = argschema.fields.Nested(
        MlflowSchema,
        required=False,
        description="MLflow params, if the model should be loaded from mlflow."
                    "If this is provided, then model_path should not be.")
    model_path = argschema.fields.InputFile(
        required=False,
        description="Local path to model source for transfer training. "
                    "If this is provided then mlflow_params should not be.")
    output_file = argschema.fields.OutputFile(
        required=True,
        description="")
    save_raw = argschema.fields.Bool(
        required=False,
        default=True,
        description="")
    rescale = argschema.fields.Bool(
        required=False,
        default=False,
        description="")

    @mm.validates_schema
    def validate_model_path(self, data):
        model_path_given = 'model_path' in data
        mlflow_params_given = 'mlflow_params' in data
        if model_path_given and mlflow_params_given:
            raise ValidationError('Either model_path or mlflow_params should '
                                  'be supplied but not both')

        if not model_path_given and not mlflow_params_given:
            raise ValidationError('One of model_path or mlflow_params should '
                                  'be supplied')


class InferenceInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    run_uid = argschema.fields.Str(
        required=False,
        default=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"),
        description="unique identifier")
    inference_params = argschema.fields.Nested(
        InferenceSchema,
        default={})
    generator_params = argschema.fields.Nested(
        GeneratorSchema,
        default={})
    output_full_args = argschema.fields.Bool(
        required=False,
        default=False,
        description=("whether to output the full set of args to a json. "
                     "this will show the args sent to the underlying classes "
                     "including defaults."))
    n_frames_chunk = argschema.fields.Int(
        required=False,
        default=-1,
        description=("if not -1 and n_parallel_processors != 1 "
                     "this will set the number of frames per job "
                     "to run in parallel."))
    n_parallel_processes = argschema.fields.Int(
        required=False,
        default=1,
        description=("a lowish number to keep the processors busy"))

    @mm.post_load
    def inference_specific_settings(self, data, **kwargs):
        data['generator_params']['name'] = "OphysGenerator"
        data['generator_params']['randomize'] = 0
        return data
