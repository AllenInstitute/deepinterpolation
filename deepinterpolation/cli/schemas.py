import argschema
import marshmallow as mm
import datetime

from marshmallow import ValidationError


class GeneratorSchema(argschema.schemas.DefaultSchema):
    """defaults set in this class should be applicable to
    both ophys and ephys, and both the training generator
    and test generator.
    """
    type = argschema.fields.String(
        required=False,
        default="generator",
        description="sent to ClassLoader to instantiate a generator class.")
    name = argschema.fields.String(
        required=False,
        default="",
        description=("used in conjunction with above to instantiate object "
                     "via ClassLoader"))
    pre_post_frame = argschema.fields.Int(
        required=False,
        default=30,
        description=("number of frames considered before and after a frame "
                     "for interpolation."))
    pre_post_omission = argschema.fields.Int(
        required=False,
        default=0,
        description="")
    train_path = argschema.fields.InputFile(
        required=True,
        description="training data input path")
    batch_size = argschema.fields.Int(
        required=False,
        default=5,
        description="batch size provided to model by generator")
    start_frame = argschema.fields.Int(
        required=False,
        default=1000,
        description="first frame for starting the generator.")
    end_frame = argschema.fields.Int(
        required=False,
        default=-1,
        description=("last frame for the generator. -1 defaults to "
                     "last frame in input data set."))
    randomize = argschema.fields.Int(
        required=False,
        default=0,
        description="integer as bool. useful in training, not inference.")
    steps_per_epoch = argschema.fields.Int(
        required=False,
        default=100,
        description="")
    total_samples = argschema.fields.Int(
        required=False,
        default=-1,
        description="-1 defaults to all samples in input data set.")


class MlflowRegistrySchema(argschema.schemas.DefaultSchema):
    tracking_uri = argschema.fields.String(
        required=True,
        description="MLflow tracking URI"
    )
    model_name = argschema.fields.String(
        required=True,
        description="Model name to fetch"
    )
    model_version = argschema.fields.Int(
        required=False,
        description='Model version to fetch. If neither model_version nor '
                    'model_stage are provided, will try to fetch the latest '
                    'model without a stage'
    )
    model_stage = argschema.fields.String(
        required=False,
        description='Model stage to fetch.If neither model_version nor '
                    'model_stage are provided, will try to fetch the latest '
                    'model without a stage'
    )


class ModelSourceSchema(argschema.schemas.DefaultSchema):
    mlflow_registry = argschema.fields.Nested(
        MlflowRegistrySchema,
        required=False,
        description="MLflow registry, if the model should be loaded from mlflow."
                    "If this is provided, then local_path should not be.")
    local_path = argschema.fields.InputFile(
        required=False,
        description="Local path to model source. "
                    "If this is provided then mlflow_registry should not be.")

    @mm.validates_schema
    def validate(self, data):
        path_given = 'local_path' in data
        mlflow_params_given = 'mlflow_registry' in data
        if path_given and mlflow_params_given:
            raise ValidationError('Either local_path or mlflow_registry should '
                                  'be supplied but not both')

        if not path_given and not mlflow_params_given:
            raise ValidationError('One of local_path or mlflow_registry should '
                                  'be supplied')


class InferenceSchema(argschema.schemas.DefaultSchema):
    type = argschema.fields.String(
        required=False,
        default="inferrence",
        description=("type and name sent to ClassLoader for object "
                     "instantiation"))
    name = argschema.fields.String(
        required=False,
        default="core_inferrence",
        description=("type and name sent to ClassLoader for object "
                     "instantiation"))
    model_source = argschema.fields.Nested(
        ModelSourceSchema,
        description="Path to model if loading locally, or mlflow registry"
    )
    output_file = argschema.fields.OutputFile(
        required=True,
        description="where the infernce output will get written.")
    save_raw = argschema.fields.Bool(
        required=False,
        default=True,
        description=("currently using this to perform global normalization "
                     "on floating point output. Will likely stop doing this "
                     "soon."))
    rescale = argschema.fields.Bool(
        required=False,
        default=False,
        description=("currently not using the chunked rescaling as it does "
                     "not handle negative values and convert to uint16."))


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

    @mm.post_load
    def inference_specific_settings(self, data, **kwargs):
        # Commented to allow CLI to work with more than just h5 files
        # data['generator_params']['name'] = "OphysGenerator"
        # To remove when updating CLI with pre/post processing modules
        data['generator_params']['randomize'] = 0
        return data
