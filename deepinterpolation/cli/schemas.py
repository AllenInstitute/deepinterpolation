import argschema
import marshmallow as mm
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


class InferenceSchema(argschema.schemas.DefaultSchema):
    type = argschema.fields.String(
        required=False,
        default="inferrence",
        description="")
    name = argschema.fields.String(
        required=False,
        default="core_inferrence",
        description="")
    model_path = argschema.fields.InputFile(
        required=True,
        description="path to model source for transfer training.")
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
        data['generator_params']['name'] = "OphysGenerator"
        data['generator_params']['randomize'] = 0
        return data
