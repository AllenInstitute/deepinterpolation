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
    model_path = argschema.fields.InputFile(
        required=True,
        description="path to model source for transfer training.")
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
