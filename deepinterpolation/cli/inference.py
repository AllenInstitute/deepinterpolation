import argschema
import json
import h5py
import numpy as np
from pathlib import Path

from deepinterpolation.cli.schemas import InferenceInputSchema
from deepinterpolation.generic import ClassLoader


def normalize_uint16_output(input_file: Path,
                            output_file: Path) -> np.ndarray:
    """function to globally normalize and convert datatype of output
    movies. The output movie is shifted to be >=0 and scaled to the
    same max value as the input movie. The movie is zero-padded at the
    end to match the number of frames of the input movie.

    Parameters
    ----------
    input_file: Path
        the pre-denoising movie. Used to establish histogram bounds
        of intensity.
    output_file: Path
        the post-denoising movie. It is floating point and needs to
        be scaled and type converted

    Returns
    -------
    data: np.ndarray
        the scaled and type converted result

    Notes
    -----
    We anticipate removing this when the Inference class is upgraded to
    take care of negative values and convert to uint16.

    """
    with h5py.File(input_file, "r") as f:
        inmax = f["data"][()].max()
        inshape = f["data"].shape

    with h5py.File(output_file, "r") as f:
        out = f["data"][()].squeeze()

    out = (out - out.min()) * inmax / out.ptp()
    out = out.astype('uint16')
    nextra = inshape[0] - out.shape[0]
    dextra = np.zeros((nextra, *out.shape[1:]), dtype='uint16')
    out = np.concatenate((out, dextra), axis=0)
    return out


class Inference(argschema.ArgSchemaParser):
    default_schema = InferenceInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        outdir = Path(self.args['inference_params']['output_file']).parent
        if self.args["output_full_args"]:
            full_args_path = outdir / "inference_full_args.json"
            with open(full_args_path, "w") as f:
                json.dump(self.args, f, indent=2)
            self.logger.info(f"wrote {full_args_path}")
        uid = self.args['run_uid']

        # This is used to send to the legacy parameter tracking system
        # to specify each sub-object type.
        self.args["generator_params"]["type"] = "generator"
        self.args["inference_params"]["type"] = "inferrence"

        # save the json parameters to 2 different files
        inference_json_path = outdir / f"{uid}_inference.json"
        with open(inference_json_path, "w") as f:
            json.dump(self.args['inference_params'], f,  indent=2)
        self.logger.info(f"wrote {inference_json_path}")

        generator_json_path = outdir / f"{uid}_generator.json"
        with open(generator_json_path, "w") as f:
            json.dump(self.args['generator_params'], f, indent=2)
        self.logger.info(f"wrote {generator_json_path}")

        generator_obj = ClassLoader(generator_json_path)
        data_generator = generator_obj.find_and_build()(generator_json_path)

        inferrence_obj = ClassLoader(inference_json_path)
        inferrence_class = inferrence_obj.find_and_build()(
            inference_json_path,
            data_generator)

        self.logger.info("created objects for inference")
        inferrence_class.run()

        # patch up the output movie
        # This code below will go within the inference library as pre/post
        # processings modules. Adding temporary fix to remove for non-h5 files
        # so that the CLI works with tiff, dat, ... files.
        if '.h5' in self.args["generator_params"]["train_path"]:
            self.logger.info("fixing up the range and shape of the result")
            data = normalize_uint16_output(
                    Path(self.args["generator_params"]["train_path"]),
                    Path(self.args["inference_params"]["output_file"]))
            with h5py.File(
                    self.args["inference_params"]["output_file"], "w") as f:
                f.create_dataset("data", data=data)
            self.logger.info(
                f"wrote {self.args['inference_params']['output_file']}")


if __name__ == "__main__":  # pragma: nocover
    infer = Inference()
    infer.run()
