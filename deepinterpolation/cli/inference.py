import argschema
import json
import h5py
import time
import numpy as np
from pathlib import Path

from deepinterpolation.cli.schemas import InferenceInputSchema
from deepinterpolation.generic import ClassLoader


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

        p = Path(self.args["inference_params"]["output_file"])
        outfile = p.parent / f"{uid}_{p.name}"
        self.args["inference_params"]["output_file"] = str(outfile)

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

        t0 = time.time()
        inferrence_class.run()
        dt = time.time() - t0
        with h5py.File(self.args["generator_params"]["train_path"], "r") as f:
            nframes = f["data"].shape[0]
        rate = nframes / (dt / 60.0)
        self.logger.info(f"inference on {nframes} frames in {int(dt)} "
                         f"seconds: {rate:.2f} frames / min")

        del inferrence_obj
        del generator_obj

        # patch up the output movie
        self.logger.info("fixing up the range and shape of the result")
        with h5py.File(self.args["generator_params"]["train_path"], "r") as f:
            dmax = f["data"][()].max()
            dshape = f["data"].shape

        with h5py.File(outfile, "r") as f:
            d = f["data"][()].squeeze()

        d = (d - d.min()) * dmax / d.ptp()
        d = d.astype('uint16')
        nextra = dshape[0] - d.shape[0]
        dextra = np.zeros((nextra, *d.shape[1:]), dtype='uint16')
        d = np.concatenate((d, dextra), axis=0)

        with h5py.File(outfile, "w") as f:
            f.create_dataset("data", data=d)
        self.logger.info(f"wrote {outfile}")


if __name__ == "__main__":
    infer = Inference()
    infer.run()
