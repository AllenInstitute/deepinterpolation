import json
from pathlib import Path

import argschema
import tensorflow as tf

from deepinterpolation.cli.schemas import InferenceInputSchema
from deepinterpolation.generic import ClassLoader


class Inference(argschema.ArgSchemaParser):
    default_schema = InferenceInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        outdir = Path(self.args["inference_params"]["output_file"]).parent
        if self.args["output_full_args"]:
            full_args_path = outdir / "inference_full_args.json"
            with open(full_args_path, "w") as f:
                json.dump(self.args, f, indent=2)
            self.logger.info(f"wrote {full_args_path}")
        uid = self.args["run_uid"]

        # This is used to send to the legacy parameter tracking system
        # to specify each sub-object type.
        self.args["generator_params"]["type"] = "generator"
        self.args["inference_params"]["type"] = "inferrence"

        # To be removed once fully transitioned to CLI
        self.args["generator_params"]["train_path"] = self.args["generator_params"][
            "data_path"
        ]

        # disable multiprocessing if gpu_available
        if tf.test.is_gpu_available():
            self.args["inference_params"]["use_multiprocessing"] = False
            self.logger.warn("GPU is available, multiprocessing is disabled")

        # save the json parameters to 2 different files
        inference_json_path = outdir / f"{uid}_inference.json"
        with open(inference_json_path, "w") as f:
            json.dump(self.args["inference_params"], f, indent=2)
        self.logger.info(f"wrote {inference_json_path}")

        generator_json_path = outdir / f"{uid}_generator.json"
        with open(generator_json_path, "w") as f:
            json.dump(self.args["generator_params"], f, indent=2)
        self.logger.info(f"wrote {generator_json_path}")

        generator_obj = ClassLoader(generator_json_path)
        data_generator = generator_obj.find_and_build()(generator_json_path)

        inferrence_obj = ClassLoader(inference_json_path)
        inferrence_class = inferrence_obj.find_and_build()(
            inference_json_path, data_generator
        )

        self.logger.info("created objects for inference")
        if self.args["inference_params"].get("use_multiprocessing"):
            #            tf.config.threading.set_inter_op_parallelism_threads(1)
            #            tf.config.threading.set_intra_op_parallelism_threads(1)
            #            inferrence_class.run_multiprocessing()
            self.logger.warn(
                "use_multiprocessing for inference is current"
                "in progress, running with tensorflow native"
                "processor management"
            )
        inferrence_class.run()


if __name__ == "__main__":  # pragma: nocover
    infer = Inference()
    infer.run()
