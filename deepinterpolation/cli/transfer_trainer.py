import argschema
import json
from pathlib import Path

from deepinterpolation.cli.schemas import TransferTrainerInputSchema
from deepinterpolation.generic import ClassLoader


class TransferTrainer(argschema.ArgSchemaParser):
    default_schema = TransferTrainerInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        outdir = Path(self.args['output_dir'])

        # save the json parameters to 3 different files
        training_json_path = outdir / "training.json"
        with open(training_json_path, "w") as f:
            json.dump(self.args['training_params'], f,  indent=2)
        self.logger.info(f"wrote {training_json_path}")

        generator_json_path = outdir / "generator.json"
        with open(generator_json_path, "w") as f:
            json.dump(self.args['generator_params'], f, indent=2)
        self.logger.info(f"wrote {generator_json_path}")

        test_generator_json_path = outdir / "test_generator.json"
        with open(test_generator_json_path, "w") as f:
            json.dump(self.args['generator_test_params'], f, indent=2)
        self.logger.info(f"wrote {test_generator_json_path}")

        # run the training
        generator_obj = ClassLoader(generator_json_path)
        generator_test_obj = ClassLoader(test_generator_json_path)
        trainer_obj = ClassLoader(training_json_path)

        self.logger.info("created objects for training")

        train_generator = generator_obj.find_and_build()(generator_json_path)
        self.logger.info("built train_generator")
        test_generator = generator_test_obj.find_and_build()(
                test_generator_json_path)
        self.logger.info("built test_generator")
        training_class = trainer_obj.find_and_build()(
            train_generator, test_generator, training_json_path)
        self.logger.info("built trainer")
        training_class.run()
        self.logger.info("trainer.run() complete")
        training_class.finalize()
        self.logger.info("trainer.finalize() complete")


if __name__ == "__main__":
    tt = TransferTrainer()
    tt.run()
