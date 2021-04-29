from pathlib import Path

import argschema
import mlflow
import numpy as np
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from tensorflow.python.keras.models import load_model

import deepinterpolation.loss_collection as lc


class InputSchema(argschema.ArgSchema):
    mlflow_tracking_uri = argschema.fields.String(
        required=True,
        description='MLFLow tracking URI'
    )
    model_paths = argschema.fields.List(
        argschema.fields.InputFile,
        required=True,
        description='Trained model paths'
    )


class RegisterModels(argschema.ArgSchemaParser):
    default_schema = InputSchema

    def run(self):
        mlflow.set_tracking_uri(self.args['mlflow_tracking_uri'])

        for model_path in self.args['model_paths']:
            model_path = Path(model_path)
            self._register_model(model_path=model_path)

    def _register_model(self, model_path: Path,
                        experiment_name='DeepInterpolation'):
        experiment_id = self._fetch_or_create_experiment(
            experiment_name=experiment_name)

        with mlflow.start_run(experiment_id=experiment_id):
            model = load_model(model_path,
                               custom_objects={
                                   "annealed_loss": lc.loss_selector(
                                       "annealed_loss")})
            mlflow.keras.log_model(keras_model=model, artifact_path='models',
                                   registered_model_name=model_path.stem)

    @staticmethod
    def _fetch_or_create_experiment(experiment_name):
        client = MlflowClient()
        experiments = client.list_experiments(view_type=ViewType.ALL)
        experiment_names = [e.name for e in experiments]
        if experiment_name not in experiment_names:
            experiment_id = mlflow.create_experiment(name=experiment_name)
        else:
            experiment = client.get_experiment_by_name(name=experiment_name)
            experiment_id = experiment.experiment_id
        return experiment_id


def main():
    app = RegisterModels()
    app.run()


if __name__ == '__main__':
    main()
