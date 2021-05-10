# Using the CLI

## Inference

### Using mlflow

[mlflow](https://mlflow.org/docs/latest/index.html) is an open-source package for managing machine learning models.

It supports an idea of [model registry](https://mlflow.org/docs/latest/model-registry.html) in which models are versioned. 

In addition to loading a pre-trained model for inference from a local file, DeepInterpolation supports loading a model from an mlflow registry.

#### Example

To use a model registered with mlflow instead of a local model, just supply `mlflow_params`to `inference_params` instead of `model_path `.

```
python -m deepinterpolation.cli.inference \
    --inference_params.mlflow_params.tracking_uri <tracking_uri> \
    --inference_params.mlflow_params.model_name <model_name> 
    ...
```

If running locally, the mlflow tracking_uri would be something like `http://localhost:5000`

For more details about running an mlflow server, visit [here](https://mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers).