import argschema


class MlflowSchema(argschema.schemas.DefaultSchema):
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
        description='Model version to fetch'
    )
    model_stage = argschema.fields.String(
        default='None',
        description='Model stage to fetch'
    )