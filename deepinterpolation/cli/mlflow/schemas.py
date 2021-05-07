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