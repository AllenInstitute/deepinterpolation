import json

import h5py
import numpy as np
import pytest

from deepinterpolation.cli.schemas import GeneratorSchema


@pytest.fixture
def movie_json_generator_args(tmpdir):
    """Returns args for MovieJSONGenerator"""
    data_meta_path = tmpdir / "train_data.json"
    data_path = tmpdir / 'train_mov.h5'
    ophys_experiment_id = 123

    with h5py.File(data_path, "w") as f:
        f.create_dataset("data", data=np.ones((80, 512, 512), dtype='int16'))

    with open(data_meta_path, "w") as f:
        f.write(json.dumps({
            ophys_experiment_id: {
                "path": str(data_path),
                "frames": [40],
                "std": 1,
                "mean": 0
            }
        }))
    args = GeneratorSchema().load({
        "name": "MovieJSONGenerator",
        "data_path": str(data_meta_path)
    })
    yield args
