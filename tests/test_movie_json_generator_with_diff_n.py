# a re-implementation of test_movie_json_generator.py with a
# different number of frames per movie

import copy
import pytest
import h5py
import json
import tempfile
import pathlib
import numpy as np
from deepinterpolation.generator_collection import MovieJSONGenerator


@pytest.fixture(scope='session')
def random_seed_fixture():
    return 221


@pytest.fixture(scope='session')
def frame_list_fixture():
    """
    Indexes of frames returned by MovieJSONGenerator
    """
    return [[4, 3, 5, 7],
            [8, 6, 3],
            [6, 7]]


@pytest.fixture(scope='session')
def movie_path_list_fixture(tmpdir_factory):
    """
    yields a list of paths to test movie files
    """
    path_list = []

    parent_tmpdir = tmpdir_factory.mktemp('movies_for_test')
    rng = np.random.default_rng(172312)
    this_dir = tempfile.mkdtemp(dir=parent_tmpdir)
    this_path = tempfile.mkstemp(dir=this_dir, suffix='.h5')[1]
    with h5py.File(this_path, 'w') as out_file:
        out_file.create_dataset('data',
                                data=rng.random((12, 512, 512)))

    path_list.append(this_path)

    this_dir = tempfile.mkdtemp(dir=parent_tmpdir)
    this_dir = pathlib.Path(this_dir) / 'processed'
    this_dir.mkdir()
    this_path = this_dir / 'concat_31Hz_0.h5'
    with h5py.File(this_path, 'w') as out_file:
        out_file.create_dataset('data',
                                data=rng.random((12, 512, 512)))
    path_list.append(str(this_path.resolve().absolute()))

    this_dir = tempfile.mkdtemp(dir=parent_tmpdir)
    this_dir = pathlib.Path(this_dir) / 'processed'
    this_dir.mkdir()
    this_path = this_dir / 'motion_corrected_video.h5'
    with h5py.File(this_path, 'w') as out_file:
        out_file.create_dataset('data',
                                data=rng.random((12, 512, 512)))
    path_list.append(str(this_path.resolve().absolute()))

    yield path_list

    for this_path in path_list:
        this_path = pathlib.Path(this_path)
        if this_path.is_file():
            this_path.unlink()


@pytest.fixture(scope='session')
def json_frame_specification_fixture(movie_path_list_fixture,
                                     tmpdir_factory,
                                     frame_list_fixture,
                                     random_seed_fixture):
    """
    yields a dict with the following key/value pairs

    'json_path' -- path to the file specifying the
                   movies/frames for the generator

    'expected_input' -- list of expected input
                        datasets returned by the generator

    'expected_output' -- list of expected output
                         datasets returned by the generator
    """

    params = dict()

    for ii, movie_path in enumerate(movie_path_list_fixture):
        this_params = dict()
        this_params['path'] = movie_path
        this_params['frames'] = frame_list_fixture[ii]
        this_params['mean'] = (ii+1)*2.1
        this_params['std'] = (ii+1)*3.4
        params[str(ii)] = this_params

    tmpdir = tmpdir_factory.mktemp('frame_specification')
    json_path = tempfile.mkstemp(
                    dir=tmpdir,
                    prefix='frame_specification_params_',
                    suffix='.json')[1]
    with open(json_path, 'w') as out_file:
        out_file.write(json.dumps(params))

    # now construct the input and output frames that
    # we expect this generator to yield
    expected_output_frames = []
    expected_input_frames = []

    path_to_data = dict()
    for movie_path in movie_path_list_fixture:
        with h5py.File(movie_path, 'r') as in_file:
            data = in_file['data'][()]
        path_to_data[movie_path] = data

    # replicate shuffling that happens inside the generator
    rng = np.random.default_rng(random_seed_fixture)
    index_list = list(range(len(movie_path_list_fixture)))
    # rng.shuffle(index_list)

    for ii in index_list:
        for i_frame in range(len(frame_list_fixture[ii])):
            this_params = params[str(ii)]
            mu = this_params['mean']
            std = this_params['std']
            movie_path = movie_path_list_fixture[ii]
            data = path_to_data[movie_path]
            frame = frame_list_fixture[ii][i_frame]
            output_data = (data[frame, :, :] - mu)/std

            input_indexes = np.array([frame-2, frame-1, frame+1, frame+2])
            input_data = (data[input_indexes, :, :]-mu)/std

            expected_output_frames.append(output_data)
            expected_input_frames.append(input_data)

    rng = np.random.default_rng(1234)
    rng.shuffle(expected_output_frames)
    rng = np.random.default_rng(1234)
    rng.shuffle(expected_input_frames)

    yield {'json_path': json_path,
           'expected_input': expected_input_frames,
           'expected_output': expected_output_frames}

    json_path = pathlib.Path(json_path)
    if json_path.is_file():
        json_path.unlink()


@pytest.fixture(scope='session')
def generator_params_fixture(
        tmpdir_factory,
        json_frame_specification_fixture,
        random_seed_fixture):
    """
    yields dict with generator params
    """

    params = dict()
    params['pre_post_omission'] = 0
    params['total_samples'] = -1
    params['name'] = 'MovieJSONGenerator'
    params['start_frame'] = 0
    params['end_frame'] = -1
    params['pre_frame'] = 2
    params['post_frame'] = 2
    params['randomize'] = True
    params['data_path'] = json_frame_specification_fixture['json_path']
    params['steps_per_epoch'] = -1
    params['train_path'] = json_frame_specification_fixture['json_path']
    params['type'] = 'generator'
    params['seed'] = random_seed_fixture
    return params


@pytest.mark.parametrize("batch_size", (1, 3, 5, 7, 50))
def test_movie_json_generator(
        movie_path_list_fixture,
        json_frame_specification_fixture,
        generator_params_fixture,
        frame_list_fixture,
        batch_size,
        tmpdir):

    json_path = tempfile.mkstemp(dir=tmpdir, suffix='.json')[1]
    params = copy.deepcopy(generator_params_fixture)
    params["batch_size"] = batch_size
    with open(json_path, 'w') as out_file:
        out_file.write(json.dumps(params, indent=2))

    expected_input = json_frame_specification_fixture['expected_input']
    expected_output = json_frame_specification_fixture['expected_output']

    generator = MovieJSONGenerator(json_path)
    lims_id_list = generator.lims_id

    n_frames = len(frame_list_fixture)
    dataset_ct = 0

    for batch in generator:
        # check that the dataset contains the expected input/output frames
        input_batch = batch[0]
        output_batch = batch[1]
        assert len(batch) == 2
        assert input_batch.shape[0] == output_batch.shape[0]
        for i_batch in range(input_batch.shape[0]):
            expected_i = expected_input[dataset_ct]
            expected_o = expected_output[dataset_ct]

            actual_o = output_batch[i_batch, :, :, 0]
            np.testing.assert_array_equal(actual_o, expected_o)

            actual_i = input_batch[i_batch, :, :, :].transpose(2, 0, 1)
            np.testing.assert_array_equal(actual_i, expected_i)

            dataset_ct += 1

    # make sure we got the expected number of datasets
    assert dataset_ct == len(lims_id_list)*n_frames
