import argschema
import marshmallow
import json
import h5py
import numpy as np
import time
import pathlib
import multiprocessing


class DataCacheGeneratorSchema(argschema.ArgSchema):

    output_path = argschema.fields.OutputFile(
            required=True,
            description='HDF5 file where cache will be written')

    training_path = argschema.fields.InputFile(
            required=True,
            description=('Path to json file specifying the training '
                         'frames to use'))

    validation_path = argschema.fields.InputFile(
            required=True,
            description=('Path to json file specifying the validation '
                         'frames to use'))

    clobber = argschema.fields.Boolean(
                required=False,
                default=False,
                description='Whether or not to overwrite existing output_path')

    n_parallel_workers = argschema.fields.Integer(
            required=False,
            default=1,
            description=('Number of workers to use when creating the cache'))

    compression_level = argschema.fields.Integer(
            required=False,
            default=6,
            description=('Level of compression applied to data when '
                         'writing to the cache (higher is more compressed '
                         'but slower; must be 0-9)'))

    pre_frame = argschema.fields.Int(
        required=False,
        default=30,
        description=(
            "Number of frames fed to the DeepInterpolation model before a \
            center frame for interpolation. Omitted frames will not be used \
            to fetch pre_frames. All pre_frame frame(s) will be fetched \
            before pre_omission frame(s)."
        ),
    )

    post_frame = argschema.fields.Int(
        required=False,
        default=30,
        description=(
            "Number of frames fed to the DeepInterpolation model after a \
            center frame for interpolation. Omitted frames will not be used \
            to fetch post_frames. All post_frame frame(s) will be fetch after \
            post_omission frame(s)."
        ),
    )

    @marshmallow.post_load
    def validate_paths(self, data, **kwargs):
        """
        Make sure data_path points to .json file and
        output_path points to .h5 file
        """
        msg = ''
        if data['compression_level'] < 0 or data['compression_level'] > 9:
           msg += "\ncompresion_level must be between 0 and 9; you gave "
           msg += f"{data['compression_level']}\n"

        for k in ('training_path', 'validation_path'):
            if not data[k].endswith('.json'):
                msg += "\n{k} needs to point to .json file\n"
                msg += f"you gave: {data[k]}\n"

        if not data['output_path'].endswith('.h5'):
            msg += "\noutput_path needs to point to .h5 file\n"
            msg += f"you gave: {data['output_path']}"

        if len(msg) > 0:
            raise ValueError(msg)

        if not data['clobber']:
            output_path = pathlib.Path(data['output_path'])
            if output_path.exists():
                raise ValueError(f"{data['output_path']} exists")

        return data

def video_frame_to_locale(video_key,
                          i_frame,
                          cache_manifest,
                          frame_index_to_group,
                          video_key_to_index,
                          n_frames_per_frame):
    """
    video_key -- a str from self.video_key_list
    i_frame -- index within json_data[video_key]['frames']
    cache_manifest -- dict mapping group_tag to (i0, i1)
    frame_index_to_group -- dict mapping i_global to group_tag
    """

    i_video = video_key_to_index[video_key]
    i_frame_global = i_frame*len(video_key_to_index)+i_video
    group_tag = frame_index_to_group[i_frame_global]
    group_window = cache_manifest[group_tag]
    i_output = i_frame_global - group_window[0]

    i0 = i_output*n_frames_per_frame
    i1 = i0+n_frames_per_frame
    return {'i_output': i_output,
            'input_window': (i0, i1),
             'group': group_tag,
             'i_frame_global': i_frame_global}


def cache_population_worker(
        video_key_list,
        video_json_data,
        total_frame_lookup,
        output_lock,
        output_path):

    for video_key in video_key_list:
        video_path = video_json_data[video_key]['path']
        frame_index = total_frame_lookup[video_key]
        with h5py.File(video_path, 'r') as in_file:
            frame_data = in_file['data'][()]
        frame_data = frame_data[frame_index, :, :]
        with output_lock:
            with h5py.File(output_path, 'a') as out_file:
                np.testing.assert_array_equal(
                    out_file[f'{video_key}_index'][()],
                    frame_index)
                out_file[video_key][:, :, :] = frame_data



class DataCacheGenerator(argschema.ArgSchemaParser):

    default_schema = DataCacheGeneratorSchema

    @property
    def n_frames_per_video(self):
        frames = self.video_json_data[self.video_key_list[0]]['frames']
        n_frames_per_video = len(frames)
        return n_frames_per_video

    @property
    def n_frames_per_frame(self):
        n_frames_per_frame = self.args['post_frame'] + self.args['pre_frame']
        return n_frames_per_frame

    def read_json_data(self):
        """
        Load and validate the JSON file that points to the video data
        this cache will rely upon

        Save contents of json file in self.video_json_data and a sorted
        video key list in self.video_key_list
        """
        with open(self.args['data_path'], 'rb') as in_file:
            json_data = json.load(in_file)
        video_key_list = list(json_data.keys())
        #video_key_list.sort()

        # make sure that every video has the same number of frames
        # sampled from it (that is a requirement of the generator
        # this cache will feed)
        n_frames = len(json_data[video_key_list[0]]['frames'])
        msg = ''
        for video_key in video_key_list:
            n = len(json_data[video_key]['frames'])
            if n != n_frames:
                msg += f'video {video_key} has {n} frames; '
                msg += f'it should have {n_frames}\n'

        # make sure every specified video points to a valid path
        for video_key in video_key_list:
            video_path = pathlib.Path(json_data[video_key]['path'])
            if not video_path.is_file():
                msg += f'{video_path.resolve().absolute()} is not a file\n'

        if len(msg) > 0:
            raise RuntimeError(msg)

        self.video_key_list = video_key_list
        self.video_json_data = json_data
        self.video_key_to_index = {k: ii for ii, k in enumerate(video_key_list)}

    def validate_videos(self):
        """
        Check that videos all have the same dtype and that none of them
        require frames that are within pre/post of the end
        """
        video_dtype = None
        frame_shape = None
        msg = ""
        for video_key in self.video_key_list:
            video_path = self.video_json_data[video_key]['path']
            with h5py.File(video_path, 'r') as in_file:
                this_dtype = in_file['data'].dtype
                if video_dtype is None:
                    video_dtype = this_dtype
                else:
                    if in_file['data'].dtype != video_dtype:
                        msg += f"{video_path} dtype is {str(this_dtype)}; "
                        msg += f"should be {str(video_dtype)}\n"
                if frame_shape is None:
                    frame_shape = in_file['data'].shape[1:]
                else:
                    if in_file['data'].shape[1:] != frame_shape:
                        msg += f"{video_path} shape is "
                        msg += f"{in_file['data'].shape}; "
                        msg += "frame shape should be {frame_shape}\n"

                these_frames = self.video_json_data[video_key]['frames']
                min_frame = min(these_frames)
                max_frame = max(these_frames)
                if min_frame < self.pre_frame:
                    msg += f"{video_path} min_frame {min_frame}; "
                    msg += f"is within {self.pre_frame} of beginning of movie\n"
                if max_frame > (in_file['data'].shape[0]-self.post_frame):
                    msg += f"{video_path} max_frame {max_frame}; "
                    msg += "is within {self.post_frame} of end of movie\n"
            if len(msg) > 0:
                raise RuntimeError(msg)
        self.video_dtype = video_dtype
        self.frame_shape = frame_shape

    def create_empty_cache(self):
        """
        Create datasets for video_key and video_key_index
        (video_key_index is a numpy array of the frames needed
        from the movie pointed to by video_key)
        """

        total_frame_lookup = dict()
        i_frame_to_frame = dict()
        for video_key in self.video_key_list:
            frame_list = self.video_json_data[video_key]['frames']
            video_path = self.video_json_data[video_key]['path']
            with h5py.File(video_path, 'r') as in_file:
                n_frames = in_file['data'].shape[0]
            frame_set = set()
            i_frame_arr = []
            for frame in frame_list:
                i_frame_arr.append(frame)
                start_frame = max(0, frame-self.args['pre_frame'])
                end_frame = min(n_frames, frame+self.args['post_frame']+1)
                for i_frame in range(start_frame, end_frame):
                    frame_set.add(i_frame)
            frame_set = np.sort(np.array([f for f in frame_set]))
            total_frame_lookup[video_key] = frame_set
            i_frame_to_frame[video_key] = np.array(i_frame_arr)

        with h5py.File(self.args['output_path'], 'w') as output_file:
            for video_key in self.video_key_list:
                frame_set = total_frame_lookup[video_key]
                i_frame_arr = i_frame_to_frame[video_key]
                output_file.create_dataset(
                                video_key,
                                shape=(len(frame_set),
                                       self.frame_shape[0],
                                       self.frame_shape[1]),
                                dtype=self.video_dtype,
                                shuffle=True,
                                chunks=(1+self.args['post_frame']-self.args['pre_frame'],
                                        self.frame_shape[0],
                                        self.frame_shape[1]),
                                compression='gzip',
                                compression_opts=self.args['compression_level'])

                # this will list all of the frame indices as they are in the
                # original movie
                output_file.create_dataset(
                            f'{video_key}_index',
                            data=frame_set)

                # this will map i_frame in the training dataset to frame index
                # in the original movie
                output_file.create_dataset(
                            f'{video_key}_i_frame_to_video_frame',
                            data=i_frame_arr)

        return total_frame_lookup


    def populate_cache(self, total_frame_lookup):
        n_workers = self.args['n_parallel_workers']
        mgr = multiprocessing.Manager()

        output_lock = mgr.Lock()

        if n_workers > 1:
            n_videos = len(self.video_key_list)
            n_videos_per_worker = np.ceil(n_videos/n_workers).astype(int)
            n_videos_per_worker = max(n_videos_per_worker, 1)
            process_list = []
            for i0 in range(0, n_videos, n_videos_per_worker):
                i1 = i0 + n_videos_per_worker
                sub_list = self.video_key_list[i0:i1]
                process = multiprocessing.Process(
                            target = cache_population_worker,
                            args = (
                               sub_list,
                               self.video_json_data,
                               total_frame_lookup,
                               output_lock,
                               self.args['output_path']))
                process.start()
                process_list.append(process)

            for process in process_list:
                process.join()

        else:
            cache_population_worker(
                self.video_key_list,
                self.video_json_data,
                total_frame_lookup,
                output_lock,
                self.args['output_path'])

    def run(self):
        self.pre_frame = self.args['pre_frame']
        self.post_frame = self.args['post_frame']
        self.read_json_data()
        self.validate_videos()

        print('writing empty cache')
        total_frame_lookup = self.create_empty_cache()

        print('populating cache')
        self.populate_cache(total_frame_lookup)

        print('assembling metadata')
        mean_lookup = dict()
        std_lookup = dict()
        for video_key in self.video_key_list:
            mean_lookup[video_key] = self.video_json_data[video_key]['mean']
            std_lookup[video_key] = self.video_json_data[video_key]['std']

        metadata = dict()
        metadata['pre_frame'] = int(self.args['pre_frame'])
        metadata['post_frame'] = int(self.args['post_frame'])
        metadata['mean_lookup'] = mean_lookup
        metadata['std_lookup'] = std_lookup
        metadata['n_frames_per_video'] = int(self.n_frames_per_video)
        metadata['video_key_list'] = self.video_key_list
        with h5py.File(self.args['output_path'], 'a') as output_file:
            output_file.create_dataset(
                    'metadata',
                    data=json.dumps(metadata).encode('utf-8'))


if __name__ == "__main__":
    gen = DataCacheGenerator()
    gen.run()
