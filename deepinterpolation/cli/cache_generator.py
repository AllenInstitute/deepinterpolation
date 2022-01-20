import argschema
import marshmallow
import json
import h5py
import numpy as np
import time
import pathlib


class DataCacheGeneratorSchema(argschema.ArgSchema):

    output_path = argschema.fields.OutputFile(
            required=True,
            description='HDF5 file where cache will be written')

    data_path = argschema.fields.InputFile(
            required=True,
            description='Path to json file specifying the frames to use')

    clobber = argschema.fields.Boolean(
                required=False,
                default=False,
                description='Whether or not to overwrite existing output_path')

    frames_per_dataset = argschema.fields.Integer(
            required=False,
            default=2000,
            description=('Number of frames and their pre/post to store per '
                         'dataset in the final HDF5 file'))

    flush_every = argschema.fields.Integer(
            required=False,
            defaulte=500,
            desription=('Write data to cache every time you have '
                        'this many frames in memory (approximately)'))

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
        if not data['data_path'].endswith('.json'):
            msg += "\ndata_path needs to point to .json file\n"
            msg += f"you gave: {data['data_path']}\n"
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

    def video_frame_to_locale(self,
                              video_key,
                              i_frame,
                              cache_manifest,
                              frame_index_to_group):
        """
        video_key -- a str from self.video_key_list
        i_frame -- index within json_data[video_key]['frames']
        cache_manifest -- dict mapping group_tag to (i0, i1)
        frame_index_to_group -- dict mapping i_global to group_tag
        """

        i_video = self.video_key_to_index[video_key]
        i_frame_global = i_frame*len(self.video_key_list)+i_video
        group_tag = frame_index_to_group[i_frame_global]
        group_window = cache_manifest[group_tag]
        i_output = i_frame_global - group_window[0]

        i0 = i_output*self.n_frames_per_frame
        i1 = i0+self.n_frames_per_frame
        return {'i_output': i_output,
                'input_window': (i0, i1),
                'group': group_tag,
                'i_frame_global': i_frame_global}

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
        Returns
        -------
        cache_manifest -- dict mapping group name to (i0, i1) of frames
        frame_index_to_group -- dict mapping frame name to (i0, i1)
        """

        n_total_frames = self.n_frames_per_video*len(self.video_key_list)

        cache_manifest = dict()
        frame_index_to_group = dict()

        # create the empty HDF5 file
        with h5py.File(self.args['output_path'], 'w') as out_file:
            dataset_ct = 0
            for i0 in range(0, n_total_frames, self.args['frames_per_dataset']):
                i1 = min(n_total_frames, i0+self.args['frames_per_dataset'])
                group_tag = f'group_{dataset_ct}'
                data_group = out_file.create_group(group_tag)
                n_frames = i1-i0
                n_full_frames = n_frames*self.n_frames_per_frame
                data_group.create_dataset('input_frames',
                                          compression='gzip',
                                          shuffle=True,
                                          shape=(n_full_frames,
                                                 self.frame_shape[0],
                                                 self.frame_shape[1]),
                                          dtype=self.video_dtype,
                                          chunks=(1,
                                                  self.frame_shape[0],
                                                  self.frame_shape[1]))
                data_group.create_dataset('output_frames',
                                          compression='gzip',
                                          shuffle=True,
                                          shape=(n_frames,
                                                 self.frame_shape[0],
                                                 self.frame_shape[1]),
                                          dtype=self.video_dtype,
                                          chunks=(1,
                                                  self.frame_shape[0],
                                                  self.frame_shape[1]))

                cache_manifest[group_tag] = (int(i0), int(i1))
                for ii in range(i0, i1):
                    frame_index_to_group[ii] = group_tag
                dataset_ct += 1

        return cache_manifest, frame_index_to_group

    def populate_cache(self, cache_manifest, frame_index_to_group):

        i_frame_to_mean = dict()
        i_frame_to_std = dict()

        data_chunks = []

        n_video_keys = len(self.video_key_list)
        read_t0 = time.time()
        for i_video_key, video_key in enumerate(self.video_key_list):
            video_path = self.video_json_data[video_key]['path']
            mu = self.video_json_data[video_key]['mean']
            std = self.video_json_data[video_key]['std']
            with h5py.File(video_path, 'r') as in_file:
                for i_frame in range(self.n_frames_per_video):
                    locale = self.video_frame_to_locale(
                                    video_key,
                                    i_frame,
                                    cache_manifest,
                                    frame_index_to_group)
                    i_frame_to_mean[locale['i_frame_global']] = mu
                    i_frame_to_std[locale['i_frame_global']] = std

                    frame = self.video_json_data[video_key]['frames'][i_frame]
                    f0 = frame - self.args['pre_frame']
                    f1 = frame + self.args['post_frame'] + 1
                    full_data = in_file['data'][f0:f1, :, :]
                    output_frame = full_data[self.args['pre_frame'], :, :]
                    input_dexes = np.ones(f1-f0, dtype=bool)
                    input_dexes[self.args['pre_frame']] = False
                    input_frames = full_data[input_dexes, :, :]
                    data_chunks.append({'locale': locale,
                                        'output_frame': output_frame,
                                        'input_frames': input_frames})

            read_duration = time.time()-read_t0
            per = read_duration / (i_video_key+1)
            predicted = per*n_video_keys
            remaining = predicted-read_duration
            print(f'read {i_video_key} of {n_video_keys} in '
                  f'{read_duration:.2e} seconds; '
                  f'predict {remaining:.2e} of {predicted:.2e} left to go '
                  f'-- rate {per:.2e}')

            do_flush = False
            if len(data_chunks) >= self.args['flush_every']:
                do_flush = True
            if video_key == self.video_key_list[-1]:
                do_flush = True
            if do_flush:
                print('flushing')
                with h5py.File(self.args['output_path'], 'a') as out_file:
                    for chunk in data_chunks:
                        locale = chunk['locale']
                        output_frame = chunk['output_frame']
                        input_frames = chunk['input_frames']
                        group = out_file[locale['group']]
                        group['output_frames'][locale['i_output']] = output_frame
                        input_window = locale['input_window']
                        group['input_frames'][input_window[0]:
                                              input_window[1]] = input_frames

                data_chunks = []

        return i_frame_to_mean, i_frame_to_std

    def run(self):
        self.pre_frame = self.args['pre_frame']
        self.post_frame = self.args['post_frame']
        self.read_json_data()
        self.validate_videos()

        print('writing empty cache')
        (cache_manifest,
         frame_index_to_group) = self.create_empty_cache()

        print('populating cache')
        (i_frame_to_mean,
         i_frame_to_std) = self.populate_cache(cache_manifest, frame_index_to_group)

        print('assembling metadata')
        metadata = dict()
        metadata['manifest'] = cache_manifest
        metadata['frame_index_to_group'] = frame_index_to_group
        metadata['pre_frame'] = int(self.args['pre_frame'])
        metadata['post_frame'] = int(self.args['post_frame'])
        metadata['n_frames'] = int(len(frame_index_to_group))
        metadata['mean_lookup'] = i_frame_to_mean
        metadata['std_lookup'] = i_frame_to_std
        metadata['n_videos'] = int(len(self.video_key_list))
        metadata['n_frames_per_video'] = int(self.n_frames_per_video)
        with h5py.File(self.args['output_path'], 'a') as output_file:
            output_file.create_dataset(
                    'metadata',
                    data=json.dumps(metadata).encode('utf-8'))


if __name__ == "__main__":
    gen = DataCacheGenerator()
    gen.run()
