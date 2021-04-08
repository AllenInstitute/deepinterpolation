import argschema
import json
import h5py
import numpy as np
import multiprocessing
from pathlib import Path

from deepinterpolation.cli.schemas import InferenceInputSchema
from deepinterpolation.generic import ClassLoader


def mp_job(inf_params, gen_params, uid, job_index, indices):
    outdir = Path(inf_params['output_file']).parent

    inference_json_path = outdir / f"{uid}_{job_index}_inference.json"
    out_path = Path(inf_params["output_file"])
    split_name = f"{out_path.stem}_{job_index}{out_path.suffix}"
    inf_params["output_file"] = str(out_path.parent / split_name)
    with open(inference_json_path, "w") as f:
        json.dump(inf_params, f,  indent=2)

    generator_json_path = outdir / f"{uid}__{job_index}_generator.json"
    gen_params["start_frame"] = indices[0]
    gen_params["end_frame"] = indices[1]
    with open(generator_json_path, "w") as f:
        json.dump(gen_params, f, indent=2)

    generator_obj = ClassLoader(generator_json_path)
    data_generator = generator_obj.find_and_build()(generator_json_path)

    inferrence_obj = ClassLoader(inference_json_path)
    inferrence_class = inferrence_obj.find_and_build()(
            inference_json_path,
            data_generator)

    inferrence_class.run()

    return inf_params["output_file"]


class Inference(argschema.ArgSchemaParser):
    default_schema = InferenceInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        outdir = Path(self.args['inference_params']['output_file']).parent
        if self.args["output_full_args"]:
            full_args_path = outdir / "inference_full_args.json"
            with open(full_args_path, "w") as f:
                json.dump(self.args, f, indent=2)
            self.logger.info(f"wrote {full_args_path}")
        uid = self.args['run_uid']
        self.logger.info(f"inference run {self.args['run_uid']} started")

        if (self.args["n_frames_chunk"] != -1) & \
                (self.args["n_parallel_workers"] != 1):
            index_list = []
            with h5py.File(
                    self.args["generator_params"]["train_path"], "r") as f:
                nframes = f["data"].shape[0]
            start_frame = 0
            while True:
                index_list.append([start_frame,
                                   start_frame + self.args["n_frames_chunk"]])
                if index_list[-1][-1] >= nframes:
                    index_list[-1][-1] = nframes
                    break
                start_frame = index_list[-1][-1] - \
                    self.args["generator_params"]["pre_post_frame"]

            mp_args = []
            for job_index, indices in enumerate(index_list):
                mp_args.append([self.args["inference_params"],
                                self.args["generator_params"],
                                uid,
                                job_index,
                                indices])
                print(indices)
        else:
            mp_args = [[self.args["inference_params"],
                        self.args["generator_params"],
                        uid,
                        job_index,
                        indices]]

        self.logger.info(f"breaking job into {len(mp_args)} chunks")

        if len(mp_args) == 1:
            split_files = [mp_job(*mp_args)]
        else:
            with multiprocessing.Pool(self.args["n_parallel_workers"]) as pool:
                split_files = pool.starmap(mp_job, mp_args)

        self.logger.info("inference jobs done, "
                         "concatenating and normalizing.")

        # stats about the input movie
        with h5py.File(self.args["generator_params"]["train_path"], "r") as f:
            dmax = f["data"][()].max()
            dshape = f["data"].shape

        # read in result, concatenating
        outputs = []
        for split_file in split_files:
            with h5py.File(split_file, "r") as f:
                outputs.append(f["data"][()].squeeze())
            self.logger.info(f"read {outputs[-1].shape[0]} "
                             "frames from {split_file}")
            # delete the file
            Path(split_file).unlink()
        d = np.concatenate(outputs, axis=0)

        # normalize
        d = (d - d.min()) * dmax / d.ptp()
        d = d.astype('uint16')
        nextra = dshape[0] - d.shape[0]
        dextra = np.zeros((nextra, *d.shape[1:]), dtype='uint16')
        d = np.concatenate((d, dextra), axis=0)

        # output
        with h5py.File(
                self.args["inference_params"]["output_file"], "w") as f:
            f.create_dataset("data", data=d)
        self.logger.info(
            f"wrote {self.args['inference_params']['output_file']}")


if __name__ == "__main__":
    infer = Inference()
    infer.run()
