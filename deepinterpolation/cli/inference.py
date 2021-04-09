import argschema
import json
import h5py
import time
import numpy as np
from pathlib import Path

from deepinterpolation.cli.schemas import InferenceInputSchema
from deepinterpolation.generic import ClassLoader


def chunk_job(inf_params, gen_params, uid, job_index, indices):
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

        if (self.args["n_frames_chunk"] != -1):
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

            chunk_args = []
            for job_index, indices in enumerate(index_list):
                chunk_args.append([self.args["inference_params"],
                                self.args["generator_params"],
                                uid,
                                job_index,
                                indices])
        else:
            chunk_args = [[self.args["inference_params"],
                           self.args["generator_params"],
                           uid,
                           job_index,
                           indices]]

        self.logger.info(f"breaking job into {len(chunk_args)} chunks"
                         f"with limit of {self.args['n_frames_chunk']} frames"
                         "per chunk")

        split_files = []
        for chunk_arg in chunk_args:
            t0 = time.time()
            split_files.append(chunk_job(chunk_arg))
            dt = time.time() - t0
            with h5py.File(slpit_files[-1], "r") as f:
                nframes = f["data"].shape[0]
            rate = nframes / (dt / 60.0)
            self.logger.info(f"inference on {nframes} frames in {int(dt)} "
                             f"seconds: {rate:.2f} frames / min")
        
        self.logger.info("inference jobs done, "
                         "concatenating and normalizing.")


        # stats about the input movie
        with h5py.File(self.args["generator_params"]["train_path"], "r") as f:
            indata_max = f["data"][()].max()
            indata_shape = f["data"].shape
            indata_dtype = f["data"].dtype

        # read in result, concatenating to output file
        p = Path(self.args["inference_params"]["output_file"])
        outfile = p.parent / f"{uid}_{p.name}"
        with h5py.File(outfile, "w") as fout:
            dout = fout.create_dataset("data",
                                       shape=indata_shape,
                                       dtype=indata_dtype,
                                       fillvalue=0)
            start_index = 0
            for split_file in split_files:
                with h5py.File(split_file, "r") as fin:
                    partial_data = fin["data"][()].squeeze()
                pmin = partial_data.min()
                pmax = partial_data.max()
                pptp = partial_data.ptp()
             partial_data = (partial_data - pmin) * indata_max / pptp
             partial_data = partial_data.astype('uint16')
             npartial = partial_data.shape[0]
             dout[start_index:(start_index + npartial)] = partial_data
             start_index += npartial

             # delete the file
             Path(split_file).unlink()
        self.logger.info(f"wrote {outfile}")


if __name__ == "__main__":
    infer = Inference()
    infer.run()
