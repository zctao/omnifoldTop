import os

import util
from generate_slurm_jobs import get_all_samples_from_config, get_sample_tarball_map

def generate_slurm_hist_jobs(
    job_name, # job name
    result_dirs, # list of unfolding result directories
    sample_dir = "./inputs", # top directory for sample files
    tarball_dir = "${DATA_DIR}/ntuplerTT/latest/tarballs", # directory for tarballs
    submitdir = ".",
    histograms_args = "" # options passed to make_histogramsv2.py
    ):

    result_dirs = [os.path.abspath(rdir) for rdir in result_dirs]

    # input samples
    samples_from_tarball = set()
    for rdir in result_dirs:
        fpath_args = os.path.join(rdir, "arguments.json")
        args_d = util.read_dict_from_json(fpath_args, parse_env=False)

        samples = get_all_samples_from_config(args_d)

        for sample in samples:
            # check if exist
            if not os.path.isfile(sample):
                # try to retrieve from tarball later
                samples_from_tarball.add(sample)

    if samples_from_tarball:
        # retrieve tarball map
        tarballs_map = get_sample_tarball_map(
            samples_from_tarball,
            sample_dir = sample_dir,
            tarball_dir = os.path.abspath(tarball_dir),
        )
    else:
        tarballs_map = {}

    # tarball list
    # try to replace absolute paths to the data directory with the soft link
    tarball_names = [tname.replace(os.path.expandvars("/data/${USER}"), os.path.expandvars("${HOME}/data"), 1) for tname in tarballs_map.keys()]

    # filelist for untarring
    filelists_tarball = []
    for i,tarball in enumerate(tarballs_map):
        flist_name = os.path.abspath(os.path.join(submitdir, f"filelist_{i}.txt"))
        filelists_tarball.append(flist_name)
        with open(flist_name, 'w') as flist:
            for fpath in tarballs_map[tarball]:
                flist.write(f"{fpath}\n")

    inputdir_job = './inputs'

     # Write the slurm job file
    job_common_dict = {
        "LOGFILE" : os.path.join(os.path.abspath(submitdir), f"slurm-{job_name}-%j.log"),
        "TARBALLLIST" : " ".join(tarball_names),
        "INFILELIST" : " ".join(filelists_tarball),
        "INPUTDIR" : inputdir_job,
        "RESULTDIRLIST" : " ".join(result_dirs),
        "HISTARGS": histograms_args
    }

    slurm_tempalte = os.path.expandvars("${SOURCE_DIR}/slurm/ubcHistJob.template")
    format_dict = job_common_dict.copy()

    with open(slurm_tempalte) as ftmp:
        job_str = ftmp.read()
        job_str = job_str.format_map(format_dict)

    jobfile_name = os.path.join(submitdir, job_name+".slurm")
    with open(jobfile_name, 'w') as fout:
        fout.write(job_str)

    return jobfile_name

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("job_name", type=str, help="job name")
    parser.add_argument("result_dirs", type=str, nargs='+',
                        help="list of unfolding result directories")
    parser.add_argument('-d', '--sample-dir', type=str, action=util.ParseEnvVar,
                        default="./inputs",
                        help='Top direcotry for sample files')
    parser.add_argument("-t", "--tarball-dir", type=str, action=util.ParseEnvVar, 
                        default="${DATA_DIR}/ntuplerTT/latest/tarballs",
                        help="directory for tarballs")
    parser.add_argument("-j", "--submitdir", type=str, default='.',
                        help="directory to submit the job")
    parser.add_argument("-a", "--histograms-args", type=str, default="",
                        help="options passed to make_histogramsv2.py")

    args = parser.parse_args()

    generate_slurm_hist_jobs(**vars(args))