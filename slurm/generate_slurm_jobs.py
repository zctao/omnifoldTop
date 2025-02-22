import os
import tarfile

import util

from datahandler_base import filter_filepaths

def get_all_samples_from_config(
    runCfg_d,
    sample_keys=['data', 'signal', 'background', 'bdata'],
    realpath = False,
    filterpath = True
    ):
    all_samples = []
    for key in sample_keys:
        samples = runCfg_d.get(key)
        if not samples:
            continue

        # clean up the sample names
        if filterpath:
            samples = filter_filepaths(samples)[0]

        if realpath:
            samples = [os.path.realpath(os.path.expandvars(sample)) for sample in samples]

        all_samples += samples

    return all_samples

def check_files_in_tarballs(tarballs, filelist):
    all_sample_files = []
    for sample_tar in tarballs:
        with tarfile.open(sample_tar) as ftar:
            for tarinfo in ftar:
                all_sample_files.append(tarinfo.name)

    # filelist should be a subset of all_sample_files
    if set(filelist) <= set(all_sample_files):
        return True
    else:
        print(f"Missing files in tarballs: {set(filelist) - set(all_sample_files)}")
        return False

def get_sample_tarball_map(
    runCfg_d,
    sample_dir, # top direcotry for sample files
    tarball_dir = 'tarballs', # top directory to look for sample tarball files
    sample_keys=['data', 'signal', 'background', 'bdata'],
    check_exist = True, # If True, check if the tarball exists
    include_unmatched = False, # If True, include the unmatched signal samples
    ):

    # tarball
    if not os.path.isabs(tarball_dir):
        tarball_dir = os.path.join(sample_dir, tarball_dir)

    tarballs_dict = {}

    sample_files_config = get_all_samples_from_config(runCfg_d, sample_keys, realpath=True, filterpath=True)
    for sample_file in sample_files_config:
        sample_relpath = os.path.relpath(sample_file, sample_dir)
        sample_type, syst_type = sample_relpath.split('/')[0:2]

        if sample_type in ['obs', 'fakes']: # data samples
            tarball = os.path.join(tarball_dir, 'nominal.tar')
        elif sample_type in ['Wjets', 'Zjets', 'singleTop_sch', 'singleTop_tch', 'singleTop_tW_DR_dyn', 'singleTop_tW_DS_dyn', 'ttH', 'ttV', 'VV'] + ['ttbar']: # MC samples
            tarball = os.path.join(tarball_dir, f"{syst_type}.tar")
        elif sample_type.startswith('ttbar_'): # alternative signal samples
            assert syst_type=='nominal', f"No {syst_type} avaialble for alternative ttbar sample {sample_type}"
            tarball = os.path.join(tarball_dir, f"{sample_type}.tar")
        else:
            raise ValueError(f"Unknown sample type {sample_type} with syst_type {syst_type}")

        if not tarball in tarballs_dict:
            tarballs_dict[tarball] = set()
        tarballs_dict[tarball].add(sample_relpath)

        # A special case: include unmatched ttbar samples
        if include_unmatched and sample_type=='ttbar' and syst_type=='nominal':
            tarballs_dict[tarball].add("{}_unmatched_truth{}".format(*os.path.splitext(sample_relpath)))

    if check_exist:
        for tarball in tarballs_dict:
            assert os.path.isfile(tarball), f"Cannot find file {tarball}!"
            assert check_files_in_tarballs([tarball], tarballs_dict[tarball])

    return tarballs_dict

def generate_slurm_jobs(
    config_name, # file name of the run config
    sample_dir, # top direcotry for sample files
    email = os.getenv('USER')+'@phas.ubc.ca', # for now
    account = "def-alister",
    output_name = None, # slurm job file name,
    output_dir = None, # output directory
    check_tarfiles = True, # if True, check if all input files are available in tarballs
    site = "cc" # site name; cc or ubc
    ):

    if output_dir is None:
        # creat a 'slurm' directory in the same directory as the config file
        output_dir = os.path.join(os.path.dirname(os.path.abspath(config_name)), f'slurm_{site}')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not output_name:
        # set it to the same as config_name but with a different extension
        output_name = os.path.splitext(os.path.basename(config_name))[0]

    jobfile_name = os.path.join(output_dir, output_name+'.slurm')
    jobcfg_name = os.path.join(output_dir, output_name+'.json')

    # sample directory
    sample_realdir = os.path.realpath(os.path.expandvars(sample_dir))

    # Load run config
    runcfg = util.read_dict_from_json(config_name, parse_env=False)

    # Determine the needed tarballs in case they are not provided
    tarballs_map = get_sample_tarball_map(
        runcfg,
        sample_realdir,
        tarball_dir='tarballs',
        sample_keys=['data', 'signal', 'background', 'bdata'],
        check_exist=check_tarfiles,
        include_unmatched = runcfg.get('correct_acceptance')!=False # no need to include unmatched samples for computing binned efficiency corrections if correct_acceptance is explicitly set to False
    )

    # Modify run config for batch jobs
    inputdir_job = './inputs'
    outputdir_job = './outputs'

    # inputs
    for key in ['data', 'signal', 'background', 'bdata']:
        samples = get_all_samples_from_config(runcfg, sample_keys=[key], realpath=True, filterpath=False)
        if samples:
            # replace the absolute paths of samples with the paths on the node
            runcfg[key] = [os.path.join(inputdir_job, os.path.relpath(sample, sample_realdir)) for sample in samples]

    # write filelists
    filelists_tarball = []
    for i,tarball in enumerate(tarballs_map):
        flist_name = os.path.join(output_dir, f"{output_name}_filelist_{i}.txt")
        filelists_tarball.append(flist_name)
        with open(flist_name, 'w') as flist:
            for fpath in tarballs_map[tarball]:
                flist.write(f"{fpath}\n")

    # tarball list
    # try to replace absolute paths to the data directory with the soft link
    tarball_names = [tname.replace(os.path.expandvars("/data/${USER}"), os.path.expandvars("${HOME}/data"), 1) for tname in tarballs_map.keys()]

    # output
    resultdir = os.path.abspath(runcfg['outputdir'])
    if not os.path.isdir(resultdir):
        print(f"Create output directory {resultdir}")
        os.makedirs(resultdir)

    result_tarball = os.path.join(resultdir, "results.tar")

    # replace the output directory in the config with a local directory on the node
    runcfg['outputdir'] = outputdir_job

    if site=="ubc":
        # choose gpu
        runcfg["gpu"] = 0 if output_name.endswith("_down") else 1

    # write the new config file
    util.write_dict_to_json(runcfg, jobcfg_name)

    # Write the slurm job file
    job_common_dict = {
        "USEREMAIL" : email,
        "LOGFILE" : os.path.join(resultdir, "slurm-%j.log"),
        "TARBALLLIST" : " ".join(tarball_names),
        "INFILELIST" : " ".join(filelists_tarball),
        "RUNCONFIG" : jobcfg_name,
        "INPUTDIR" : inputdir_job,
        "OUTPUTDIR" : outputdir_job,
        "RESULT" : result_tarball
    }

    # Load slurm job template
    if site == "cc":
        slurm_template = os.path.expandvars("${SOURCE_DIR}/slurm/ccJob.template")
        format_dict = job_common_dict.copy()
        format_dict.update({
            "ACCOUNT" : account,
        })
    elif site == "ubc":
        slurm_template = os.path.expandvars("${SOURCE_DIR}/slurm/ubcJob.template")
        format_dict = job_common_dict
    else:
        raise ValueError(f"Unknown site: {site}")

    with open(slurm_template) as ftmp:
        job_str = ftmp.read()
        job_str = job_str.format_map(format_dict)

    with open(jobfile_name, 'w') as fout:
        fout.write(job_str)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create Slurm jobs for a given run configuration')

    parser.add_argument('config_name', type=str, help='The run configuration file')
    parser.add_argument('-d', '--sample-dir', type=str, action=util.ParseEnvVar,
                        default="${DATA_DIR}/ntuplerTT/latest",
                        help='Top direcotry for sample files')
    parser.add_argument('-e', '--email', type=str, action=util.ParseEnvVar, default="${USER}@phas.ubc.ca", help="Email address for job notifications")
    parser.add_argument('-a', '--account', type=str, default="def-alister", help="Slurm account for computecanada")
    parser.add_argument('-o', '--output-name', type=str, help="Slurm job file name")
    parser.add_argument('--output-dir', type=str, help="Output directory")
    parser.add_argument('-s', '--site', choices=['cc', 'ubc'], help="Site to run jobs")

    args = parser.parse_args()

    generate_slurm_jobs(**vars(args))