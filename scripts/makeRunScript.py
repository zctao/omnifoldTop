#!/usr/bin/env python3
import os
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('configFiles', nargs='*',
                    help='config files for running tests')
parser.add_argument('-b', '--base-config', dest='baseConfig',
                    default='configs/run/basic_tests.json')
parser.add_argument('-o', '--output', type=str,
                    help='output shell script file name')
parser.add_argument('-c', '--cluster', action='store_true',
                    help="If true, write the job file for a Slurm cluster")
parser.add_argument('-s', '--source-dir', dest='sourceDir',
                    default='/home/ztao/topUnfolding/omnifoldTop')
parser.add_argument('--cluster-outdir', dest='clusterOutdir',
                    default="/home/ztao/work/batch_output/OmniFold/latest",
                    help="Output directory to transfer batch job results")

args = parser.parse_args()

def read_dict_from_json(filename):
    jfile = open(filename, "r")
    try:
        jdict = json.load(jfile)
    except json.decoder.JSONDecoderError:
        jdict = {}
    jfile.close()
    return jdict

def update_nested_dict(baseDictionary, newDictionary):
    for newkey in newDictionary:
        if newkey not in baseDictionary:
            # new key is not in the base dictionary, add it
            baseDictionary[newkey] = newDictionary[newkey]
        else:
            # key exists in both dictoinaries
            # if values of the key are both of dictionary type
            if type(baseDictionary[newkey])==dict and type(newDictionary[newkey])==dict:
                # keep updating
                update_nested_dict(baseDictionary[newkey], newDictionary[newkey])
            else:
                # overwrite the base dictionary with the new one
                baseDictionary[newkey] = newDictionary[newkey]
    return baseDictionary

# read base config
runConfig = read_dict_from_json(args.baseConfig)

# read all configs and update the base one
for jfile in args.configFiles:
    config_dict = read_dict_from_json(jfile)
    runConfig = update_nested_dict(runConfig, config_dict)
#print(runConfig)

testLabel = runConfig.get('label','')

# create output file
if args.output:
    outfilename = args.output
else:
    outfilename = 'runtests'
    if testLabel:
        outfilename += '_{}'.format(testLabel)
    if args.cluster:
        outfilename += '_sbatch'
    outfilename += '.sh'

print("Run script created: {}".format(outfilename))
f_run = open(outfilename, 'w')

# shebang
f_run.write("#!/bin/bash\n")
f_run.write("\n")

if args.cluster:
    # Slurm directives
    f_run.write("#SBATCH --gres=gpu:1\n")
    f_run.write("#SBATCH --cpus-per-task=1\n")
    f_run.write("#SBATCH --mem=8000M\n")
    f_run.write("#SBATCH --time=1:00:00\n")
    f_run.write("#SBATCH --output=%N-%j.out\n\n")
    # environment setup
    f_run.write("SUBMIT_DIR=$PWD\n")
    f_run.write("SOURCE_DIR={}\n".format(args.sourceDir))
    f_run.write("OUTPUT_DIR={}\n".format(args.clusterOutdir))
    f_run.write("WORK_DIR=$SLURM_TMPDIR\n")
    f_run.write("cd $WORK_DIR\n\n")
    f_run.write("source $SOURCE_DIR/setup_cedar.sh\n\n")

# samples
f_run.write("###########\n")
f_run.write("# samples\n")
sample_dict = runConfig['samples']
for varname, samplelist in runConfig['samples'].items():
    sample_str = varname+"="+"'"+" ".join(samplelist)+"'"
    f_run.write(sample_str+'\n')
f_run.write('\n')

def get_argument_str(argname, argvalue):
    if isinstance(argvalue, bool):
        if argvalue:
            return '--'+argname
        else:
            return ''
    else:
        return '--'+argname+' '+str(argvalue)

def get_label_str(keyname, argvalue):
    if isinstance(argvalue, bool):
        if argvalue:
            return keyname
        else:
            return 'no'+keyname
    else:
        return keyname+str(argvalue)

def write_options(parConfig_dict):
    options = ['']
    labels = ['']
    for argname, argvalue in parConfig_dict.items():
        if argname.startswith('_'): # reserve for comments
            continue

        options_new = []
        labels_new = []

        for opt, l in zip(options, labels):
            if isinstance(argvalue, list):
                for aval in argvalue:
                    options_new.append(opt+' '+get_argument_str(argname, aval))
                    labels_new.append(l+'_'+get_label_str(argname[:4],aval))
            elif isinstance(argvalue, dict):
                for k, v in argvalue.items():
                    options_new.append(opt+' '+get_argument_str(argname,v))
                    labels_new.append(l+'_'+k)
            else:
                options_new.append(opt+' '+get_argument_str(argname, argvalue))
                labels_new.append(l)

        options = options_new
        labels = labels_new

    return options, labels

executable = os.path.join(args.sourceDir, 'unfold.py')

# tests
for testname, testConfig in runConfig['tests'].items():
    if testname.startswith('_'):
        continue
    if not testConfig.get('enable', False):
        continue

    f_run.write("###########\n")
    f_run.write('# '+testname+'\n')
    run_str = 'python3 {}'.format(executable)
    run_str += ' -d '+testConfig['data']
    run_str += ' -s '+testConfig['signal']
    if 'background' in testConfig:
        run_str += ' -b '+testConfig['background']

    # weight file
    if "unfolded-weights" in testConfig:
        run_str += ' --unfolded-weights '+testConfig['unfolded-weights']

    # reweight data for stress tests
    if "reweight-data" in testConfig:
        run_str += ' --reweight-data '+testConfig['reweight-data']

    # parameters
    parOptions, parLabels = write_options(runConfig['parameters'])
    for opt, pl in zip(parOptions, parLabels):
        f_run.write('# '+pl+'\n')
        outname = 'output_'+testLabel+pl+'_'+testname
        f_run.write(run_str + opt + ' -o ' + outname + '\n')

        # transfer results to the output directory if running on clusters
        if args.cluster:
            f_run.write("cp -r "+outname+" $OUTPUT_DIR/.\n")

f_run.close()
