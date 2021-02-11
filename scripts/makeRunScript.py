#!/usr/bin/env python3
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('configFiles', nargs='*',
                    help='config files for running tests')
parser.add_argument('-b', '--base-config', dest='baseConfig',
                    default='configs/run/basic_tests.json')
parser.add_argument('-o', '--output', type=str,
                    help='output shell script file name')

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
elif testLabel:
    outfilename = 'runtests_{}.sh'.format(testLabel)
else:
    outfilename = 'runtests.sh'

print("Run script created: {}".format(outfilename))
f_run = open(outfilename, 'w')

# shebang
f_run.write("#!/bin/bash\n")
f_run.write("\n")

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

# tests
for testname, testConfig in runConfig['tests'].items():
    if testname.startswith('_'):
        continue
    if not testConfig.get('enable', False):
        continue

    f_run.write("###########\n")
    f_run.write('# '+testname+'\n')
    run_str = 'python3 unfold.py'
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
        out_str = ' -o ./output_'+testLabel+pl+'_'+testname
        f_run.write(run_str + opt + out_str + '\n')

f_run.close()
