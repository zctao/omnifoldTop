import reweight_samples as rws
import createRun2Config as r2c

def reweightDataStress(
    sample_dir,
    output_dir,
    observables = ['th_pt','th_y','tl_pt','tl_y','mtt','ptt','ytt'],
    subcampaigns = ["mc16a", "mc16d", "mc16e"]
    ):

    data = r2c.get_samples_data(sample_dir, subcampaigns=subcampaigns)
    signal = r2c.get_samples_signal(sample_dir, subcampaigns=subcampaigns)
    backgrounds = r2c.get_samples_backgrounds(sample_dir, subcampaigns=subcampaigns)

    args_list = ['-t']+data + ['-s']+signal + ['-b']+backgrounds
    args_list += ['--observables'] + observables
    args_list += ['-o', output_dir]
    args_list += ['-n']
    args_list += ['-v']
    args_list += ['-pp']
    args_list += ['-r', 'histogram'] # or 'direct'

    rw_args = rws.getArgsParser(args_list)

    rws.reweight_samples(**vars(rw_args))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("sample_dir", type=str, 
                        help="Top directory of samples")
    parser.add_argument("output_dir", type=str,
                        help="Output directory")
    parser.add_argument('--observables', nargs='+', type=str,
                        default=['th_pt','th_y','tl_pt','tl_y','mtt','ptt','ytt'],
                        help="List of observables used to train the classifier")
    parser.add_argument("-e", "--subcampaigns", nargs='+', choices=["mc16a", "mc16d", "mc16e"], default=["mc16a", "mc16d", "mc16e"])

    args = parser.parse_args()

    reweightDataStress(**vars(args))