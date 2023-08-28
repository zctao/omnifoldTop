import os
import reweight_samples_binned as rwsb
import createRun2Config as r2c

polynomial_deg_d = {
    'th_pt': 3,
    'th_y': 3,
    'tl_pt': 3,
    'tl_y': 4,
    'mtt': 4,
    'ptt': 4,
    'ytt': 5
}

def reweightDataStressBinned(
    sample_dir,
    output_dir,
    observables = ['th_pt','th_y','tl_pt','tl_y','mtt','ptt','ytt'],
    subcampaigns = ["mc16a", "mc16d", "mc16e"]
    ):

    data = r2c.get_samples_data(sample_dir, subcampaigns=subcampaigns)
    signal = r2c.get_samples_signal(sample_dir, subcampaigns=subcampaigns)
    backgrounds = r2c.get_samples_backgrounds(sample_dir, subcampaigns=subcampaigns)

    args_list = ['-t']+data + ['-s']+signal + ['-b']+backgrounds
    args_list += ['-p', '-v']

    # for now
    for obs in observables:
        args_obs = args_list + [
            '--observables', obs, 
            '-o', os.path.join(output_dir, obs), 
            '--polynomial-degree', str(polynomial_deg_d.get(obs, 4))
            ]

        rw_args_obs = rwsb.getArgsParser(args_obs)

        rwsb.reweight_samples_binned(**vars(rw_args_obs))

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

    reweightDataStressBinned(**vars(args))