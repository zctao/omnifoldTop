#!/usr/bin/env python3
import time
from datahandler import DataToy

def generate_toydata(
    nevents,
    filepath_sim,
    filepath_obs,
    varnames,
    mu_sim,
    sigma_sim,
    mu_obs,
    sigma_obs,
    cov_meas,
    eff,
    acc
    ):

    # Gen and Sim
    toy_sim = DataToy()
    print(f"Generating toy data sim: mu={mu_sim}, sigma={sigma_sim}, cov_meas={cov_meas}, eff={eff}, acc={acc}")
    t_sim_start = time.time()
    toy_sim.generate(nevents=nevents, varnames=varnames, mu=mu_sim, sigma=sigma_sim, covariance=cov_meas, eff=eff, acc=acc)
    t_sim_done = time.time()
    print(f"Done. {(t_sim_done-t_sim_start):.3f} seconds")

    print(f"Save toy data to {filepath_sim}")
    toy_sim.save_data(filepath_sim)

    # Data and Truth
    toy_obs = DataToy()
    print(f"Generating toy data obs: mu={mu_obs}, sigma={sigma_obs}, cov_meas={cov_meas}, eff={eff}, acc={acc}")
    t_obs_start = time.time()
    toy_obs.generate(nevents=nevents, varnames=varnames, mu=mu_obs, sigma=sigma_obs, covariance=cov_meas, eff=eff, acc=acc)
    t_obs_done = time.time()
    print(f"Done. {(t_obs_done-t_obs_start):.3f} seconds")

    print(f"Save toy data to {filepath_obs}")
    toy_obs.save_data(filepath_obs)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--nevents', type=int, default=100,  
                        help="Number of events to generate")
    parser.add_argument('-s', '--filepath-sim', type=str, default="toydata_sim.npz",
                        help="File path to store the generated toy data as simulated sample")
    parser.add_argument('-d', '--filepath-obs', type=str, default="toydata_obs.npz",
                        help="File path to store the generated toy data as observed sample")
    parser.add_argument('-e', '--eff', type=float, default=1., 
                        help="Efficiency")
    parser.add_argument('-a', '--acc', type=float, default=1., 
                        help="Acceptance")
    parser.add_argument('-p', '--preset', choices=['1d', '2d_ind', '2d_cor'], default='1d')

    args = parser.parse_args()

    if args.preset == '1d':
        generate_toydata(
            nevents = args.nevents,
            filepath_sim = args.filepath_sim,
            filepath_obs = args.filepath_obs,
            eff = args.eff,
            acc = args.acc,
            varnames = ['x'],
            mu_sim = 0.,
            sigma_sim = 1.,
            mu_obs = 0.2,
            sigma_obs = 0.8,
            cov_meas = 1.
        )
    elif args.preset == '2d_ind':
        generate_toydata(
            nevents = args.nevents,
            filepath_sim = args.filepath_sim,
            filepath_obs = args.filepath_obs,
            eff = args.eff,
            acc = args.acc,
            varnames = ['x', 'y'],
            mu_sim = [0., 0.],
            sigma_sim = [1., 1.],
            mu_obs = [0.2, -0.3],
            sigma_obs = [0.8, 0.9],
            cov_meas = [[1.,0.],[0.,1.]]
        )
    elif args.preset == '2d_cor':
        generate_toydata(
            nevents = args.nevents,
            filepath_sim = args.filepath_sim,
            filepath_obs = args.filepath_obs,
            eff = args.eff,
            acc = args.acc,
            varnames = ['x', 'y'],
            mu_sim = [0., 0.],
            sigma_sim = [1., 1.],
            mu_obs = [0.2, -0.3],
            sigma_obs = [0.8, 0.9],
            cov_meas = [[1.,-0.5],[-0.5,1.]]
        )