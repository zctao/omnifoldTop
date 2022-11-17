#!/usr/bin/env python3
import time
from datahandler import DataToy

def generate_toydata(
    nevents,
    filepath_sim,
    filepath_obs,
    varnames,
    mu_sim,
    cov_sim,
    mu_obs,
    cov_obs,
    cov_meas,
    eff,
    acc
    ):

    # Gen and Sim
    toy_sim = DataToy()
    print(f"Generating toy data sim: mu={mu_sim}, cov={cov_sim}, cov_meas={cov_meas}, eff={eff}, acc={acc}")
    t_sim_start = time.time()
    toy_sim.generate(nevents=nevents, varnames=varnames, mean=mu_sim, covariance=cov_sim, covariance_meas=cov_meas, eff=eff, acc=acc)
    t_sim_done = time.time()
    print(f"Done. {(t_sim_done-t_sim_start):.3f} seconds")

    print(f"Save toy data to {filepath_sim}")
    toy_sim.save_data(filepath_sim)

    # Data and Truth
    print(f"Generating toy data obs: mu={mu_obs}, cov={cov_obs}, cov_meas={cov_meas}, eff={eff}, acc={acc}")
    toy_obs = DataToy()
    t_obs_start = time.time()
    toy_obs.generate(nevents=nevents, varnames=varnames, mean=mu_obs, covariance=cov_obs, covariance_meas=cov_meas, eff=eff, acc=acc)
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
    parser.add_argument('-p', '--preset', choices=['1d', '2d_ind', '2d_cor', '2d_meas_cor', '2d_true_cor', '4d_ind', '4d_cor', '4d_meas_cor', '4d_true_cor', '2d_true_cor_diff', '2d_cor_diff'], default='1d')

    args = parser.parse_args()

    # set parameters
    if args.preset == '1d':
        varnames = ['x']
        mu_sim = 0.
        cov_sim = 1.
        mu_obs = 0.2
        cov_obs = 0.64
        cov_meas = 1.
    elif args.preset == '2d_ind':
        varnames = ['x', 'y']
        mu_sim = [0., 0.]
        cov_sim = [[1.,0.], [0.,1.]]
        mu_obs = [0.2, -0.3]
        cov_obs = [[0.64,0.], [0.,0.81]]
        cov_meas = [[1.,0.],[0.,1.]]
    elif args.preset == '2d_meas_cor':
        varnames = ['x', 'y']
        mu_sim = [0., 0.]
        cov_sim = [[1.,0.], [0.,1.]]
        mu_obs = [0.2, -0.3]
        cov_obs = [[0.64,0.], [0.,0.81]]
        cov_meas = [[1.,-0.5],[-0.5,1.]]
    elif args.preset == '2d_true_cor':
        varnames = ['x', 'y']
        mu_sim = [0., 0.]
        cov_sim = [[1.,-0.5], [-0.5,1.]]
        mu_obs = [0.2, -0.3]
        cov_obs = [[0.64,-0.36], [-0.36,0.81]] # same correlation as sim.
        cov_meas = [[1.,0.],[0.,1.]]
    elif args.preset == '2d_true_cor_diff':
        varnames = ['x', 'y']
        mu_sim = [0., 0.]
        cov_sim = [[1.,-0.1], [-0.1,1.]]
        mu_obs = [0.2, -0.3]
        cov_obs = [[0.64,-0.36], [-0.36,0.81]] # different correlation as sim.
        cov_meas = [[1.,0.],[0.,1.]]
    elif args.preset == '2d_cor':
        varnames = ['x', 'y']
        mu_sim = [0., 0.]
        cov_sim = [[1.,-0.5], [-0.5,1.]]
        mu_obs = [0.2, -0.3]
        cov_obs = [[0.64,-0.36], [-0.36,0.81]] # same correlation as sim. Use a slightly different correlation instead?
        cov_meas = [[1.,0.5],[0.5,1.]]
    elif args.preset == '2d_cor_diff':
        varnames = ['x', 'y']
        mu_sim = [0., 0.]
        cov_sim = [[1.,-0.1], [-0.1,1.]]
        mu_obs = [0.2, -0.3]
        cov_obs = [[0.64,-0.36], [-0.36,0.81]] # same correlation as sim. Use a slightly different correlation instead?
        cov_meas = [[1.,0.5],[0.5,1.]]
    elif args.preset == '4d_ind':
        varnames = ['x', 'y', 'z', 'v']
        mu_sim = [0., 0., 0., 0.]
        cov_sim = [[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.], [0.,0.,0.,1.]]
        mu_obs = [0.2, -0.3, -0.5, 1.0]
        cov_obs = [[0.64,0.,0.,0.], [0.,0.81,0.,0.], [0.,0.,1.21,0.], [0.,0.,0.,2.25]]
        cov_meas = [[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.], [0.,0.,0.,1.]]
    elif args.preset == '4d_meas_cor':
        varnames = ['x', 'y', 'z', 'v']
        mu_sim = [0., 0., 0., 0.]
        cov_sim = [[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.], [0.,0.,0.,1.]]
        mu_obs = [0.2, -0.3, -0.5, 1.0]
        cov_obs = [[0.64,0.,0.,0.], [0.,0.81,0.,0.], [0.,0.,1.21,0.], [0.,0.,0.,2.25]]
        cov_meas = [[1.,-0.5,0.5,0.], [-0.5,1.,0.,0.], [0.5,0.,1.,0.], [0.,0.,0.,1.]]
    elif args.preset == '4d_true_cor':
        varnames = ['x', 'y', 'z', 'v']
        mu_sim = [0., 0., 0., 0.]
        cov_sim = [[1.,-0.5,0.5,0.], [-0.5,1.,0.,0.], [0.5,0.,1.,0.], [0.,0.,0.,1.]]
        mu_obs = [0.2, -0.3, -0.5, 1.0]
        cov_obs = [[0.64,-0.36,0.44,0.], [-0.36,0.81,0.,0.], [0.44,0.,1.21,0.], [0.,0.,0.,2.25]]
        cov_meas = [[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.], [0.,0.,0.,1.]]
    elif args.preset == '4d_cor':
        varnames = ['x', 'y', 'z', 'v']
        mu_sim = [0., 0., 0., 0.]
        cov_sim = [[1.,-0.5,0.5,0.], [-0.5,1.,0.,0.], [0.5,0.,1.,0.], [0.,0.,0.,1.]]
        mu_obs = [0.2, -0.3, -0.5, 1.0]
        cov_obs = [[0.64,-0.36,0.44,0.], [-0.36,0.81,0.,0.], [0.44,0.,1.21,0.], [0.,0.,0.,2.25]]
        cov_meas = [[1.,0.5,-0.5,0.], [0.5,1.,0.,0.], [-0.5,0.,1.,0.], [0.,0.,0.,1.]]

    generate_toydata(
        nevents = args.nevents,
        filepath_sim = args.filepath_sim,
        filepath_obs = args.filepath_obs,
        eff = args.eff,
        acc = args.acc,
        varnames = varnames,
        mu_sim = mu_sim,
        cov_sim = cov_sim,
        mu_obs = mu_obs,
        cov_obs = cov_obs,
        cov_meas = cov_meas
    )