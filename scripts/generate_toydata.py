#!/usr/bin/env python3
import time
from datahandler import DataToy

def generate_toydata(
    nevents,
    filepath_sim,
    filepath_obs,
    mu_sim = 0,
    sigma_sim = 1,
    mu_obs = 0.2,
    sigma_obs = 0.8,
    eff=1,
    acc=1
):

    # Smaring
    epsilon = sigma_sim

    # Gen and Sim
    toy_sim = DataToy()
    print(f"Generating toy data sim: mu={mu_sim}, sigma={sigma_sim}, epsilon={epsilon}, eff={eff}, acc={acc}")
    t_sim_start = time.time()
    toy_sim.generate(nevents=nevents, mu=mu_sim, sigma=sigma_sim, epsilon=epsilon, eff=eff, acc=acc)
    t_sim_done = time.time()
    print(f"Done. {(t_sim_done-t_sim_start):.3f} seconds")

    print(f"Save toy data to {filepath_sim}")
    toy_sim.save_data(filepath_sim)

    # Data and Truth
    toy_obs = DataToy()
    print(f"Generating toy data obs: mu={mu_obs}, sigma={sigma_obs}, epsilon={epsilon}, eff={eff}, acc={acc}")
    t_obs_start = time.time()
    toy_obs.generate(nevents=nevents, mu=mu_obs, sigma=sigma_obs, epsilon=epsilon, eff=eff, acc=acc)
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

    args = parser.parse_args()

    generate_toydata(**vars(args))