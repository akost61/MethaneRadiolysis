from run_simulation import run_simulations, run_generation_simulations
import numpy as np


def main():
    data_1kev, t_e_1kev, EA_1kev = run_simulations(1000, 40000)

    np.save('./Data/data_1kev.npy', data_1kev)
    np.save('./Data/EA_1kev.npy', EA_1kev)
    np.save('./Data/t_e_1kev.npy', t_e_1kev)

    data_20kev, t_e_20kev, EA_20kev = run_simulations(20000, 50000)

    np.save('./Data/data_20kev.npy', data_20kev)
    np.save('./Data/EA_20kev.npy', EA_20kev)
    np.save('./Data/t_e_20kev.npy', t_e_20kev)

    data_40kev, t_e_40kev, EA_40kev = run_simulations(40000, 50000)
    np.save('./Data/data_40kev.npy', data_40kev)
    np.save('./Data/EA_40kev.npy', EA_40kev)
    np.save('./Data/t_e_40kev.npy', t_e_40kev)

    data_60kev, t_e_60kev, EA_60kev = run_simulations(60000, 50000)
    np.save('./Data/data_60kev.npy', data_60kev)
    np.save('./Data/EA_60kev.npy', EA_60kev)
    np.save('./Data/t_e_60kev.npy', t_e_60kev)

    data_80kev, t_e_80kev, EA_80kev = run_simulations(80000, 50000)
    np.save('./Data/data_80kev.npy', data_80kev)
    np.save('./Data/EA_80kev.npy', EA_80kev)
    np.save('./Data/t_e_80kev.npy', t_e_80kev)

    data_100kev, t_e_100kev, EA_100kev = run_simulations(100000, 10000)

    np.save('./Data/data_100kev.npy', data_100kev)
    np.save('./Data/EA_100kev.npy', EA_100kev)
    np.save('./Data/t_e_100kev.npy', t_e_100kev)

    data_gen_100kev, t_e_gen_100kev, EA_gen_100kev = run_generation_simulations(100000, 10000)
    np.save('./Data/data_gen_100kev.npy', data_gen_100kev)
    np.save('./Data/t_e_gen_100kev.npy', t_e_gen_100kev)
    np.save('./Data/EA_gen_100kev.npy', EA_gen_100kev)


    batch_data_100kev, _, _ = run_simulations(100000, 10000)
    np.save('./Data/monte_carlo_100kev.npy', batch_data_100kev)


if __name__ == "__main__":
    main()