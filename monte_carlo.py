import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cross_section import event_names
from run_simulation import run_simulations
import numpy as np

def combine_data(simulation_results):
    cumsum = np.cumsum(simulation_results, axis=0)
    counts = np.arange(1, len(simulation_results) + 1).reshape(-1, 1)
    return cumsum / counts


def create_pdf(batch_data, initial_eV, total_simulations):
    x_values = np.arange(1, total_simulations + 1, 1)
    mean_data = batch_data[-1]
    with PdfPages(f"./MonteCarloSimulation/{int(initial_eV/1000)}keV_mean_sim.pdf") as pdf:

        for i in range(len(event_names)):
            ax = plt.gca()
            mean_value = mean_data[i]
            plus_2 = mean_value * 1.02
            minus_2 = mean_value * 0.98
            
            plt.figure(figsize=(8, 6))
            plt.title(f'Initial Electron at {int(initial_eV/1000)}keV For {event_names[i]} Mean Occurance vs Simulations Ran')
            plt.xlabel('Simulation Runs')
            plt.ylabel('Mean Occurance')
            plt.axhline(y=plus_2, color='green', linestyle='--', label='+2%')
            plt.axhline(y=minus_2, color='red', linestyle='--', label='-2%')
            plt.plot(x_values, batch_data[:, i])
            ax.set_xticks(np.linspace(0, total_simulations, 6))
            plt.legend()
            pdf.savefig()
            plt.close()

def main():
    data_1kev, _, _ = run_simulations(1000, 1000000)
    data_1kev = combine_data(data_1kev)
    create_pdf(data_1kev, 1000, 1000000)
    data_20kev, _, _ = run_simulations(20000, 100000)
    data_20kev = combine_data(data_20kev)
    create_pdf(data_20kev, 20000, 100000)
    data_60kev, _, _ = run_simulations(60000, 100000)
    data_60kev = combine_data(data_60kev)
    create_pdf(data_60kev, 60000, 100000)
    data_100kev, _, _ = run_simulations(100000, 100000)
    data_100kev = combine_data(data_100kev)
    create_pdf(data_100kev, 100000, 100000)

if __name__ == "__main__":
    main()