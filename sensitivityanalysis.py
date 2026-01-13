from run_simulation import run_simulations, run_generation_simulations
from cross_section import event_names
from monte_carlo import combine_data
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

def get_sensitivity_results(eV, total_sims, manipulated_variable):
    return run_simulations(eV, total_sims, manipulated=manipulated_variable)




def create_pdf_sensitivity_analysis(eV, sensitivity_sims, batch_base_data, base_sims, manipulated_variable):
    results_sensitivity, t_e_sensitivity, EA_sensitivity = get_sensitivity_results(eV, sensitivity_sims, manipulated_variable)
    results_sensitivity = results_sensitivity.sum(axis=0) / sensitivity_sims
    x_values = np.arange(1, base_sims + 1, 1)
    mean_values = batch_base_data[-1]
    with PdfPages(f'./SensitivityAnalysis/sensitivity_analysis_{event_names[manipulated_variable]}.pdf') as pdf:
        for i in range(len(event_names)):
            ax = plt.gca()
            event = event_names[i]
            mean_value = mean_values[i]
            plus_10 = mean_value * 1.10
            minus_10 = mean_value *0.90
            plus_2 = mean_value * 1.02
            minus_2 = mean_value * 0.98
            plus_5 = mean_value * 1.05
            minus_5 = mean_value *0.95

            plt.figure(figsize=(8, 6))
            plt.title(f'Primary Electron at {int(eV/1000)}keV For {event} Mean Count vs Simulations Ran')
            plt.xlabel('Number of Electrons')
            plt.ylabel('Mean Occurrence')
            plt.axhline(y=plus_2, color='lightgreen', linestyle='--', label='+2%')
            plt.axhline(y=minus_2, color='lightcoral', linestyle='--', label='-2%')
            plt.axhline(y=plus_5, color='limegreen', linestyle='--', label='+5%')
            plt.axhline(y=minus_5, color='tomato', linestyle='--', label='-5%')            
            plt.axhline(y=plus_10, color='green', linestyle='--', label='+10%')
            plt.axhline(y=minus_10, color='red', linestyle='--', label='-10%')
            plt.plot(sensitivity_sims, results_sensitivity[i], marker='o', color='orange', label = event_names[manipulated_variable])
            plt.plot(x_values, batch_base_data[:, i])
            ax.set_xticks(np.linspace(0, base_sims, 6))
            plt.legend()
            pdf.savefig()
            plt.close()

#Ran for CH4+, CH3+, CH3*, mode4, j=3, j=4: index [0, 1, 7, 14, 15, 16] 10% increase in cross section results

def main():
    manipulated_variables = [0, 1, 7, 14, 15, 16]
    data_100kev = np.load('./Data/data_100kev.npy')
    data_100kev = combine_data(data_100kev)
    for v in manipulated_variables:
        create_pdf_sensitivity_analysis(100000, 5000, data_100kev, 10000, v)

    return


if __name__ == "__main__":
    main()