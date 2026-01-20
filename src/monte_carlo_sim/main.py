"""
Main Execution Script for Radiolysis Monte Carlo Simulations

This script orchestrates the complete simulation workflow: running Monte Carlo simulation,
processing results, generating convergence diagnostics, and exporting data to Excel. 

This script also tracks events organized by electron generation (primary, secondary, 
tertiary, etc.) to analyze event patterns in radiolysis simulations.

WORKFLOW:
1. Run Monte Carlo simulations at specified incident energy
2. Generate convergence analysis plots
3. Process results into three data views: events, energy, and species
4. Perform energy conservation validation
5. Export results to Excel workbook

6. Run Monte Carlo simulation with generation tracking at specified incident energy
7. Organize event data by generation number (primary = 0; secondary = 1; tertiary = 2; etc.)
8. Calculate total events per generation
9. Export generation-resolved data to Excel

DATA PROCESSING:

create_dataframes(data, electron_attachment_energy):
  - Converts raw simulation data into three analysis perspectives:
    * df_events: Event frequency counts for all 28 event channels
    * df_energy: Energy transfer per event type (Count × delta_k), except electron attachment is set to the total electron attachment energy
    * df_species: Species production counts aggregated across all reactions
 - Note - electron attachment is different from other events because it includes both changes in the potential energy of the molecules, but also the removal of the electron 
        and any excess kinetic energy from the simulation

run_standard_simulation_report(incident_energy, total_simulations):
  - Runs the standard Monte Carlo simulation
  - Generates convergence plots
  - Processes simulation results into three dataframes (events, energy, species)
  - Validates energy conservation: input energy = transferred energy + terminating energy
  - Exports results to three-sheet Excel workbook:
    * Event_data: Raw counts for the 28 different events/channels
    * Energy_data: Energy distribution for each of the 28 different events/channels
    * Species_data: Net species production with stoichiometry
  - File naming: {energy_in_keV}keV_{num_sims}_simulations_results.xlsx

run_generational_simulation_report(incident_energy, total_simulations):
  - Runs Monte Carlo simulation with generation tracking
  - Returns event counts organized by generation rather than by trajectory
  - Each row represents a generation level (0=primary, 1=secondary, etc.)
  - Each column represents one of the 28 collision event types
  - Adds 'Total' column summing all events per generation
  - Exports single-sheet Excel workbook:
    * {energy_in_keV}keV_{total_simulations}_generational_results.xlsx

main():
  - Configurable parameters: incident_energy (eV), total_simulations
  - Orchestrates both standard and generation-resolved simulation runs
  - Calls run_standard_simulation() and run_generational_simulation()
  - Default configuration: 100 keV incident electrons, 10,000 simulations

ENERGY CONSERVATION CHECK:
Total input = incident_energy × total_simulations
Total output = Σ(event_count × delta_k) + terminating_energy + attachment_energy
These should match within numerical precision.
Default configuration: 100 keV incident electrons, 10,000  simulations.
"""

import matplotlib.pyplot as plt
import pandas as pd
from monte_carlo_sim.simulation.run_simulation import run_simulations, run_generation_simulations
from monte_carlo_sim.plotting.convergence import create_convergence_graphs
from monte_carlo_sim.simulation.constants import event_names, delta_k, reaction_produced




def create_dataframes(data, electron_attachment_energy):
    combined_data = data.sum(axis=0)

    df_events = pd.DataFrame({
        'Event': event_names,
        'Count': combined_data
    })

    df_energy = df_events.copy()
    df_energy['delta k'] = delta_k
    df_energy['Energy Transferred'] = df_energy['Count'] * df_energy['delta k']

    ea_mask = df_energy['Event'] == 'CH₄ + e⁻ -> CH₃* + H⁻'
    df_energy.loc[ea_mask, 'Energy Transferred'] = electron_attachment_energy.sum()
    df_energy = df_energy[['Event', 'Energy Transferred']]

    species_rows = []
    for i, event in enumerate(event_names):
        if event in reaction_produced:
            count = combined_data[i]
            for species, stoich in reaction_produced[event].items():
                species_rows.append({
                    'Species': species,
                    'Count': count * stoich
                })

    df_species = pd.DataFrame(species_rows)
    df_species = df_species.groupby('Species', as_index=True)['Count'].sum().to_frame()

    return df_events, df_energy, df_species


def run_standard_simulation_report(incident_energy: float, total_simulations: int):
    data, terminating_energy, electron_attachment_energy = run_simulations(
        incident_energy, total_simulations
    )

    create_convergence_graphs(data, incident_energy, total_simulations)

    df_events, df_energy, df_species = create_dataframes(data, electron_attachment_energy)

    total_energy = terminating_energy.sum() + df_energy['Energy Transferred'].sum()
    print(f"Total energy input: {incident_energy * total_simulations}")
    print(f"Total energy used: {df_energy['Energy Transferred'].sum()}")
    print(f"Total energy lost at termination: {terminating_energy.sum()}")
    print(f"Total energy used in model: {total_energy}")

    file_name = f"./results/{int(incident_energy/1000)}keV_{total_simulations}_simulations_results.xlsx"
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        df_events.to_excel(writer, sheet_name='Event_data', index=False)
        df_energy.to_excel(writer, sheet_name='Energy_data', index=False)
        df_species.to_excel(writer, sheet_name='Species_data', index=True)


def run_generational_simulation_report(incident_energy: float, total_simulations: int):
    data, _, _ = run_generation_simulations(incident_energy, total_simulations)

    df_events = pd.DataFrame(data, columns=event_names)
    df_events['Total'] = df_events.sum(axis=1)

    file_name = f"./results/{int(incident_energy/1000)}keV_{total_simulations}_generational_results.xlsx"
    df_events.to_excel(file_name, sheet_name='generation_data', index=True)


def main():
    incident_energy = 100_000  # in eV
    total_simulations = 10_000

    run_standard_simulation_report(incident_energy, total_simulations)
    run_generational_simulation_report(incident_energy, total_simulations)


if __name__ == "__main__":
    main()
