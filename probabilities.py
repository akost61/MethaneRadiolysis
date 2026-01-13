from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from cross_section import cross_section_calc
import numpy as np




event_code = ['Ion_1', 'Ion_2', 'Ion_3', 'Ion_4', 'Ion_5', 'Ion_6','Ion_7','EIE_1', 'EIE_2', 'EIE_3', 'EA',
          'Nu1', 'Nu2', 'Nu3', 'Nu4', 'Jto3', 'Jto4', 'Ly_a', 'Ly_b', 'Ly_g', 'H_a', 'H_b',
          'H_g', 'H_d', 'CH G-band', 'C3', 'C1', 'C4']

event_names = ['CH₄ + e⁻ -> CH₄⁺ + 2e⁻',
 'CH₄ + e⁻ -> CH₃⁺ + H* +  2e⁻',
 'CH₄ + e⁻ -> CH₂⁺ + H₂ +  2e⁻',
 'CH₄ + e⁻ -> CH₃* + H⁺ +  2e⁻',
 'CH₄ + e⁻ -> CH⁺ + H₂ + H* +  2e⁻',
 'CH₄ + e⁻ -> CH₂* + H₂⁺ + 2e⁻',
 'CH₄ + e⁻ -> C⁺ + 2H₂ + 2e⁻',
 'CH₄ + e⁻ -> CH₃* + H* + e⁻',
 'CH₄ + e⁻ -> CH₂* + H₂ + e⁻',
 'CH₄ + e⁻ -> CH* + H₂ + H* + e⁻',
 'CH₄ + e⁻ -> CH₃* + H⁻',
 'mode v₁',
 'mode v₂',
 'mode v₃',
 'mode v₄',
 'J = 0 to J = 3',
 'J = 0 to J = 4',
 'Ly-α',
 'Ly-β',
 'Ly-γ',
 'H-α',
 'H-β',
 'H-γ',
 'H-δ',
 'CH G-band',
 'C III',
 'C I',
 'C IV']

ionization = event_names[0:7]
radicals = event_names[7:10]
attachment = event_names[10]
reactives = ionization + radicals + [attachment]
molecular_excitation = event_names[11:17]
photon_producing = event_names[17:28]

event_dict = dict(zip(event_code, event_names))

energies = np.logspace(0,5,400)

cross_sections_total = []
for e in energies:
    cross_sections = cross_section_calc(e)
    total = np.sum(cross_sections)
    cross_sections = cross_sections / total
    cross_sections_total.append(cross_sections)



with PdfPages(f"./Probability/Probabilities.pdf") as pdf:
    for i in range(len(event_names)):
        y_values = [cross_sections_total[E][i] for E in range(400)]
        plt.figure()
        plt.plot(energies, y_values)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('eV')
        plt.ylabel('Probability')
        plt.title(f'Probability of {event_names[i]}')
        pdf.savefig()
        plt.close()

all_cross_sections= []
totals = []
for e in energies:
    cross_sections = cross_section_calc(e)
    cross_sections = cross_sections
    table_1_cs = cross_sections[0:11]
    table_2_cs = cross_sections[11:17]
    table_3_cs = cross_sections[17:28]
    table_1_total = np.sum(table_1_cs)
    table_2_total = np.sum(table_2_cs)
    table_3_total = np.sum(table_3_cs)
    total = [table_1_total, table_2_total, table_3_total]
    totals.append(total)
    all_cross_sections.append(cross_sections)

table_1_y_values = [totals[E][0] for E in range(400)]
table_2_y_values = [totals[E][1] for E in range(400)]
table_3_y_values = [totals[E][2] for E in range(400)]
reactives_y_values = [all_cross_sections[E][0:11] for E in range(400)]
molecular_excitation_y_values = [all_cross_sections[E][11:17] for E in range(400)]
photon_producing_y_values = [all_cross_sections[E][17:28] for E in range(400)]

with PdfPages(f"./Probability/Total_Probabilities.pdf") as pdf:
    plt.figure()
    plt.plot(energies, table_1_y_values, label='Reactives')
    plt.plot(energies, table_2_y_values, label='Molecular Excitation')
    plt.plot(energies, table_3_y_values, label='Photon Producing')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Electron Energy (eV)')
    plt.ylabel('Probability')
    plt.legend()
    plt.gca().set_xticks([1, 10, 100, 1000, 10000, 100000])
    plt.gca().set_xticklabels(['1', '10', '100', '1000', '10000', '100000'])
    plt.ylim(bottom=1E-3)
    plt.ylim(top=1)
    pdf.savefig()
    plt.close()

with PdfPages(f"./Probability/Probabilities_Reactives.pdf") as pdf:
    plt.figure(figsize=(10, 6))
    for i in range(len(reactives)):
        plt.plot(energies, [reactives_y_values[E][i] for E in range(400)], label=reactives[i])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Electron Energy (eV)')
        plt.ylabel('Probability')
    plt.plot(energies, table_1_y_values, label='Total Reactives', linewidth=2, color='black')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.gca().set_xticks([1, 10, 100, 1000, 10000, 100000])
    plt.gca().set_xticklabels(['1', '10', '100', '1000', '10000', '100000'])
    plt.ylim(bottom=1E-3)
    plt.ylim(top=1)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

with PdfPages(f"./Probability/Probabilities_Molecular_Excitation.pdf") as pdf:
    plt.figure(figsize=(10, 6))
    for i in range(len(molecular_excitation)):
        plt.plot(energies, [molecular_excitation_y_values[E][i] for E in range(400)], label=molecular_excitation[i])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Electron Energy (eV)')
        plt.ylabel('Probability')
    plt.plot(energies, table_2_y_values, label='Total Molecular Excitation', linewidth=2, color='black')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.gca().set_xticks([1, 10, 100, 1000, 10000, 100000])
    plt.gca().set_xticklabels(['1', '10', '100', '1000', '10000', '100000'])
    plt.ylim(bottom=1E-3)
    plt.ylim(top=1)
    pdf.savefig()
    plt.close()

with PdfPages(f"./Probability/Probabilities_Photon.pdf") as pdf:
    plt.figure(figsize=(10, 6))
    for i in range(len(photon_producing)):
        plt.plot(energies, [photon_producing_y_values[E][i] for E in range(400)], label=photon_producing[i])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Electron Energy (eV)')
        plt.ylabel('Probability')
    plt.plot(energies, table_3_y_values, label='Total Photon Producing', linewidth=2, color='black')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.gca().set_xticks([1, 10, 100, 1000, 10000, 100000])
    plt.gca().set_xticklabels(['1', '10', '100', '1000', '10000', '100000'])
    plt.ylim(bottom=1E-3)
    plt.ylim(top=1)
    pdf.savefig()
    plt.close()