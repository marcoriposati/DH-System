from Dataset_2023 import * 
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def load_model_data():

    folder_path = folder_path = '../02. Output'
    file_path = os.path.join(folder_path, 'model_data.pkl') # Name of your output file
    # Load the data from the file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


# Load the data
model_data = load_model_data()

heat = model_data['heat_production']
TES_Charge = model_data['TES_charge']
TES_Discharge = model_data['TES_discharge']
TES_Level = model_data['TES_level']


T = list(range((n_weeks-1)*h))



#------------------------------------------- TES PLOT SECTION -----------------------------------------------------------------------------------------------------------------------------------------------

show_TES_plot = True 
if show_TES_plot:

    # Create a figure for each TES system
    for s in S:
        tes_level = [TES_Level[(s, t)] for t in T]
        tes_charge = [TES_Charge[(s, t)] for t in T]
        tes_discharge = [TES_Discharge[(s, t)] for t in T]

        # Create a figure and a set of subplots
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))

        # Plotting TES energy level on the first subplot
        ax[0].plot(T, tes_level, label='TES Energy Level', color='blue', linewidth=2)
        ax[0].fill_between(T, 0, tes_level, color='blue', alpha=0.1)
        ax[0].set_title(f'{s} Energy Level Over Time')
        ax[0].set_ylabel('Energy Level')
        ax[0].grid(True)

        # Plotting TES charging and discharging rates on the second subplot
        ax[1].plot(T, tes_charge, label='TES Charging Rate', color='green', linestyle='--')
        ax[1].plot(T, tes_discharge, label='TES Discharging Rate', color='red', linestyle='--')
        ax[1].set_title(f'{s} Charging and Discharging Rates Over Time')
        ax[1].set_xlabel('Time Period')
        ax[1].set_ylabel('Rate')
        ax[1].legend()
        ax[1].grid(True)

        # Adjust the layout
        plt.tight_layout()
        plt.show()


#------------------------------------------- ENERGY MIX PLOT SECTION -----------------------------------------------------------------------------------------------------------------------------------------------


# Define a colormap for all units and storage types
color_mapping = {
        'Amager_Res': 'lightgreen',
        'Roskilde': 'green',
        'Vestforbrænding': 'darkgreen',
        'Vestforbrænding_bp': 'yellow',
        'Amager1': 'lightblue',
        'Amager4': 'red',
        'Amager4_bp': 'black',
        'Avedøre1': 'royalblue',
        'Avedøre2S': 'blue',
        'Avedøre2W': 'mediumblue',
        'Køge': 'darkblue',
        'Køge_bp': 'grey',
        'Avedøre2N': 'lightsalmon',
        'DTU': 'darkred',
        'El_boiler': 'blueviolet',
        'Heat_pump': 'violet',
        'NG_boiler': 'orange',
        'TES_Discharge': 'brown'
    }

# Extend the U_plot list with 'TES_Charge'
U_plot_names = ['Amager_Res', 'Roskilde', 'Vestforbrænding', 'Vestforbrænding_bp', 'Amager1', 'Amager4', 'Amager4_bp', 'Avedøre1', 'Avedøre2S', 'Avedøre2W', 'Køge', 'Køge_bp', 'Avedøre2N', 'DTU', 'El_boiler', 'Heat_pump', 'NG_boiler', 'TES_Discharge']

# Assign a special index or marker for 'TES_Discharge'
special_indices = {'TES_Discharge': 22}

# Transform the names in U_plot to their corresponding indices
U_plot = [u_index[name] if name in u_index else special_indices[name] for name in U_plot_names]


# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# Plot for the main stacked bar chart (heat production) in the first subplot
ax1.set_title("Total Heat Production for the First 4 Week of 2023", fontsize=16, fontweight='bold')
bottom = np.zeros(len(T))  # Initialize stacking


# Calculate total TES charge for each time step
total_TES_charge_for_t = [sum(TES_Charge[(s, t)]  for s in S) for t in T]
total_TES_discharge_for_t = [sum(TES_Discharge[(s, t)]  for s in S) for t in T]


# Stacked bar chart for heat production
for u in U_plot:
    unit_name = index_u[u] if u in index_u else 'TES_Discharge'  # Handle special case for 'TES_Discharge'
    heat_values = total_TES_discharge_for_t if unit_name == 'TES_Discharge' else [heat[(u, t)] for t in T]
    
    ax1.bar(T, heat_values, label=unit_name, color=color_mapping.get(unit_name, 'grey'), bottom=bottom, alpha=0.8, edgecolor='none')
    bottom = [sum(x) for x in zip(bottom, heat_values)]
    

# Line chart for heat demand
heat_demand_values = [heat_demand[t] for t in T]
ax1.plot(T, heat_demand_values, color='red', label="Heat Demand", linestyle='--')

# Customize the main plot
ax1.set_ylabel("Heat Production [MW]", fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.legend(title="Unit Names", fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

# Plot the total TES charge for all units at each timestep in the second subplot
ax2.bar(T, total_TES_charge_for_t, color='gold', label='Total TES Charge')
ax2.set_title("Total TES Charge Over Time", fontsize=16, fontweight='bold')
ax2.set_xlabel("Hours", fontsize=12)
ax2.set_ylabel("TES Charge [MW]", fontsize=12)

# Customize the TES charge plot
ax2.grid(axis='y', linestyle='--', alpha=0.6)
ax2.legend()

# Adjust layout and remove spines
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  # Adjust horizontal space between subplots
for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Show the plot
plt.show()
