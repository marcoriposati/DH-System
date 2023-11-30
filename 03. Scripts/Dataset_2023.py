import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from tabulate import tabulate

# Define for how many weeks the model should be run
n_weeks = 4                                                                                    # Number of weeks investigated
h = 168   

# HEAT DEMAND
heat_demand_file = '../01. Input/01_Heat_Demand_2023.csv'                                                    # Defines the path of Heat Demand   
heat_demand_df = pd.read_csv(heat_demand_file, parse_dates=['Date'], index_col='Timestep')      # Read the CSV and set 'Timestep' as the index  # Creates dataframe with hourly Heat Demand in MWh
heat_demand = heat_demand_df.Heat_demand.values                                                             # Creates a series with hourly Heat Demand in MWh

# ELECTRICITY SPOT PRICE
el_price_file = '../01. Input/02_El_spot_price_2023.csv'                                                     # Defines the path of El_spot_price  
el_price_df = pd.read_csv(el_price_file, parse_dates=['Date'], index_col='Timestep')            # Read the CSV and set 'Timestep' as the index  # Creates dataframe with hourly electricty spot prices
el_price = el_price_df.El_price.values                                                                         # Creates a series with hourly electricty spot prices


# FINANCIAL PARAMETERS                                                                                     # Number of hours investigated
n_period = 8760/h                                                                              # Number of periods in a year
r_annual= 0.04                                                                                  # Annual discount rate
r_period = 1 + r_annual**(1/n_period) - 1                                                              # Periodic discount rate



#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°     PLANTS     °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

#PLANTS DATAFRAMES  

CHP_df = pd.read_excel('../01. Input/03_Units_Plants_Data.xlsx', sheet_name='CHP')                   # Read the Excel file and set 'PlantName' as the index 
El_cons_df = pd.read_excel('../01. Input/03_Units_Plants_Data.xlsx', sheet_name='El_cons')           # Read the Excel file and set 'PlantName' as the index
NG_boiler_df = pd.read_excel('../01. Input/03_Units_Plants_Data.xlsx', sheet_name='NG_boiler')       # Read the Excel file and set 'PlantName' as the index
TES_df = pd.read_excel('../01. Input/03_Units_Plants_Data.xlsx', sheet_name='TES')                   # Read the Excel file and set 'PlantName' as the index

# Combine the DataFrames for all plant types into a single DataFrame 
Units = pd.concat([CHP_df, El_cons_df, NG_boiler_df], ignore_index=True)

# For this "Units" Dataframe convert all NaN values to 0
Units = Units.fillna(0)



#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°     FUEL/EMISSION PRICES for 2023    °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

# FUELS
wood_pellets_c_price = 40.7                             # Unit = EUR / MWh
wood_pellets_c_tax = 1.55                                  # Unit = €/MWh-f

wood_chips_c_price = 30.5                                # Unit = EUR / MWh
wood_chips_c_tax = 0                                    # Unit = €/MWh-f

natural_gas_c_price = 38.4                              # Unit = EUR / MWh 17.1
natural_gas_c_tax = 37.10                                # Unit = €/MWh-f

natural_gas_d_price = 41.1                             # Unit = EUR / MWh
natural_gas_d_tax = 37.10                                  # Unit = €/MWh-f

straw_c_price = 26                                     # Unit = EUR / MWh
straw_c_tax= 1.17                                       # Unit = €/MWh-f

waste_d_price = -18                               # Unit = EUR / MWh
waste_d_tax = 25.86                               # Unit = €/MWh-f

electricity_d_tax = 36                            # Unit = €/MWh-e






# Create dictionaries for fuel prices and taxes
fuel_prices = {'wood_pellets': wood_pellets_c_price, 'wood_chips': wood_chips_c_price, 'natural_gas_c': natural_gas_c_price, 'natural_gas_d': natural_gas_d_price, 'straw': straw_c_price, 'waste': waste_d_price}
fuel_taxes = {'wood_pellets': wood_pellets_c_tax, 'wood_chips': wood_chips_c_tax, 'natural_gas_c': natural_gas_c_tax, 'natural_gas_d': natural_gas_d_tax, 'straw': straw_c_tax, 'waste': waste_d_tax}

#Put these 2 dictionaries together in a single dictionary
fuels = {'fuel_prices': fuel_prices, 'fuel_taxes': fuel_taxes}  


#EMISSIONS
CO2_price = 0.03                                     # Unit = €/kg
SO2_price = 1.7                                          # Unit = €/kg
NOx_price = 1.6                                         # Unit = €/kg
PM_price  = 8                                          # Unit = €/kg

# Put emission prices in a dictionary
emission_prices = {'CO2': CO2_price, 'SO2': SO2_price, 'NOx': NOx_price, 'PM': PM_price}

#QUOTAS

CO2_quota = 0.085                                  # Unit = €/kg  #0.085



#--------------------------------------------------------------------------------------------------------------------------------------------------

# Maps unit names to indices
u_index = {name: idx for idx, name in enumerate(Units['PlantName'])}
s_index = {name: idx for idx, name in enumerate(TES_df['PlantName'])}

index_u = {idx: name for name, idx in u_index.items()}

U = list(range(len(Units['PlantName'])))
S = list(range(len(TES_df['PlantName'])))             # Set of TES storages S(s

# Precompute parameters into arrays for fast access in the model

max_H_cap_array = np.zeros(len(U))
max_El_cap_array = np.zeros(len(U))
min_cap_array = np.zeros(len(U))
HPR_array = np.zeros(len(U))
CO2_array = np.zeros(len(U))
El_eff_array = np.zeros(len(U))
H_eff_array = np.zeros(len(U))
Up_time_array = np.zeros(len(U))
Down_time_array = np.zeros(len(U))
Start_up_cost_array = np.zeros(len(U))
Var_OM_array = np.zeros(len(U))
Fix_OM_array = np.zeros(len(U))
CAPEX_array = np.zeros(len(U))
Lifetime_array = np.zeros(len(U))
Fuel_type_array = np.zeros(len(U))
TES_Charging_cap_array = np.zeros(len(S))
TES_Cap_array = np.zeros(len(S))
TES_Max_Cycles_24_array = np.zeros(len(S))
TES_Capex_array = np.zeros(len(S))
TES_Lifetime_array = np.zeros(len(S))

# Populate the arrays
for name, cap in Units.set_index('PlantName')['H_cap'].items():
    max_H_cap_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['El_cap'].items():
    max_El_cap_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['Minimum_cap'].items():
    min_cap_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['HPR'].items():
    HPR_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['CO2'].items():
    CO2_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['El_eff'].items():
    El_eff_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['H_eff'].items():
    H_eff_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['Up_time'].items():
    Up_time_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['Down_time'].items():
    Down_time_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['Start-up_cost'].items():
    Start_up_cost_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['Var. O&M'].items():
    Var_OM_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['Fix. O&M'].items():
    Fix_OM_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['CAPEX'].items():
    CAPEX_array[u_index[name]] = cap

for name, cap in Units.set_index('PlantName')['Lifetime'].items():
    Lifetime_array[u_index[name]] = cap

Fuel_type_dict = {u_index[name]: cap for name, cap in Units.set_index('PlantName')['Fuel_type'].items()}

for name, cap in TES_df.set_index('PlantName')['Charging_cap'].items():
    TES_Charging_cap_array[s_index[name]] = cap


for name, cap in TES_df.set_index('PlantName')['Cap'].items():
    TES_Cap_array[s_index[name]] = cap

for name, cap in TES_df.set_index('PlantName')['Max_Cycles_24'].items():
    TES_Max_Cycles_24_array[s_index[name]] = cap

for name, cap in TES_df.set_index('PlantName')['CAPEX'].items():
    TES_Capex_array[s_index[name]] = cap

for name, cap in TES_df.set_index('PlantName')['Lifetime'].items():
    TES_Lifetime_array[s_index[name]] = cap






# Precompute unit types
el_cons_unit = set(u_index[name] for name in Units['PlantName'] if name in ['El_boiler', 'Heat_pump'])
heat_pumps = next(u_index[name] for name in Units['PlantName'] if name == "Heat_pump")
el_boilers = next(u_index[name] for name in Units['PlantName'] if name == "El_boiler")
ng_boilers = next(u_index[name] for name in Units['PlantName'] if name =="NG_boiler")
CHP_normal = set(u_index[name] for name in Units['PlantName'] if name in ['Amager1', 'Amager4', 'Avedøre1', 'Avedøre2N', 'Avedøre2S', 'Avedøre2W', 'DTU', 'Køge', 'Amager_Res', 'Roskilde', 'Vestforbrænding'])
CHP_all = set(u_index[name] for name in Units['PlantName'] if name in ['Amager1', 'Amager4', 'Amager4_bp', 'Avedøre1', 'Avedøre2N', 'Avedøre2S', 'Avedøre2W', 'DTU', 'Køge', 'Køge_bp', 'Amager_Res', 'Roskilde', 'Vestforbrænding', 'Vestforbrænding_bp'])
ALL_no_bypass = set(u_index[name] for name in Units['PlantName'] if name in ['Amager1', 'Amager4', 'Avedøre1', 'Avedøre2N', 'Avedøre2S', 'Avedøre2W', 'DTU', 'Køge', 'Amager_Res', 'Roskilde', 'Vestforbrænding', 'Heat_pump'])
All_fuel = set(u_index[name] for name in Units['PlantName'] if name in ['Amager1', 'Amager4', 'Amager4_bp', 'Avedøre1', 'Avedøre2N', 'Avedøre2S', 'Avedøre2W', 'DTU', 'Køge', 'Køge_bp', 'Amager_Res', 'Roskilde', 'Vestforbrænding', 'Vestforbrænding_bp','NG_boiler'])

amager4_bp_unit = next(u_index[name] for name in Units['PlantName'] if name == 'Amager4_bp')
koge_bp_unit = next(u_index[name] for name in Units['PlantName'] if name  == 'Køge_bp')
vestforbrænding_bp_unit = next(u_index[name] for name in Units['PlantName'] if name == 'Vestforbrænding_bp')
no_bp_units = (u_index[name] for name in Units['PlantName'] if u_index[name] not in amager4_bp_unit and u_index[name] not in koge_bp_unit and u_index[name] not in vestforbrænding_bp_unit)
