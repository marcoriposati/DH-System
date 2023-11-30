# Import Dataset_2023 to get the data
from Dataset_2023 import * 
import logging 
import pickle
import os

# Loop year 2023
def run_weekly_model(initial_level, initial_OnOFF, week, U, T, S):
    try:
            
            print(f"Running weekly model for week {week}")  # Start of function




            # Create a Gurobi model
            model = gp.Model("Week_"+str(week))


            #---------------------- VARIABLES ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
            total_T = n_weeks * h

            heat = model.addMVar((len(U), total_T), lb=0, name='heat')                                                  # Heat generation at each time step
            el = model.addMVar((len(U), total_T), lb=0, name='el')                                                      # Electricity generation at each time step
            el_revenue = model.addMVar((len(U), total_T), lb=-GRB.INFINITY, name='el_revenue')                                      # Electricity revenue for each unit at each time step
            f_cons = model.addMVar((len(U), total_T), lb=0, vtype='C', name='f_consumption')                                       # Fuel consumption for each unit at each time step
            f_cost = model.addMVar((len(U), total_T), name='f_cost')                                                    # Fuel cost for each unit at each time step
            el_cost_hp = model.addMVar((len(U), total_T), lb=-GRB.INFINITY,  vtype='C', name='el_cost_hp')                 # Electricity cost for Heat Pumps at each time step
            el_cost_boil = model.addMVar((len(U), total_T), lb=-GRB.INFINITY,  vtype='C', name='el_cost_boil')               # Electricity cost for Boilers at each time step
            el_cost = model.addMVar((len(U), total_T),lb=-GRB.INFINITY, vtype='C', name='el_cost')                      # Total electricity cost for each unit at each time step
            emiss_cost = model.addMVar((len(U), total_T), lb=0, name='emission_costs')                                  # Emissions costs for each unit at each time step
            OnOFF = model.addMVar((len(U), total_T), vtype=GRB.BINARY, name='OnOff')                                    # On/Off status of each unit at each time step
            OM = model.addMVar((len(U), total_T), lb=0, name='OM')                                                      # Variable O&M cost for each unit at each time step 
            tot_emiss=model.addMVar((len(U), total_T), lb=0, name='Emiss')                                              # Used to print the amount of emissions for each unit at each time step
            SU_cost = model.addMVar((len(U), total_T), lb=0, name='SU_cost')                                            # Start-up cost for each unit at each time step
            startUp = model.addMVar((len(U), total_T), vtype=GRB.BINARY, name='startUp')                                # Start-up status of each unit at each time step
            shutDown = model.addMVar((len(U), total_T), vtype=GRB.BINARY, name='startdownp')                            # Start-up status of each unit at each time step
            delta = model.addMVar((len(U), total_T), lb=-GRB.INFINITY, name="delta")                                    # Auxiliary variable to link start-up and shut-down variables with OnOff
            gamma = model.addMVar((len(U), total_T), lb=-GRB.INFINITY, name="gamma")                                    # Auxiliary variable to link start-up and shut-down variables with OnOff
            neg_el_loss = model.addMVar((len(U), total_T), lb=-GRB.INFINITY,vtype='C', name="neg_el_loss")  
            CAPEX_periodic = model.addMVar(len(U), lb=0, name="CAPEX_periodic")                              # Periodic CAPEX for each unit          
            Fix_OM_periodic = model.addMVar(len(U), lb=0, name="Fix_O&M_periodic")                           # Periodic Fixed O&M for each unit             


            # TES variables

            TES_Level = model.addMVar((len(S), total_T), lb=0,  name="TES_Level")  # Energy level in the TES at time t
            TES_Charge = model.addMVar((len(S), total_T), lb=0, name="TES_Charge")  # Energy charged into the TES at time t
            TES_Discharge = model.addMVar((len(S), total_T), lb=0, name="TES_Discharge")  # Energy discharged from the TES at time t
            ChargeStart = model.addMVar((len(S), total_T), vtype=GRB.BINARY, name="ChargeStart") # Charging  start indicator
            DischargeStart = model.addMVar((len(S), total_T), vtype=GRB.BINARY, name="DischargeStart") # Discharging start indicator
            Cycle = model.addMVar((len(S), total_T), vtype=GRB.BINARY, name="Cycle") # Cycle indicator


            print(f"Length of U: {len(U)}, Length of T: {len(T)}")


            #---------------------- OBJECTIVE FUNCTION ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            Heat_Cost = gp.quicksum((f_cost[u,t] + el_cost[u,t] + emiss_cost[u,t] + OM[u,t] + SU_cost[u,t] - el_revenue[u,t] )  for u in U for t in T)        # Minimize the cost of heat at each time step    
            model.setObjective(Heat_Cost, GRB.MINIMIZE) 



            #---------------------- CONSTRAINTS ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # HEAT DEMAND 
            for t in T:
                model.addConstr(sum(heat[u,t] for u in U) + (sum(TES_Discharge[s,t] for s in S)) - (sum(TES_Charge[s,t] for s in S)) == heat_demand[t]) # Ensure that demand is met


            for u in U:
                for t in T:


                    # MAX CAPACITY constraints --------------------------------------------------------------------------------------------------------
                    model.addConstr(heat[u,t] <= max_H_cap_array[u] * OnOFF[u,t]) # Max Heat capacity constraint
                    model.addConstr(el[u,t] <= max_El_cap_array[u] * OnOFF[u,t]) # Max Electricity capacity constraint


                    # FUEL constraints ------------------------------------------------------------------------------------------------------------------------
                    if u in All_fuel:  # Exclued heat pumps and boilers from calculations as they run on electricity
                        model.addConstr(f_cons[u,t] == (heat[u,t] + el[u,t]) / (H_eff_array[u]/100 + El_eff_array[u] /100)) # Calculate fuel consumption for each unit
                    

                        # Gets the fuel type of each unit
                        fuel_type = Fuel_type_dict[u]  

                        # Calculate fuel cost for each unit
                        fuel_price = fuels['fuel_prices'][fuel_type]
                        fuel_tax = fuels['fuel_taxes'][fuel_type]
                        model.addConstr(f_cost[u,t] == f_cons[u,t] * (fuel_price + fuel_tax ))  # Calculate fuel cost for each unitf
                


                    # ELECTRICITY CONSUMPTION COST constraint---------------------------------------------------------------------------------------------------------------------------------
                    if u == heat_pumps:  # imposes the if conditions to only calculate the electricity cost for heat pumps and el boilers
                        model.addConstr(el_cost_hp[u,t] == (el[u,t]) *  (el_price[t] + electricity_d_tax ), name='Electricity Cost for Heat Pumps')
                    else:
                        model.addConstr(el_cost_hp[u,t] == 0) 

                    if u == el_boilers:  # imposes the if conditions to only calculate the electricity cost for heat pumps and el boilers
                        model.addConstr(el_cost_boil[u,t] == (heat[u,t]) *  (el_price[t] + electricity_d_tax) , name='Electricity Cost for Boilers')
                    else:
                        model.addConstr(el_cost_boil[u,t] == 0)


                    


                    # TOTAL ELECTRICITY COST constraint------------------------------------------------------------------------------------------------------------------------------------------------
                    model.addConstr(el_cost[u,t] == el_cost_hp[u,t] + el_cost_boil[u,t]) # Calculate total electricity cost for each unit

                    # CO2 EMISSION COST constraints-----------------------------------------------------------------------------------------------------------------------------------------------------
                    model.addConstr(emiss_cost[u,t] == f_cons[u,t] * CO2_array[u] * emission_prices['CO2']) # Calculate CO2 emissions cost for each unit )
                    model.addConstr(tot_emiss[u,t]== f_cons[u,t] * CO2_array[u]) #not relevant for the model, just to print the amount of emissions for each unit at each time step

                    # MINIMUM CAPACITY constraint for CHPs----------------------------------------------------------------------------------------------------------------------------------------------
                    model.addConstr(heat[u,t] + el[u,t] >= (min_cap_array[u]/100) * (max_H_cap_array[u] + max_El_cap_array[u])* OnOFF[u,t] )    # Minimum capacity constraint for CHPs   #could also loop it only for CHP.unit dataframe


                    # O&M cost constraint
                    if u == no_bp_units: 
                        model.addConstr(OM[u,t] == Var_OM_array[u] * el[u,t] )  # Calculate variable O&M cost for each unit for each time step

                    if u == amager4_bp_unit: 
                        model.addConstr(OM[u,t] == Var_OM_array[u] * (heat[u,t]/(415/150)))  # Calculate variable O&M cost for each unit for each time step

                    if u == koge_bp_unit: 
                        model.addConstr(OM[u,t] == Var_OM_array[u] * (heat[u,t]/(70/25)))

                    if u == vestforbrænding_bp_unit:
                        model.addConstr(OM[u,t] == Var_OM_array[u] * (heat[u,t]/(143/51)))


    
                    #------------------------------------------------ CONSTRAINTS ONLY FOR CHPs -----------------------------------------------------------------------------------------------------------------------------------
                    if u in CHP_normal: #Include only CHPs that are not bypass in the constraint

                        # Heat-to-power ratio constraint
                        model.addConstr(heat[u,t] == HPR_array[u] * el[u,t]) # Relates the heat production to the electricity production of CHPs

                    # Minimum up and down time constraints
                        max_j_up = min(int(Up_time_array[u]), len(T) - t )
                        for j in range(max_j_up):
                            model.addConstr(OnOFF[u, t+j] >= OnOFF[u, t])

                        max_j_down = min(int(Down_time_array[u]), len(T) - t )

                        for j in range(max_j_down):
                           model.addConstr((shutDown[u, t] == 1) >> (OnOFF[u, t+j] == 0))


                    if u in ALL_no_bypass: #Include only CHPs that are not bypass in the constraint

                        # Start-up cost

                        model.addConstr(SU_cost[u,t] == startUp[u, t] * (Start_up_cost_array[u]) * (max_El_cap_array[u]))  # Calculate start-up cost for each unit for each time step

                        # Linking Start-Up and ShutDown Variable with OnOff

                        if initial_OnOFF.get(u) is not None: 
                             model.addConstr(OnOFF[u, T[0]] == initial_OnOFF[u])
                    
                        if t > T[0]:

                            # Constraints to capture the max value (max function is not supported in Gurobipy)
                            model.addConstr(delta[u, t] >= 0)
                            model.addConstr(delta[u, t] >= OnOFF[u, t] - OnOFF[u, t-1])
                            model.addConstr(startUp[u, t] == delta[u, t])
            
                            model.addConstr(gamma[u, t] >= 0)
                            model.addConstr(gamma[u, t] >= OnOFF[u, t-1] - OnOFF[u, t])
                            model.addConstr(shutDown[u, t] == gamma[u, t])
                        else:
                            model.addConstr(startUp[u, t] == OnOFF[u, t])
                            model.addConstr(shutDown[u, t] == 0)


                    if u in CHP_all: #Include all CHPs 

                        # Power sale constraint
                         model.addConstr(el_revenue[u,t] ==  el[u,t] * el_price[t])

                        
                    
                    else:
                        model.addConstr(el_revenue[u,t] == 0) # Set the electricity revenue to 0 for units that do not produce electricity
                        #model.addConstr(SU_cost[u,t] == 0) # Set the start-up cost to 0 for units that do not produce electricity   

                # CAPEX Calculation
                T_period_u= Lifetime_array[u] * n_period                                                                             # Calculate the total number of periods in the lifetime of each unit
            
                CAPEX_value_u = {}
                CAPEX_periodic_u = {}
                CAPEX_value_u[u] = CAPEX_array[u] * 1000000
                CAPEX_periodic_u[u] = CAPEX_value_u[u] * ((1 + r_period) * (1 - (1 + r_period)**(-T_period_u))) / r_period                                                       # Calculate the periodic CAPEX for each unit
            
                # Fixed O&M constraint
                Fix_OM_value_u = {}
                Fix_OM_periodic_u = {}
                Fix_OM_value_u[u] = Fix_OM_array[u]
                Fix_OM_periodic_u[u] = Fix_OM_value_u[u] * r_period / (1 - (1 + r_period)**(-n_period))                                                                           # Calculate the periodic Fixed O&M for each unit

            for t in T:   
            # By-pass mutally exclusive constraint
                model.addConstr(OnOFF[u_index['Amager4'],t] + OnOFF[u_index['Amager4_bp'],t] == 1 )                            # Amager4 and Amager4_bp are mutually exclusive
                model.addConstr(OnOFF[u_index['Køge'],t] + OnOFF[u_index['Køge_bp'],t] == 1 )                                  # Køge and Køge_bp are mutually exclusive
                model.addConstr(OnOFF[u_index['Vestforbrænding'],t] + OnOFF[u_index['Vestforbrænding_bp'],t] == 1 )            # Vestforbrænding and Vestforbrænding_bp are mutually exclusive

            #------------------------------------------------ TES CONSTRAINTS  -----------------------------------------------------------------------------------------------------------------------------------

            #Initial_Level = 0  # Initial energy level in the TES at t=1


            Charge_Efficiency = 1
            Discharge_Efficiency = 1


            for s in S:

                # STORAGE dynamics        
                for t in T:
                    if t == T[0]:  # Initial condition
                        model.addConstr(TES_Level[s,t] == initial_level[s] + Charge_Efficiency * TES_Charge[s,t] - TES_Discharge[s,t] / Discharge_Efficiency)
                    else:
                        model.addConstr(TES_Level[s,t] == TES_Level[s,t-1] + Charge_Efficiency * TES_Charge[s,t] - TES_Discharge[s,t] / Discharge_Efficiency)

                # STORAGE capacity  
                for t in T:
                    model.addConstr(TES_Level[s,t] <= TES_Cap_array[s]) # Max capacity constraint
                    model.addConstr(TES_Level[s,t] >= 0)

                # CHARGING and DISCHARGING rates

                for t in T:
                    model.addConstr(TES_Charge[s,t] <= TES_Charging_cap_array[s])
                    model.addConstr(TES_Discharge[s,t] <= TES_Charging_cap_array[s])
                    
                # MUTUALLY EXCLUSIVE of charging and discharging
                for t in T:
                    model.addConstr(TES_Charge[s,t] * TES_Discharge[s,t] == 0)




                # CYCLE DETERMINATION

                epsilon = 0.01 # Tolerance value

                # Handling the start time ( t=1 )
                model.addConstr(DischargeStart[s,T[0]] <= TES_Discharge[s,T[0]] + epsilon)
                model.addConstr(ChargeStart[s,T[0]] <= TES_Charge[s,T[0]] + epsilon)

                # For the rest of the time steps
                for t in range(T[2], len(T)):
                    # Constraints for ChargeStart: Only consider positive changes
                    model.addGenConstrIndicator(ChargeStart[s,t], True, TES_Charge[s,t] - TES_Charge[s,t-1], GRB.GREATER_EQUAL, epsilon)
                    model.addGenConstrIndicator(ChargeStart[s,t], False, TES_Charge[s,t] - TES_Charge[s,t-1], GRB.LESS_EQUAL, 0)
                    
                    # Constraints for DischargeStart: Only consider positive changes
                    model.addGenConstrIndicator(DischargeStart[s,t], True, TES_Discharge[s,t] - TES_Discharge[s,t-1], GRB.GREATER_EQUAL, epsilon)
                    model.addGenConstrIndicator(DischargeStart[s,t], False, TES_Discharge[s,t] - TES_Discharge[s,t-1], GRB.LESS_EQUAL, 0)
                    

                # CYCLE LIMIT CONSTRAINT

                # If there's a charge start or discharge start, it is considered as part of a cycle
                for t in range(T[0], len(T)):
                    model.addConstr(Cycle[s, t] >= ChargeStart[s, t])
                    model.addConstr(Cycle[s, t] >= DischargeStart[s, t])
                    
                # Limit the number of cycles allowed in 24 hours
                for t in range(T[25], len(T)):
                    model.addConstr(gp.quicksum(Cycle[s, i] for i in range(t-24, t+1)) <= TES_Max_Cycles_24_array[s])


                # CAPEX Calculation
                T_period_s = TES_Lifetime_array[s] * n_period                                                                              # Calculate the total number of periods in the lifetime of each unit 

                CAPEX_value_s = {}
                CAPEX_periodic_s = {}
                CAPEX_value_s[s] = TES_Capex_array[s] * 1000000 
                CAPEX_periodic_s[s] = CAPEX_value_s[s] * ((1 + r_period) * (1 - (1 + r_period)**(-T_period_s))) / r_period 

                # Fixed O&M constraint
                Fix_OM_value_s = {}
                Fix_OM_periodic_s = {}
                #Fix_OM_value_s[s] = TES.loc[Units['PlantName'] == s, 'Fix. O&M'].values[0]
                #Fix_OM_periodic_s[s] = Fix_OM_value_s[s] * r_period / (1 - (1 + r_period)**(-n_period))   
                
                                                                                        # Calculate the periodic Fixed O&M for each unit

#-----------------------------------------------------------------------------------------

            # Optimize the model
            print("Starting optimization")
            model.optimize()
            model.update()
            print("Optimization completed")


             # Collect final TES Level for each storage
            final_TES_Level = {s: TES_Level[s, T[-1]].X for s in S}
            final_OnOFF = {u: OnOFF[u, T[-1]].X for u in U}

            weekly_heat = {(u, t):  heat[u, t].X for u in U for t in T}
            weekly_TES_charge = {(s, t): TES_Charge[s, t].X for s in S for t in T}
            weekly_TES_discharge = {(s, t): TES_Discharge[s, t].X for s in S for t in T}
            weekly_TES_level = {(s, t): TES_Level[s, t].X for s in S for t in T}

            print("Returning data from run_weekly_model")
            return final_TES_Level, final_OnOFF, weekly_heat, weekly_TES_charge, weekly_TES_discharge, weekly_TES_level


    except gp.GurobiError as e:
        logging.error(f"Gurobi Error in week {week}: {str(e)}")
        print(f"Gurobi Error: {str(e)}")  # Log the specific Gurobi error
        raise  # Re-raise the error for now to see the traceback
    except Exception as e:
        logging.error(f"Error in week {week}: {str(e)}")
        print(f"Other Error: {str(e)}")  # Log the general error
        raise  # Re-raise the error


# Function to save TES_Level and OnOFF status to a file
def save_state(final_TES_Level, final_OnOFF, week):

    folder_path = '../02. Output'

    # Define the full path for the file
    file_path = os.path.join(folder_path, f"state_week{week}.pkl")

    state = {
        "final_TES_Level": final_TES_Level,
        "initial_OnOFF": final_OnOFF
    }
    
     # Save the state to the file
    with open(file_path, "wb") as file:
        pickle.dump(state, file)



# Function to load TES_Level and OnOFF status from a file
def load_state(week):

    folder_path = '../02. Output'
    file_path = os.path.join(folder_path, f"state_week{week}.pkl")


    # Load the state from the file
    with open(file_path, "rb") as file:
        state = pickle.load(file)


    return state.get("final_TES_Level", {}), state.get("initial_OnOFF", {})  # Return empty dictionaries if no state is found


# Main execution loop


def main(U, S):
    
    
    logging.basicConfig(filename='model_log.log', level=logging.INFO)

    initial_level = {s: 0 for s in S}
    initial_OnOFF = {u: 0 for u in U}
    heat_all_hours = {(u, t): 0.0 for u in U for t in list(range(n_weeks * h))}
    TES_charge_all_hours = {(s, t): 0.0 for s in S for t in list(range(n_weeks * h))}
    TES_discharge_all_hours = {(s, t): 0.0 for s in S for t in list(range(n_weeks * h))}
    TES_level_all_hours = {(s, t): 0.0 for s in S for t in list(range(n_weeks * h))}


    for week in range(1, n_weeks):
        start_timestep = (week-1) * h 
        end_timestep = start_timestep + h
        T = list(range(start_timestep, end_timestep))

        # Load state if not the first week
        if week > 1:
            initial_level, initial_OnOFF = load_state(week - 1)
   

        final_TES_Level, final_OnOFF, weekly_heat, weekly_TES_charge, weekly_TES_discharge, weekly_TES_level = run_weekly_model(initial_level, initial_OnOFF, week, U, T, S)
       


        if final_TES_Level is not None:
            save_state(final_TES_Level, final_OnOFF, week)

            for u in U:
                for t in T:
                        heat_value = weekly_heat[u, t]
                        heat_all_hours[u, t] += heat_value
            for s in S:
                for t in T:
                    TES_charge_value = weekly_TES_charge[s, t]
                    TES_discharge_value = weekly_TES_discharge[s, t]
                    TES_level_value = weekly_TES_level[s, t]
                    TES_charge_all_hours[s, t] += TES_charge_value
                    TES_discharge_all_hours[s, t] += TES_discharge_value
                    TES_level_all_hours[s, t] += TES_level_value
                    

    
        else:
            logging.error(f"No result for week {week}.")
            break

    # Bundle all data into a dictionary
    data_to_save = {
        "heat_production": heat_all_hours,
        "TES_charge": TES_charge_all_hours,
        "TES_discharge": TES_discharge_all_hours,
        "TES_level": TES_level_all_hours
    }


    # Construct the full path for the output file
    folder_path = folder_path = '../02. Output'
    output_file_path = os.path.join(folder_path, 'model_data.pkl') # Name of your output file

    # Save the aggregated heat production data for all weeks
    with open(output_file_path, 'wb') as file:
        pickle.dump(data_to_save, file)



if __name__ == "__main__":
    U = list(range(len(Units['PlantName'])))
    S = list(range(len(TES_df['PlantName'])))             # Set of TES storages S(s
    main(U, S)



















