from tools.Nth_Order_ECM import Nth_Order_ECM
from tools.tools import *
import numpy as np
# import matplotlib.pyplot as plt
import pickle
import pandas as pd
from pathlib import Path

def full_cycle_data(data, schedules, T, data_mean, plot_ecm_parameter=True):
    """
    Function that simulates an 2nd order ECM to generate Full Cycle (FC) data. The data is 
    stored in data_dict parameter.
    data: dictionary of nested dictionaries. Used to store the generated data
    schedules: dictionary containing the routines used to simulate the ECM
    T: temperature array containing strings of temperatures
    ecm_param: a dictionary containing the ECM parameters (the interpolation of its parameters)
    
    Example of schedules:
    schedules.update({'FC': {}}) # FC: full cycle
    schedules['FC'].update({'I': [0.5, 1, 2]}) # constant current of 0.5, 1 and 2 A
    """
    
    data_lad, data_ela = load_data_LGHE4C25B01()
    for temp in T:
        C_ref = (np.mean(data_lad[temp]["Cap"])+np.mean(data_ela[temp]["Cap"]))/2
        dt = (np.abs(data_lad[temp]["Ah"][1]-data_lad[temp]["Ah"][0])/np.abs(data_lad[temp]["I"][0])+np.abs(data_ela[temp]["Ah"][1]-data_ela[temp]["Ah"][0])/np.abs(data_ela[temp]["I"][0]))/2
        f = interpolate_ecm_parameter(data_mean,deg=5,deg_ocv=20,plot=plot_ecm_parameter,T=temp)
        data[temp].update({'FC': {}}) # add an FC key to the dictionary
        for cur in schedules['FC']['I']: # for loop to iterate over the FC I schedules
            print(cur) # print the current schedule
            bm = Nth_Order_ECM(f["R0"],[f["R1"],f["R2"]],[f["Tau1"],f["Tau2"]],f["ocv_grid"],C_ref=C_ref,ts=dt,soc_init=0.01,eta=1, T=temp) # create an emc instance with the above defined paramters
            data[temp]['FC'].update({cur: {'I':[], 'soc':[], 'V':[], 'V1':[], 'V2':[], 'T':[]}}) # updates the data dictionary with the current schedule as a key 
            # and a value that is also a dictionary containing Current I, SOC, V, V1 and V2

            # Charge the battery with the current value I until the SOC reaches 1
            flag = False # boolean variable. True: battery fully charged or fully discharged. Otherwise False
            u = np.copy(cur) # u is a vector containing current value I
            x = np.copy(bm.state) # x is the state vector of the battery
            while flag==False: # while loop: as long as flag is False -> charge the battery
                xp = bm.state_equation(u, x) # xp is the next state vector of the battery at the next time step
                y, flag = bm.output_equation(u, x, flag=True) # y is the output terminal voltage (Klemmspannung), flag is the same boolean variable as earlier
                bm.update_parameter(xp[0]) # update ecm model with new SOC, V1 and V2
                data[temp]['FC'][cur]['I'].append(u.item()) # appends the current I value to data at given keys
                data[temp]['FC'][cur]['V'].append(y.item()) # appends the current V value to data at given keys
                data[temp]['FC'][cur]['soc'].append(x[0].item()) # appends the current SOC value to data at given keys
                data[temp]['FC'][cur]['V1'].append(x[1].item()) # appends the current V1 value to data at given keys
                data[temp]['FC'][cur]['V2'].append(x[2].item()) # appends the current V2 value to data at given keys
                data[temp]['FC'][cur]['T'].append(float(temp))
                x = np.copy(xp) # change the old state vector with the new state vector of the battery

            # Discharge the battery with the current value I until SOC reaches 0
            # Same as previously
            flag = False
            u = np.copy(cur)*-1 # same current array but neg. to discharge the battery
            x = np.copy(bm.state)
            while flag==False:
                xp = bm.state_equation(u, x)
                y, flag = bm.output_equation(u, x, flag=True)
                bm.update_parameter(xp[0])
                data[temp]['FC'][cur]['I'].append(u.item())
                data[temp]['FC'][cur]['V'].append(y.item())
                data[temp]['FC'][cur]['soc'].append(x[0].item())
                data[temp]['FC'][cur]['V1'].append(x[1].item())
                data[temp]['FC'][cur]['V2'].append(x[2].item())
                data[temp]['FC'][cur]['T'].append(float(temp))
                x = np.copy(xp)

def pulse_cycle_data(data, schedules, T, data_mean, plot_ecm_parameter=True, C_ref = 2.5):
    """
    Function that simulates a 2nd order ECM to generate Pulse Cycle (PC) data. The
    data is then stored in the data parameter. The ECM is subjected to pulse currents
    under various duty cycles.
    data: dictionary of nested dictionaries. Used to store the generated data
    schedules: dictionary containing the routines used to simulate the ECM
    T: temperature array containing strings of temperatures
    ecm_param: a dictionary containing the ECM parameters (the interpolation of its parameters)
    
    Example of schedules:
    schedules['PC'].update({'I': [0.5, 1, 2]}) # pulse current of 0.5, 1 and 2 A
    schedules['PC'].update({'mSOC': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}) # 
    schedules['PC'].update({'DOD': [[0.2], [0.2, 0.4], [0.2, 0.4, 0.6], [0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6], [0.2, 0.4], [0.2]]})

    """
    data_lad, data_ela = load_data_LGHE4C25B01()
    for temp in T:
        C_ref = (np.mean(data_lad[temp]["Cap"])+np.mean(data_ela[temp]["Cap"]))/2
        dt = (np.abs(data_lad[temp]["Ah"][1]-data_lad[temp]["Ah"][0])/np.abs(data_lad[temp]["I"][0])+np.abs(data_ela[temp]["Ah"][1]-data_ela[temp]["Ah"][0])/np.abs(data_ela[temp]["I"][0]))/2
        f = interpolate_ecm_parameter(data_mean,deg=5,deg_ocv=20,plot=plot_ecm_parameter,T=temp)
        data[temp].update({'PC':{}}) # adds key "PC" to data dictionary
        for mSOC, DOD in zip(schedules['PC']['mSOC'], schedules['PC']['DOD']): # iterates over pair of values mSOC and DOD from schedules["PC"]
            print('mSOC:', mSOC) # print current mean SOC
            data[temp]['PC'].update({mSOC:{}}) # add a new value to PC key of data which is an empty dictionary
            for _DOD in DOD: # iterates over DOD list
                print('DOD:', _DOD) # print current DOD
                data[temp]['PC'][mSOC].update({_DOD: {}}) # add empty dictionary to data["PC"][current mSOC][current _DOD]
                for cur in schedules['PC']['I']: # iterate over current list at schedules["PC"]["I"]
                    print('I:', cur) # print current current I
                    bm = Nth_Order_ECM(f["R0"],[f["R1"],f["R2"]],[f["Tau1"],f["Tau2"]],f["ocv_grid"],C_ref=C_ref,ts=dt,soc_init=mSOC-_DOD/2,eta=1, T=temp) # instantiate a 2nd order ecm with the above defined paramters
                    data[temp]['PC'][mSOC][_DOD].update({cur: {'I':[], 'soc':[], 'V':[], 'V1':[], 'V2':[], 'T':[]}})# update the data dictionary with the current schedule as a key 
                    # and a value that is also a dictionary containing Current I, SOC, V, V1 and V2

                    # Initialize the loop for simulating the charging cycle of the battery modelized by the ecm instance
                    # Charge
                    flag = False # boolean variable. Check if SOC reached the desired limit
                    u = np.copy(cur) # set u to the current value of cur
                    x = np.copy(bm.state) # set x to the current battery model state [SOC, U1, U2]
                    while flag==False:
                        xp = bm.state_equation(u, x) # next state vector at the next timestep of the ecm
                        y, flag = bm.output_equation(u, x, flag=True) # y: estimated terminal voltage
                        bm.update_parameter(xp[0]) # update ecm with the new state vector
                        data[temp]['PC'][mSOC][_DOD][cur]['I'].append(u.item()) # appends the current I value to data at given keys
                        data[temp]['PC'][mSOC][_DOD][cur]['V'].append(y.item()) # appends the current V value to data at given keys
                        data[temp]['PC'][mSOC][_DOD][cur]['soc'].append(x[0].item()) # appends the current SOC value to data at given keys
                        data[temp]['PC'][mSOC][_DOD][cur]['V1'].append(x[1].item()) # appends the current V1 value to data at given keys
                        data[temp]['PC'][mSOC][_DOD][cur]['V2'].append(x[2].item()) # appends the current V2 value to data at given keys
                        data[temp]['PC'][mSOC][_DOD][cur]['T'].append(float(temp))
                        x = np.copy(xp) # store new state vector of ecm in x variable
                        if x[0] >= mSOC + _DOD/2: # condition that checks if the desired SOC has been reached
                            flag = True

                    # Similar to the charging part. During charging, when the desired SOC has been reached then
                    # start with discharge part
                    # Discharge
                    flag = False
                    u = np.copy(cur)*-1 # negative current
                    x = np.copy(bm.state)
                    while flag==False:
                        xp = bm.state_equation(u, x)
                        y, flag = bm.output_equation(u, x, flag=True)
                        bm.update_parameter(xp[0])
                        data[temp]['PC'][mSOC][_DOD][cur]['I'].append(u.item())
                        data[temp]['PC'][mSOC][_DOD][cur]['V'].append(y.item())
                        data[temp]['PC'][mSOC][_DOD][cur]['soc'].append(x[0].item())
                        data[temp]['PC'][mSOC][_DOD][cur]['V1'].append(x[1].item())
                        data[temp]['PC'][mSOC][_DOD][cur]['V2'].append(x[2].item())
                        data[temp]['PC'][mSOC][_DOD][cur]['T'].append(float(temp))
                        x = np.copy(xp)
                        if x[0] <= mSOC - _DOD/2:
                            flag = True


def incremental_ocv_data(data, schedules, T, data_mean, plot_ecm_parameter=True):
    """
    Function that simulates a 2nd order ECM to generate incremental open circuit voltage
    (iOCV) data. The ECM is subjected to a series of current pulses followed by a
    rest period to measure the voltage response of the battery.
    data: dictionary of nested dictionaries. Used to store the generated data
    schedules: dictionary containing the routines used to simulate the ECM
    T: temperature array containing strings of temperatures
    ecm_param: a dictionary containing the ECM parameters (the interpolation of its parameters)
    
    Example of schedules:
    schedules['iOCV'].update({'I': [0.5, 1, 2]}) 
    schedules['iOCV'].update({'pt': [30, 60, 120]}) # Pulse time (s)
    schedules['iOCV'].update({'rt': 600})# Relaxation time (s)
    """
    data_lad, data_ela = load_data_LGHE4C25B01()
    for temp in T:
        C_ref = (np.mean(data_lad[temp]["Cap"])+np.mean(data_ela[temp]["Cap"]))/2
        dt = (np.abs(data_lad[temp]["Ah"][1]-data_lad[temp]["Ah"][0])/np.abs(data_lad[temp]["I"][0])+np.abs(data_ela[temp]["Ah"][1]-data_ela[temp]["Ah"][0])/np.abs(data_ela[temp]["I"][0]))/2
        f = interpolate_ecm_parameter(data_mean,deg=5,deg_ocv=20,plot=plot_ecm_parameter,T=temp)
        data[temp].update({'iOCV': {}}) # add a new key with a value: an empty dictionary
        for pt in schedules['iOCV']['pt']: # iterates over the respectiv schedule
            print('pt:', pt) # prints current pulse times from the schedule
            data[temp]['iOCV'].update({pt: {}}) # add new dictionary as the value at the current keys
            for cur in schedules['iOCV']['I']: # loop over the different current schedules at iOCV
                print('I:', cur) # print current current I
                bm = Nth_Order_ECM(f["R0"],[f["R1"],f["R2"]],[f["Tau1"],f["Tau2"]],f["ocv_grid"],C_ref=C_ref,ts=dt,soc_init=0.01,eta=1) # instance of batteryModel_static 
                data[temp]['iOCV'][pt].update({cur: {'I':[], 'soc':[], 'V':[], 'V1':[], 'V2':[], 'T':[]}}) # adds new dictioanries at given keys

                # Charge
                flag = False # True: battery fully charged or fully discharged
                x = np.copy(bm.state) # current state vector of the ecm
                while flag==False: 
                    u = np.copy(cur) # current current I
                    for t in range(pt):  # Pulse: see the above pt list: it represents for how long the current is being applied
                        xp = bm.state_equation(u, x) # store the next state vector of the battery
                        y, flag = bm.output_equation(u, x, flag=True) # store terminal voltage in y and weither the limit has been reached or not (charging or discharging of the battery)
                        bm.update_parameter(xp[0]) # update the state vector of the ecm
                        # stores the output values of the ecm in data dictionary
                        data[temp]['iOCV'][pt][cur]['I'].append(u.item())
                        data[temp]['iOCV'][pt][cur]['V'].append(y.item())
                        data[temp]['iOCV'][pt][cur]['soc'].append(x[0].item())
                        data[temp]['iOCV'][pt][cur]['V1'].append(x[1].item())
                        data[temp]['iOCV'][pt][cur]['V2'].append(x[2].item())
                        data[temp]['iOCV'][pt][cur]['T'].append(float(temp))
                        x = np.copy(xp) # store the next state xp in the variable x
                        # print(y, flag)
                        if flag: # if flag==True then break from the for loop
                            break
                    u = np.copy(0) # when the pulse time has passed by or the battery is fully charged or fully discharged then sets the current to 0
                    for t in range(schedules['iOCV']['rt']):  # Rest
                        xp = bm.state_equation(u, x)
                        y = bm.output_equation(u, x, flag=False)
                        bm.update_parameter(xp[0])
                        data[temp]['iOCV'][pt][cur]['I'].append(u.item())
                        data[temp]['iOCV'][pt][cur]['V'].append(y.item())
                        data[temp]['iOCV'][pt][cur]['soc'].append(x[0].item())
                        data[temp]['iOCV'][pt][cur]['V1'].append(x[1].item())
                        data[temp]['iOCV'][pt][cur]['V2'].append(x[2].item())
                        data[temp]['iOCV'][pt][cur]['T'].append(float(temp))
                        x = np.copy(xp)

                # Discharge
                # Similar to previously 
                flag = False
                x = np.copy(bm.state)
                while flag==False:
                    u = np.copy(cur)*-1
                    for t in range(pt):  # Pulse
                        xp = bm.state_equation(u, x)
                        y, flag = bm.output_equation(u, x, flag=True)
                        bm.update_parameter(xp[0])
                        data[temp]['iOCV'][pt][cur]['I'].append(u.item())
                        data[temp]['iOCV'][pt][cur]['V'].append(y.item())
                        data[temp]['iOCV'][pt][cur]['soc'].append(x[0].item())
                        data[temp]['iOCV'][pt][cur]['V1'].append(x[1].item())
                        data[temp]['iOCV'][pt][cur]['V2'].append(x[2].item())
                        data[temp]['iOCV'][pt][cur]['T'].append(float(temp))
                        x = np.copy(xp)
                        # print(y, flag)
                        if flag:
                            break
                    u = np.copy(0)
                    for t in range(schedules['iOCV']['rt']):  # Rest
                        xp = bm.state_equation(u, x)
                        y = bm.output_equation(u, x, flag=False)
                        bm.update_parameter(xp[0])
                        data[temp]['iOCV'][pt][cur]['I'].append(u.item())
                        data[temp]['iOCV'][pt][cur]['V'].append(y.item())
                        data[temp]['iOCV'][pt][cur]['soc'].append(x[0].item())
                        data[temp]['iOCV'][pt][cur]['V1'].append(x[1].item())
                        data[temp]['iOCV'][pt][cur]['V2'].append(x[2].item())
                        data[temp]['iOCV'][pt][cur]['T'].append(float(temp))
                        x = np.copy(xp)