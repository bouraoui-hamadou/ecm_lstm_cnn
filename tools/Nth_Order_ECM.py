import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import sys

class Nth_Order_ECM:
    """
    Class that simulates an nth order ECM (n = 1, 2, 3, etc.)
    """
    def __init__(self, R0, Rs, Taus, ocv_grid, eta=1, T=15, C_ref=3600, 
                soc_init=0.01, Us_init=None, ts=1, eoc=4.2, eod=2.5):
        """
        Arguments:
            R0:
                description: first resistance in the ECM. To be interpolated if size > 1. 
                    Otherwise no.
                unit: Ohm
                type: Array of scalar
            Rs:
                description: Contains values of each resistance of each RC-circuit. 
                    The values of the second dimensions are to be interpolated if size > 1.
                    Otherwise no.
                unit: Ohm
                type: Array of arrays.
            Taus:
                description: Contains values of the time time constants for each RC-circuit.
                    The values of the second dimensions are to be interpolated if size > 1.
                    Otherwise no.
                unit: s
                type: Array of arrays
            ocv:
                description: Contains open circuit voltage values respective to the soc 
                    values of the battery
                unit: V
                type: Array
                default: None. In this case assumes a linear relationship between soc
                    and ocv. The max and min ocv would be respectively eoc and eod
            soc:
                description: Contains state of charge values
                unit: percentage
                type: Array
                defaul: None. In this case assumes a linear relationship between soc
                    and ocv.
            eta: 
                description: Coulombic deficiency
                unit: no unit
                type: scalar (between 0 and 1)
            T:
                description: Temperature
                unit: Celsius
                type: Scalar
            C_ref:
                description: Total capacity of the battery
                unit: As
                type: Scalar
            soc_init:
                description: Initial soc of the battery
                unit: Percentage
                type: Scalar
            Us_init:
                description: Initial voltage accross each RC-circuits
                unit: V
                type: Array
            ts:
                description: Sampling time
                unit: s
                type: Scalar
            eoc:
                description: end of charge of the battery. Voltage level where the battery 
                    can no longer be charged
                unit: V
                type: Scalar
            eod:
                description: end of discharge of the battery. Voltage level where the
                    battery can no longer be discharged
                unit: V
                type: Scalar
        """
        self.order = len(Rs)
        self.R0 = np.array(R0)
        self.Rs = np.array(Rs)
        self.Taus = np.array(Taus)
        self.C_ref = C_ref
        self.ocv_grid = np.array(ocv_grid) # if np.any(ocv_grid != None)  else np.array([eod, eoc])
        # self.soc_grid = np.array(soc_grid) if np.any(soc_grid != None ) else np.array([0, 1])
        self.eta = eta
        self.T = T
        self.C_ref = C_ref
        self.soc_init = soc_init
        self.Us_init = np.zeros(self.order) if Us_init == None else Us_init
        self.ts = ts
        self.eoc = eoc
        self.eod = eod
        
        # needed later for numerical derivation
        self.state = np.vstack((np.array(self.soc_init), self.Us_init.reshape(-1,1)))
        self.state_previous_1 = np.copy(self.state)
        
        # Fill dictionary f with ECM parameters as interpolated functions with numpy
        self.f = {"R"+str(x): "" for x in range(len(self.Rs)+1)}
        self.f.update({"Tau"+str(x+1): "" for x in range(len(self.Rs))})
        
        self.f["R0"] = self.R0
        for i in range(self.order):
            self.f["R"+str(i+1)] = self.Rs[i]
            self.f["Tau"+str(i+1)] = self.Taus[i]
        self.f["ocv"] = self.ocv_grid
        self.update_parameter(self.state[0])
        self.recorded_xp=[]
        
        # create a dictionary containing class attributes and their units
        #self.parameter = {"R"+str(x): "Function" if callable(f["R"+str(x)]) else f["R"+str(x)] for x in range(self.order+1)}
        #self.parameter.update({"Tau"+str(x+1): "Function" if callable(f["Tau"+str(x+1)]) else f["Tau"+str(x+1)] for x in range(self.order+1)})
        #self.parameter.update("T":)
    
    # Update ocv of the ecm
    def update_parameter(self, soc):
        self.ocv = np.polyval(self.f["ocv"], soc)
        
    # u is not the voltage but the rather the matrix (in our case containing the current I)
    def state_equation(self, u, x):
        if u >= 0:
            eta = self.eta
        else:
            eta = 1
        # self.state[0] represents current soc
        xp = self.__A__()@x + self.__B__()*u
        # update the state attributes (done wrong in the class of the dominic because of the order)
        self.state_previous_2 = np.copy(self.state_previous_1)
        self.state_previous_1 = np.copy(self.state)
        self.state = np.copy(xp)
        self.recorded_xp.append(np.copy(xp))
        return self.state
    
    
    def output_equation(self, u, x, flag=False):
        flag_voltage_limit = False
        soc = self.state[0]
        R0 =np.polyval(self.f["R0"],soc)
        y = u*R0 + np.sum(x[1:]) + self.ocv
        #print(f"Bouraoui: sum x[1:]={np.sum(x[1:])}")
        # sys.exit()
        if y >= self.eoc or y <= self.eod:
            flag_voltage_limit = True
        if flag: return y, flag_voltage_limit
        else: return y
    
    def calc_CV_current(self, voltage_cv=4.2, i_max=1, i_min=-1):
        soc = self.state[0]
        Ux_dot = self.state[1:] - self.state_previous_1[1:]
        Taux = np.array([np.polyval(self.f[key], soc) for key in self.f.keys() if key.startswith("Tau")])
        Rx = np.array([np.polyval(self.f[key], soc) for key in self.f.keys() if key.startswith("R")])                                     
        info = voltage_cv - self.ocv
        num = info + np.sum(Ux_dot*Taux)
        i = num / np.sum(Rx)
        #print(f"Ux_dot = {Ux_dot}\nTaux={Taux}\nRx={Rx}\ninfo={info}\nnum={num}\ni={i}")
        current = min(i_max, i.item())
        return (current, num, info, Ux_dot)
    
    def __A__(self):
        # define a square matrix
        A = np.zeros((self.order+1,self.order+1))
        # set first element to 1
        A[0,0] = 1
        # set diagonal the needed values
        soc = self.state[0][0]
        for i in range(self.order):
            A[i+1,i+1] = np.exp(-self.ts*np.polyval(self.f["Tau"+str(i+1)],soc)**-1)
        return A
    
    def __B__(self, eta=None):
        # define a vector of shape 1 column and rows equal to the order of the ECM +1
        B = np.zeros((self.order+1,1))
        eta = self.eta if eta == None else eta
        # set first element to
        B[0,0] = self.ts*eta/self.C_ref
        # set other elements to
        soc = self.state[0][0]
        for i in range(self.order):
            B[i+1,0] = -np.polyval(self.f["R"+str(i+1)],soc) * (np.exp(-self.ts/np.polyval(self.f["Tau"+str(i+1)],soc)) - 1)
        return B