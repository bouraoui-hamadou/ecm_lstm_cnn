import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as spi
import os
import copy
from tabulate import tabulate
from IPython.display import display
import sys
import scipy.io
import pandas as pd
import mat4py
import math
from torch.utils.data import Dataset
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import optuna
from tools.Nth_Order_ECM import Nth_Order_ECM

torch.manual_seed(1)

def load_data_LGHE4C25B01_Parameter():
    """Helper fucntion used to load the all ECM parameter from the file LGHE4C25B01_Parameter.mat.
    The file must be in the folder /data/LGHE4_data of the current directory

    Returns:
        tuple: tuple of pandas dataframe containing in the following order the charge, discharge and mean parameter values
    """
    
    # Get the absolute path of the current directory
    current_dir = os.getcwd()

    # Construct the relative path to the file you want to access (os independent)
    path = os.path.join(current_dir, "data", "LGHE4_data", "LGHE4C25B01_Parameter.mat")

    # Load target file
    data_matlab = mat4py.loadmat(path) # ["LGHE4C25B01_Parameter"]

    # Parameter names to extract from the matlab file
    parameter_name = {"QOCV":["dch","ch","mean","dchSOC","chSOC","meanSOC"],
                     "R0":["mean","ch","dch"],
                     "R1":["mean","ch","dch"],
                     "R2":["mean","ch","dch"],
                     "SOCP":["mean","ch","dch"],
                     "Tau1":["mean","ch","dch"],
                     "Tau2":["mean","ch","dch"]}

    # Define empty dictionaries
    parameter_ch = {}
    parameter_dch = {}
    parameter_mean = {}

    # Fill the empty dictionaries with their respectiv keys
    for key, values in parameter_name.items():
        parameter_ch[key] = []
        parameter_dch[key] = []
        parameter_mean[key] = []
        for x in values:
            if "dch" in x:
                parameter_dch[key].append(x)
            elif "ch" in x:
                parameter_ch[key].append(x)
            else:
                parameter_mean[key].append(x)

    # List of temperatures
    T = ["15","25","45"]

    # Dictionary to store some data from the matlab file
    data = {y: None for y in parameter_name}

    # Store the matlab file inside the dictionary data
    for j in parameter_name:
        df = pd.DataFrame(columns=parameter_name[j])
        for k in parameter_name[j]:
            df[k] = data_matlab[j][k]
        data[j] = df

    # Empty dictionary of dictionaries to store dataframes later
    # The outer keys represent the temperature
    # The inner keys represent the column names of the data_matlab variable 
    data_ch = {x: {y: None for y in parameter_name} for x in T}
    data_dch = {x: {y: None for y in parameter_name} for x in T}
    data_mean = {x: {y: None for y in parameter_name} for x in T}

    # Function to fill the above defined dictionaries
    def fill_data(parameter, data_x):
        for i in range(len(T)):
            for j in parameter:
                df = pd.DataFrame(columns = parameter[j])
                row = data[j].iloc[i]
                for k in parameter[j]:
                    df[k] = row[k]
                data_x[T[i]][j] = df

    fill_data(parameter_ch, data_ch)
    fill_data(parameter_dch, data_dch)
    fill_data(parameter_mean, data_mean)

    # unwrap cell values from each dataframe inside data_ch, data_dch and data_mean
    for j in data_ch:
        for i in data_ch[j].values():
            i.loc[:] = i.loc[:].applymap(lambda x: x[0])
    for j in data_dch:
        for i in data_dch[j].values():
            i.loc[:] = i.loc[:].applymap(lambda x: x[0])
    for j in data_mean:
        for i in data_mean[j].values():
            i.loc[:] = i.loc[:].applymap(lambda x: str(x[0]) if isinstance(x, (list, tuple)) else x)

    return data_ch, data_dch, data_mean



def load_data_LGHE4C25B01():
    """Helper function used to load the charge-discharge data of the battery in question.
    The file must be in the following folder /data/LGHE4_data of the current directory.

    Returns:
        tuple: tuple of pandas dataframe in the following order: charge data, discharge data
    """
    # Get the absolute path of the current directory
    current_dir = os.getcwd()

    # Construct the relative path to the file you want to access (os independent)
    data_matlab_path = os.path.join(current_dir, "data", "LGHE4_data", "LGHE4C25B01.mat")
    # Loads the target file
    data_matlab = mat4py.loadmat(data_matlab_path)["LGHE4C25B01"]

    sub_column_ela = ["U",
                     "I",
                     "T",
                     "Ah",
                     "Cap",
                     "t_step"] 
    sub_column_lad = ["U",
                     "I",
                     "T",
                     "Ah",
                     "Cap",
                     "t_step"]
    temperature = ["15","25","45"]
    data_ela = dict.fromkeys(temperature)
    data_lad = dict.fromkeys(temperature)

    # loads contents of the struct QOCV_ela from the matlab file in a pandas DataFrame
    for i in range(len(temperature)):
        df = pd.DataFrame(columns=sub_column_ela)
        for j in sub_column_ela:
            df[j] = data_matlab["QOCV_ela"][j][i]
        data_ela[temperature[i]] = df

    # loads contents of the struct QOCV_lad from the matlab file in a pandas DataFrame
    for i in range(len(temperature)):
        df = pd.DataFrame(columns=sub_column_lad)
        for j in sub_column_lad:
            df[j] = data_matlab["QOCV_lad"][j][i]
        data_lad[temperature[i]] = df

    # the data is still not formatted the desired format. The elements of the the columns U, I, T and Ah are lists, even if they store only one element.
    # iterate over dictionaries and typecast them to floats

    for i in data_ela.values():
        i.loc[:, i.columns != 'Cap'] = i.loc[:, i.columns != 'Cap'].applymap(lambda x: x[0])
    for i in data_lad.values():
        i.loc[:, i.columns != 'Cap'] = i.loc[:, i.columns != 'Cap'].applymap(lambda x: x[0])

    # Ah key is flipped in data_ela dictionary
    for i in data_ela:
        data_ela[i]["Ah"] = data_ela[i]["Ah"][::-1].reset_index(drop=True)
    
    # return the data frames
    return data_lad, data_ela

def interpolate_ecm_parameter(data:pd.DataFrame,
                              deg:int,
                              deg_ocv=-1,
                              T="15",
                              from_data="mean",
                              order=2, 
                              plot=False,
                              figsize=(12,8),
                              savefig=True)->dict:
    """Helper function to interpolate ECM parameters using np.polyfit.

    Args:
        data (pandas dataframe): 1st key: is the temperature (string). 2nd is the parameter to be interpolated (string). 3rd key is ch, dch or mean (string)
        deg (int): polynom degree for the interpolation of the data points of the parameters.
        deg_ocv (int, optional): polynom degree used just for the data points interpolation of the OCV parameter. Defaults to -1 to use the "deg" parameter instead.
        T (str, optional): Dataframe key of "data". The parameters in the matlab file are dependent on the temperature. Either 15, 25 or 45°C. Defaults to "15".
        from_data (str, optional): Dataframe key of "data". Either "ch", "dch" or "mean". Defaults to "mean".
        order (int, optional): Represents the order of the ECM. Needed to plot the ECM parameter interpolations. Defaults to 2.
        plot (bool, optional): If True, plot the ECM parameters. Defaults to False.
        figsize (tuple, optional): Size of the figure to be plotted. Defaults to (12,8).
        savefig (bool, optional): If True, saves the figure. Defaults to True.

    Returns:
        dictionary: Dictionary containing the interpolations
    """


    # Load the data
    R0 = data[T]["R0"][from_data].values.astype(float)
    Rs = [data[T][f"R{i}"][from_data].values.astype(float) for i in range(1, order+1)]
    Taus = [data[T][f"Tau{i}"][from_data].values.astype(float) for i in range(1, order+1)] 
    OCV = data[T]["QOCV"][from_data].values.astype(float)
    OCV_SOC = data[T]["QOCV"][from_data+"SOC"].values.astype(float)
    SOC_grid = data[T]["SOCP"][from_data].values.astype(float)
    
    # Dictionary that holds the data points
    d = {f"R{i}": None for i in range(order+1)}
    d.update({f"Tau{i}": None for i in range(1,order+1)})
    d["R0"] = R0
    for i in range(1,order+1):
        d[f"R{i}"] = Rs[i-1]
        d[f"Tau{i}"] = Taus[i-1]
    d["OCV"] = OCV
    d["OCV_SOC"] = OCV_SOC
    d["SOC_grid"] = SOC_grid
    
    # Dictionary that holds the interpolated functions
    f = {f"R{i}": None for i in range(order+1)}
    f.update({f"Tau{i}": None for i in range(1,order+1)})
    f.update({"ocv_grid": None})
    
    # Interpolate the paramters of the ECM
    f["R0"] = np.polyfit(SOC_grid, R0, deg=deg)
    for i in range(1,order+1):
        f[f"R{i}"] = np.polyfit(SOC_grid, Rs[i-1], deg=deg)
        f[f"Tau{i}"] = np.polyfit(SOC_grid, Taus[i-1], deg=deg)
    deg_ocv = deg if deg_ocv==-1 else deg_ocv
    f["ocv_grid"] = np.polyfit(OCV_SOC, OCV, deg=deg_ocv)
    
    if plot == True:
        plot_ecm_parameter(d,f,deg,deg_ocv,order,figsize=figsize)
        if savefig == True:
            plt.savefig(f"ecm_{from_data}_parameter_interp_T{T}.pdf")
        plt.show()
    return f

def plot_ecm_parameter(d:dict,f:dict,deg:int,deg_ocv:int,order=2,figsize=(8,6))->None:
    """Helper function used to plot the ECM parameters interpolated in function interpolate_ecm_parameter

    Args:
        d (dict): Dictionary that holds the data points
        f (dict): Dictionary that holds the function interpolations
        deg (int): The chosen degree in interpolate_ecm_parameter
        deg_ocv (int): The chosen degree in interpolate_ecm_parameter
        order (int, optional): The chosen order in interpolate_ecm_parameter. Defaults to 2.
        figsize (tuple, optional): The chosen figure size in interpolate_ecm_parameter. Defaults to (8,6).

    Raises:
        ValueError: Checks if the number of parameters to plot matches the chosen order of the ECM
    """
    # Define the sublout layout
    num_parameters = len(f.keys())
    num_cols = 3
    num_rows = int(np.ceil(num_parameters/num_cols))
    if len(f.keys()) != order*2+2:
        raise ValueError("The ECM order and the number of paramters in the dictionary f are not matching")
    # Create the figure and subplots
    fig, axs = plt.subplots(nrows=num_rows,ncols=num_cols,figsize=figsize)
    axs = axs.flatten()
    
    # Loop over dictionary to plot the data
    for i, (key, value) in enumerate(f.items()):
        if key == "ocv_grid":
            axs[i].plot(d["OCV_SOC"], d["OCV"], "o", label="data points")
            axs[i].plot(d["OCV_SOC"], np.polyval(f["ocv_grid"], d["OCV_SOC"]), label=f"interpolation with a degree {deg_ocv}")
        else:
            axs[i].plot(d["SOC_grid"], d[key], "o", label="data points ")
            axs[i].plot(d["SOC_grid"], np.polyval(value, d["SOC_grid"]), label=f"interpolation with degree {deg}")
        axs[i].set(xlabel="SOC", ylabel=key)
        axs[i].grid()
        axs[i].legend(loc="best")
    # fig.suptitle(f"ECM Paramters at T={T}°C ({from_data} data)")
    plt.tight_layout()
    # plt.show()
    
def simulate_ecm(ecm:Nth_Order_ECM, current:list, order=2)->dict:
    """Function used to simulate an ECM. It iterates first over the list current. 
    Each element gets passed to the ECM's output equation (to determine the terminal voltage) 
    and state equation (to determine the state vector x).

    Args:
        ecm (Nth_Order_ECM): The ecm model to be simulated.
        current (list): List containing current values.
        order (int, optional): The order of the ECM in question. Defaults to 2.

    Returns:
        dict: Dictionary containing the logged values of the ECM.
    """
    # State vector
    x = ecm.state
    
    # Logfile to return
    log = {'V': [],
            'I': [],
            'SOC': []}
    log.update({f"V{i}":[] for i in range(1,order+1)}) # Add keys V1,V2,...
    
    # Simulation loop
    for u in current:
        y = ecm.output_equation(u,x) # Estimated terminal voltage
        x = ecm.state_equation(u,x) # Update state vector x
        x[0] = np.clip(x[0], 0, 1) # Force the SOC between values of 0 and 1
        ecm.update_parameter(x[0]) # Update SOC attribute of the ECM
        log["V"].append(y.item())
        log["I"].append(u)
        log["SOC"].append(x[0].item())
        for i in range(1,order+1):
            log[f"V{i}"].append(x[i].item())
    return log

def plot_ecm_log(log:dict, dt=1, figsize=(8,6))->None:
    """Plot the ECM simulation data from the log file.

    Args:
        log (dict): Dictionary holding the simulation data of the ECM
        dt (int, optional): Refers the time step. Used to create a time array for the plots. Defaults to 1.
        figsize (tuple, optional): Figure size of the plots. Defaults to (8,6).
    """
   
    # Define subplot layout
    num_parameters = len(log.keys())
    num_cols = 3
    num_rows = int(np.ceil(num_parameters/num_cols))
    
    # Create the figure and subplots
    fig, axs = plt.subplots(nrows=num_rows,ncols=num_cols,figsize=figsize)
    axs = axs.flatten() # Easier to iterate over axs, since it is a 2d array
    
    # define time array
    time = np.arange(len(log["V"]))*dt
    # Plot loop
    for i, (key,value) in enumerate(log.items()):
        axs[i].plot(time, value, label=key)
        axs[i].set_xlabel("Time [s]")
        if key == "I":
            axs[i].set_ylabel(key+" [A]")
        elif key == "SOC":
            axs[i].set_ylabel(key)
        else:
            axs[i].set_ylabel(key+" [V]")
        axs[i].grid()
    fig.suptitle("ECM log data")
    plt.tight_layout()
    plt.show()

    
def dominic_theta_coeff(data,from_data="mean"):
    """
    Function that returns the theta coefficients used for the ecm of Dominic
    """
    T = np.array([15,25,45])
    R0 = []
    R1 = []
    R2 = []
    Tau1 = []
    Tau2 = []
    SOC_data = data["15"]["SOCP"][from_data].values.astype(float)
    
    for i in T:
        R0.append(data[str(i)]["R0"][from_data].values.astype(float))
        R1.append(data[str(i)]["R1"][from_data].values.astype(float))
        R2.append(data[str(i)]["R2"][from_data].values.astype(float))
        Tau1.append(data[str(i)]["Tau1"][from_data].values.astype(float))
        Tau2.append(data[str(i)]["Tau2"][from_data].values.astype(float))
        
    T_data_mesh, SOC_data_mesh = np.meshgrid(T, SOC_data)
    
    # Dictionary containing thetas
    theta = {"R0":None,"R1":None,"R2":None,"Tau1":None,"Tau2":None}
    
    # Create design matrix
    Y = np.column_stack((np.ones_like(SOC_data_mesh).ravel(),
                     SOC_data_mesh.ravel(), T_data_mesh.ravel(),
                     (SOC_data_mesh**2).ravel(), (T_data_mesh**2).ravel(),
                     (SOC_data_mesh * T_data_mesh).ravel(),
                     (SOC_data_mesh**2 * T_data_mesh**2).ravel()))

    
    # Flatten R0, R1, R2, Tau1, and Tau2 lists
    R0_flat = np.concatenate(R0).ravel()
    R1_flat = np.concatenate(R1).ravel()
    R2_flat = np.concatenate(R2).ravel()
    Tau1_flat = np.concatenate(Tau1).ravel()
    Tau2_flat = np.concatenate(Tau2).ravel()
    
    # Calculate theta coefficients
    X = np.copy(Y)
    theta["R0"], _, _, _ = np.linalg.lstsq(X, R0_flat, rcond=None)
    X = np.copy(Y)
    theta["R1"], _, _, _ = np.linalg.lstsq(X, R1_flat, rcond=None)
    X = np.copy(Y)
    theta["R2"], _, _, _ = np.linalg.lstsq(X, R2_flat, rcond=None)
    X = np.copy(Y)
    theta["Tau1"], _, _, _ = np.linalg.lstsq(X, Tau1_flat, rcond=None)
    X = np.copy(Y)
    theta["Tau2"], _, _, _ = np.linalg.lstsq(X, Tau2_flat, rcond=None)
    
    
    return theta

def surface_2nd_order(soc, T, theta):
    return theta[0] + theta[1]*soc + theta[2]*T + theta[3]*soc**2 + \
           theta[4]*T**2 + theta[5]*soc*T + theta[6]*soc**2*T**2

def plot_surface_2nd_order(theta, T=[15,25,45],SOC=[0,1]):
    # Define the interpolation points
    T_plot = np.linspace(min(T), max(T), 100)
    SOC_plot = np.linspace(min(SOC), max(SOC), 100)

    T_grid, SOC_grid = np.meshgrid(T_plot, SOC_plot)

    for param_name, param_theta in theta.items():
        # Evaluate the surface for the current parameter
        param_surface = surface_2nd_order(SOC_grid, T_grid, param_theta)

        # Create the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(T_grid, SOC_grid, param_surface, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)

        # Set labels and title
        ax.set_xlabel('Temperature')
        ax.set_ylabel('SOC')
        ax.set_zlabel(param_name)
        ax.set_title(f'Interpolated {param_name} as a function of Temperature and SOC')

        # Add a color bar
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Show the plot
        plt.show()

def flatten_dict(d:dict, parent_key='', sep='_')->dict:
    """Helper function to flatten a nested dictionary. The function is used recursively since
    the dictionary nesting levels are equal.
    For example: my_dict["key1"]["key2"]["key3"]
    my_dict_flattened["key1_key2_key3"]
    If the dictionary is already flat, nothing happens.

    Args:
        d (dict): Dictionary to be flattened
        parent_key (str, optional): parent_key stores the key paths until it reaches the last level. Defaults to ''.
        sep (str, optional): _description_. Defaults to '_'.

    Returns:
        dict: Dictionary "d" flattened
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def scale_dict(data:dict, scaling_keys = ["V","T"])->tuple:
    """Helper function to scale a nested dictionary. At first it gets flattened.
    Then the global maxmimum and global minimum get searched for, by iterating over the flattened dictionary
    After finding these values, peform the scaling operation on the whole array.
    Finally returns the flattened dictionary, together with global minimum and maximum dictionary.
    

    Args:
        data (dict): Dictionary containing some measurement to be scaled.
        scaling_keys (list, optional): Contains the last characters of the nested dictionary's keys. Defaults to ["V","T"].

    Returns:
        tuple: Containing first the flattened dictionary and the minimum maximum dictioanry. The min_max_dict 
        is a dictionary of lists! The first element of the list represents the minimum, the second the maximum.
    """
    flattened_dict = flatten_dict(data)
    min_max_dict = {}

    for scaling_key in scaling_keys:
        global_min = min(min(values) for key, values in flattened_dict.items() if scaling_key == key[-1])
        global_max = max(max(values) for key, values in flattened_dict.items() if scaling_key == key[-1])
        min_max_dict[scaling_key] = (global_min, global_max) # update dictionary with global min and global max

        for key in flattened_dict.keys():
            if scaling_key == key[-1]:
                flattened_dict[key] = (np.array(flattened_dict[key]) - global_min) / (global_max - global_min)

    return flattened_dict, min_max_dict

def unscale_dict(scaled_data:dict, min_max_dict:dict)->dict:
    """Helper function to unscale a scaled dictionary using the min_max dictionary.

    Args:
        scaled_data (dict): Dictionary containing the data to be unscaled.
        min_max_dict (dict): Dictionary containing the global minimum and maximum of the flattened dictioanry.

    Returns:
        dict: Dictionary containing the unscaled data.
    """

    unscaled_data = {}

    for key, values in scaled_data.items():
        scaling_key = [scaling_key for scaling_key in min_max_dict.keys() if scaling_key == key[-1]]
        if scaling_key:
            global_min, global_max = min_max_dict[scaling_key[0]]
            unscaled_data[key] = values * (global_max - global_min) + global_min
        else:
            unscaled_data[key] = values

    return unscaled_data


def compare_nested_dic(dic1:dict,dic2:dict)->bool:
    """Helper function used to test if there is difference after scaling then unscaling a dictionary 

    Args:
        dic1 (dict): Dictionary unscaled.
        dic2 (dict): Dictionary scaled then unscaled.

    Returns:
        bool: True if no difference, else Flase
    """
    if dic1.keys()!=dic2.keys():
        return False
    for key in dic1:
        if isinstance(dic1[key], dict) and isinstance(dic2[key], dict):
            compare_nested_dic(dic1[key],dic2[key])
        else: # isinstance(dic1[key], (np.ndarray,list)) and isinstance(dic2[key], (np.ndarray,list)):
            if not np.allclose(np.array(dic1[key]), np.array(dic2[key])):
                print(dic1[key],"\n",dic2[key], "\n", key)
                return False 
    return True

# Custom LSTM pytorch dataset
class LSTM_dataset(Dataset):
    """This is a custom pytorch dataset. It is going to be used in the pytorch dataloader class.
    The purpose of this class is to generate and format the inputs the way the LSTM expects
    them to be. So in our case sequence_length, input_features.
    The custom dataset class needs to redefine the 3 methods:
        __init__
        __len__
        __getitem__
    

    Args:
        Dataset (torch.utils.data): Dataset object form PyTorch to load data from
    """
    def __init__(self, dic:dict, sequence_length:int, input_keys=["soc","I","T"], output_keys=["V"])->None:
        """Method to initialize the class attributes

        Args:
            dic (dict): dictioanry containing data to be stored in the dataset
            sequence_length (int): length of the sequence to be used for the dataset
            input_keys (list, optional): Contains the input keys for the LSTM model. THE ORDER IS IMPORTANT. Defaults to ["soc","I","T"].
            output_keys (list, optional): Contains the output keys for the LSTM model. Defaults to ["V"].

        Raises:
            ValueError: If the sequence length is bigger than the length of a dictionary value
        """
        self.dic = dic
        self.sequence_length = sequence_length
        self.input_size = len(input_keys)
        for key, value in self.dic.items():
            if self.sequence_length>=len(value):
                raise ValueError(f"List in dictionary at key = {key} has a length bigger or equal than the sequence_length")
        self.start_index = 0
        self.end_index = self.calculate_end_index()
        self.input_keys = input_keys
        self.output_keys = output_keys
        
    def calculate_end_index(self)->int:
        """Helper method used to calculate the end index of the dataset

        Returns:
            int: end index of the dataset
        """
        end_index = 0
        for key, value in self.dic.items():
            end_index = self.start_index+len(value)-self.sequence_length
            return end_index
        
    
    def __len__(self)->int:
        """Method to calculate the total length of the dataset

        Returns:
            int: length of the dataset
        """
        return self.end_index-self.start_index+1
    
    def __getitem__(self, index:int)->tuple:
        """Method used to get an input-output tuple from a dataset.

        Args:
            index (int): The index of the input-output tuple from the dataset

        Raises:
            IndexError: Raises error if index is out of bond.

        Returns:
            tuple: Containing 2 tensors: first one for the input_data, second one for the output_data.
        """
        input_data = []
        output_data = []
        if index < self.start_index or index > self.end_index:
            raise IndexError(f"Index {index} is out of bounds for list with length {self.__len__()} in dictionary's list")
        for key, value in self.dic.items():
            for input_key in self.input_keys:
                if input_key == key[-len(input_key):]:
                    input_data.append(value[index:self.sequence_length + index])  # Change from .append() to .extend()
            for output_key in self.output_keys:
                if output_key == key[-len(output_key):]:
                    output_data.append(value[index + self.sequence_length - 1])
        input_data = np.array(input_data) # Reshape inputs to have shape (sequence_length, 2)
        input_data = input_data.T
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32).view(-1, 1).squeeze(1)

    
class LSTM_model(nn.Module):
    """LSTM class model created to approximate an ECM. This LSTM class has only one fully connected layer.
    The LSTM's input shape needs to be in the following order: sequence length, batch size and input size

    Args:
        nn.Module (class): Base class for all NN models
    """
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, output_size:int)->None:
        """Method to initialize the class attributes

        Args:
            input_size (int): Input size of the LSTM.
            hidden_size (int): Number of hidden units of the LSTM. 
            num_layers (int): Number of LSTM layers.
            output_size (int): Output size of the LSTM.
        """
        super(LSTM_model, self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True) # LSTM layer
        self.fc = nn.Linear(hidden_size, output_size) # Linear layer: takes the output of the last LSTM layer and produces an output of size "output_size"
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """Method used to set how the information flows through the architecture. 

        Args:
            x (torch.Tensor): The information itself.

        Returns:
            torch.Tensor: An output tensor containing the output of the NN.
        """
        out, _ = self.lstm(x) # Forward propagate through the LSTM
        out = self.fc(out[:,-1,:]) # Only take the output from the last time step
        return out
    
def train_model(model, batch_size:int, train_set, device:str, optimizer:torch.optim.Optimizer, criterion= nn.MSELoss())->float:
    """Training function used to train a given model by using the following arguments.

    Args:
        model (LSTM_model or CNN_model): Either LSTM_model or CNN_model.
        batch_size (int): Represents how many samples to be computed at the same time
        train_set (either LSTM_dataset or CNN_1D_dataset): Custom dataset used in combination with a DataLoader. 
        device (str): Either "cpu", "cuda" or "mps". String representing the device on which the calculations are occuring.
        optimizer (callable): Optimizer used internally with PyTorch to optimize the gradient calculation.
        criterion (callable, optional): Loss function. Defaults to nn.MSELoss().

    Returns:
        float: The training loss.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)    
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device) # The unsqueeze to resolve a warning about mismatch of dimensions between inputs an targets
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss
    
def test_model(model, batch_size:int, test_set, device:str, criterion=nn.MSELoss(), return_prediction=False):
    """Function used to test the passed model on a given testing set.
    Args:
        model (LSTM_model or CNN_model): Either LSTM_model or CNN_model.
        batch_size (int): Represents the number of samples used per epoch
        test_set (either LSTM_dataset or CNN_1D_dataset): Custom dataset used in combination with a DataLoader.
        device (str): Either "cpu", "cuda" or "mps". String representing the device on which the calculations are occuring.
        criterion (callable, optional): loss function. Defaults to nn.MSELoss().
        return_prediction (bool, optional): Boolean variable. If False, then prediction won't be returned. Defaults to False.

    Returns:
        float or tuple: if return_prediction is False return loss value 
            otherwise tuple of float,list,list: the first list for the predicted output and the second list for the real output.  
    """
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    test_loss = 0
    model.to(device)
    model.eval()
    output_prediction = []
    output_real = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if return_prediction == True:
                output_prediction.extend(outputs.cpu().numpy())
                output_real.extend(targets.cpu().numpy())
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    if return_prediction == True:
        return test_loss, output_prediction, output_real
    else:
        return test_loss

def val_model(model, batch_size:int, val_set, device:str, criterion=nn.MSELoss())->float:
    """Validation function used to validate a given model by using the following arguments.

    Args:
        model (LSTM_model or CNN_model): Either LSTM_model or CNN_model.
        batch_size (int): Represents how many samples to be computed at the same time
        val_set (either LSTM_dataset or CNN_1D_dataset): Custom dataset used in combination with a DataLoader. 
        device (str): Either "cpu", "cuda" or "mps". String representing the device on which the calculations are occuring.
        criterion (callable, optional): Loss function. Defaults to nn.MSELoss().

    Returns:
        float: The validation loss.
    """
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss
    
    
def train_loop(model, dataset, optimizer, device, num_epochs=10, batch_size=64, train_ratio=0.7, criterion=nn.MSELoss(), print_perf=True):
    """Function used to loop over the training and validation function.

    Args:
        model (Either LSTM_model or CNN_model.): Either LSTM_model or CNN_model.
        dataset (Either LSTM_dataset or CNN_1D_dataset): Custom dataset to be splitted using the ratio argument. Used in combination with a DataLoader. 
        num_epochs (int, optional): Represents the number of times the training and validation functions will be executed. Defaults to 10.
        batch_size (int, optional): The number of inpute samples computed in parallel. Defaults to 64.
        train_ratio (float, optional): Represents the training ratio of the dataset to be used for training. Defaults to 0.7.
        criterion (torch.optim.Optimizer, optional): Loss function. Defaults to nn.MSELoss().

    Returns:
        tuple: contains the training loss and the validation loss of the respective dataset
    """
    # Define device variable
    
    train_length = int(len(dataset)*train_ratio)
    val_length = int(len(dataset)-train_length)
    
    train_set, val_set = random_split(dataset, [train_length, val_length])
    
    
    train_loss = 0
    val_loss = 0

    for epoch in range(num_epochs):
        train_loss = train_model(model=model, batch_size=batch_size, train_set=train_set, device=device, optimizer=optimizer, criterion=criterion)
        if print_perf:
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss}") 
        val_loss = val_model(model=model, batch_size=batch_size, val_set=val_set, device=device, criterion=criterion)
        if print_perf:
            print(f"Epoch: {epoch+1}, Validation Loss: {val_loss}") 
    # return (train_loss, val_loss)

def get_sub_dictionary(dictionary:dict, index:int, nb_keys=6, input_keys=["soc","I","T"], output_keys=["V"], print_message = True)->dict:
    """Helper function used to extract a sub-dictionary starting from a given index, upto a given index + nb_keys.

    Args:
        dictionary (dict): Dictionary to be looked inside.
        index (int): Starting index.
        nb_keys (int, optional): End index used to select the sub-dictionary. Defaults to 6.
        input_keys (list, optional): Number of input features. Defaults to ["soc","I","T"].
        output_keys (list, optional): Number of output features. Defaults to ["V"].
        print_message (bool, optional): Boolean, if True then print message, else not. Defaults to True.

    Returns:
        dict: The sub-dictionary
    """
    sub_dict = {}
    keys = list(dictionary.keys())[index:index+nb_keys]
    for key in keys:
        sub_dict[key] = dictionary[key]
    if print_message:
        print(f"The current Dataset's key is: {keys[0][:-1]}")     
    return sub_dict


def format_LGHE4C25B01(min_max_dict:dict, constant_temperature=False)->tuple:
    """Helper function used to scale the charge-discharge dataset according the dictionary min_max_dict

    Args:
        min_max_dict (dict): Dictionary containing the global minimum and maximum of the keys that need to be scaled.
        constant_temperature (bool, optional): If True, then fills an array with the same shape of the other arrays, but with a constant size. Defaults to False.

    Returns:
        tuple: _description_
    """
    data_lad, data_ela = load_data_LGHE4C25B01()
    T = list(data_lad.keys())
    data_lad_formatted = {}
    data_ela_formatted = {}
    for temperature in T:
        data_lad_formatted[temperature] =  {"V": data_lad[temperature]["U"].values.astype(np.float32),
            "I": data_lad[temperature]["I"].values.astype(np.float32),
            "soc": data_lad[temperature]["Ah"].values.astype(np.float32)/data_lad[temperature]['Cap'].values.astype(np.float32),
            "T": data_lad[temperature]["T"].values.astype(np.float32)}
        data_ela_formatted[temperature] = {"V": data_ela[temperature]["U"].values.astype(np.float32),
            "I": data_ela[temperature]["I"].values.astype(np.float32),
            "soc": data_ela[temperature]["Ah"].values.astype(np.float32)/data_ela[temperature]['Cap'].values.astype(np.float32),
            "T": data_ela[temperature]["T"].values.astype(np.float32)}
        if constant_temperature:
            constant_temperature = np.array([temperature], dtype=np.float32)
            data_lad_formatted[temperature]["T"] = np.full_like(data_lad[temperature]["T"].values.astype(np.float32), constant_temperature)
            data_ela_formatted[temperature]["T"] = np.full_like(data_ela[temperature]["T"].values.astype(np.float32), constant_temperature)
        if isinstance(min_max_dict, dict):
            for key in min_max_dict.keys():
                if key in data_lad_formatted[temperature] and key in data_ela_formatted[temperature]:
                    min_val = min_max_dict[key][0] # Contains the min
                    max_val = min_max_dict[key][1] # Contains the max
                    scaled_lad = (data_lad_formatted[temperature][key] - min_val) / (max_val - min_val)
                    scaled_ela = (data_ela_formatted[temperature][key] - min_val) / (max_val - min_val)
                    
                    data_lad_formatted[temperature][key] = scaled_lad
                    data_ela_formatted[temperature][key] = scaled_ela
                    
    return data_lad_formatted, data_ela_formatted



class CNN_1D_dataset(Dataset):
    """This is a custom pytorch dataset. It is going to be used in the pytorch dataloader class.
    The purpose of this class is to generate and format the inputs the way the CNN expects
    them to be. So in our case input_features, sequence_length.
    The custom dataset class needs to redefine the 3 methods:
        __init__
        __len__
        __getitem__
    

    Args:
        Dataset (torch.utils.data): Dataset object form PyTorch to load data from
    """
    def __init__(self, data, input_keys=["soc","I","T"], output_keys=["V"])->None:
        """Method to initialize the attributes of the custom dataset class.

        Args:
            data (List of dictionaries): In the case of 3 input features and 1 output feature:
            Each dictionary is going to have 2 keys: "inputs" and "outputs".
            The value of the key inputs is a 2D numpy array with the shape (number_of_input_features,
            sequence_length)
            The value of the key outputs is a 1D numpy array of floats. In our case the 1D array has
            one element: the terminal voltage.
        """
        self.data = data
        self.input_keys = input_keys
        self.output_keys = output_keys
    def __len__(self)->int:
        """Method that returns the size of the class attribute data

        Returns:
            int: length of the attribute data
        """
        return len(self.data)
    def __getitem__(self, index)->tuple:
        """Method that returns the input and output features at a given index.

        Args:
            index (int): parameter used by the class DataLoader of Pytorch to access the input an output features in the data attribute

        Returns:
            tuple: contains the input and output features as pytorch tensors
        """
        try:
            input=[self.data[index]["inputs"][input_key] for input_key in self.input_keys]
            output = [self.data[index]["outputs"][output_key] for output_key in self.output_keys]
            return torch.tensor(input, dtype=torch.float32), torch.tensor(output, dtype=torch.float32)
        except KeyError as e:
            print(f"Error at index {index}. Available keys in inputs: {self.data[index]['inputs'].keys()}")
            raise e
        
class CNN_model(nn.Module):
    """1D CNN class model created to approximate an ECM. This CNN class has only one fully connected layer.
    The CNN expects the input in the following shape:
    The LSTM's input shape needs to be in the following order: batch size, input size and sequence length

    Args:
        nn.Module (class): Base class for all NN models
    """
    def __init__(self, num_layers, num_neurons, sequence_length, input_keys:list=["soc","I","T"], output_keys:list=["V"]):
        """Method that initializes the class attributes. The kernel size represents the size of the window used for the convolutions.
        It is set to 3 since small Kernel sizes are efficient at capturing small patterns. The number of kernels is defined by num_neurons.
        There are still further andvantages of a small kernel size. See this: https://arxiv.org/abs/1603.07285

        Args:
            num_layers (int): The number of layers
            num_neurons (int): The number of neurons and the number of kernels.
            sequence_lenght (int): The sequence length of each input data sequence. All sequences must have the same size
            input_keys (list): List representing the names of the input features
            output_keys (list): List representing the names of the targets
        """
        super(CNN_model, self).__init__()
        layers = []
        for i in range(num_layers):
            in_channels = num_neurons if i > 0 else len(input_keys)
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=num_neurons, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
        self.convs = nn.Sequential(*layers) # layers is a list of Pytorch modules. The * beofre layer used for argument unpacking
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_neurons*sequence_length, len(output_keys))
    
    def forward(self, x):
        """Method that defines the forward pass of the model. It works like this:
            - it takes the input tensor x
            - applies the convolutional layers and the ReLU function stored in self.convs
            - takes the output of the last convolutional layer and flattens it
            - finally the flattened output gets fed into a fully connected linear layer

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, num_features, sequence_length)

        Returns:
            torch.tensor: the output tensor of the fully connected linear layer is the scaled predicted output,
            in the default case: the terminal voltage.
        """
        x = self.convs(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    

def CNN_1D_data_format(sub_dic:dict, formatted_data:list, sequence_length:int, input_keys=["soc", "I", "T"],output_keys=["V"])->None:
    """Helper function to format a group of subdictionaries from the flattened dictionary containing
    the schedules data. The function is used this way:

    for i in range(0, len(data), 6):
        sub_dic = get_sub_dictionary(data, i, print_message=False)
        CNN_1D_data_format(sub_dic=sub_dic, formatted_data=formatted_data, sequence_length=sequence_length)
    A step of 6 is needed because the flattened dictionary has keys: x_I, x_soc, x_V, x_T, x_V1, x_V2.
    The x here represent the routine used on the ECM. So please change it accordingly.

    Args:
        sub_dic (dict): subdictionary containing 6 keys
        formatted_data (list): 
        sequence_length (int): _description_
        input_keys (list, optional): _description_. Defaults to ["soc", "I", "T"].
        output_keys (list, optional): _description_. Defaults to ["V"].
    """
    length = 0
    for key, value in sub_dic.items():
        length = len(value)
        break
    for index in range(length-sequence_length+1):
        formatted_data.append({"inputs":{},"outputs":{}})
        for key, value in sub_dic.items():
            for input_key in input_keys:
                if input_key == key[-len(input_key):]:
                    formatted_data[-1]["inputs"].update({input_key:value[index:sequence_length + index]})
            for output_key in output_keys:
                if output_key == key[-len(output_key):]:
                    formatted_data[-1]["outputs"].update({output_key:value[sequence_length + index - 1]})

def rmse(x:np.ndarray,y:np.ndarray)->float:
    """Function to calculate the Root Mean Square Error from 2 arrays.

    Args:
        x (np.ndarray): Array containing the measurements.
        y (np.ndarray): Array containing the measured values.

    Returns:
        float: The RMSE value
    """
    return np.sqrt(np.mean((np.array(x)-np.array(y))**2))
