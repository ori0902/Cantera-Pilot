import cantera as ct
import numpy as np
import math
import json
import sys

# simulation specification
spec = {
    "sim_type": "equilibrium",
    "fuel": "CH4",
    "oxidizer": "O2:1.0, N2:3.76",
    "phi": 1.0,
    "T": 300.0,
    "P": 101325.0,
    "mechanism": "gri30.yaml",
    "end_time": None
}

# extract simulation parameters
sim_type = spec["sim_type"]
fuel = spec["fuel"]
oxidizer = spec["oxidizer"]
phi = float(spec["phi"])
T = float(spec["T"])
P = float(spec["P"])
mechanism = spec["mechanism"]

# create a Cantera solution object from the mechanism
gas = ct.Solution(mechanism)

# set the gas state (temperature and pressure)
gas.TP = T, P

# set the equivalence ratio
gas.set_equivalence_ratio(phi, fuel, oxidizer)

# equilibrate the mixture adiabatically at constant P
gas.equilibrate('HP')

# extract the result (adiabatic flame temperature and pressure)
result = {
    "T_ad_K": float(gas.T),
    "P_Pa": float(gas.P)
}

print(json.dumps(result))