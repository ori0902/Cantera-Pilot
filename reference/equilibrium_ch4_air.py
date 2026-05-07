"""Reference implementation: adiabatic flame temperature (equilibrium).

Verified on Cantera 3.2.0, CH4/air, phi=1.0, 300 K, 1 atm → 2225.52 K
(reference value ~2226 K per GRI-Mech 3.0)

This file is a gold-standard fixture used by:
- tests/test_executor.py — confirms the sandbox can run correct Cantera code
- agents/codegen.py — the structure the LLM should produce
"""
import cantera as ct
import json

gas = ct.Solution('gri30.yaml')
gas.TP = 300.0, 101325.0
gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1.0, N2:3.76')
gas.equilibrate('HP')

result = {"T_ad_K": float(gas.T), "P_Pa": float(gas.P)}
print(json.dumps(result))
