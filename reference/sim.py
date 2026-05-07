"""Reference implementation: laminar flame speed.

Verified on Cantera 3.2.0, CH4/air, phi=1.0, 300 K, 1 atm → 0.3851 m/s
(reference value ~0.38 m/s per GRI-Mech 3.0)

FreeFlame is the standard 1D freely-propagating premixed flame solver.
The flame speed is the velocity at the inlet (f.velocity[0]).
"""
import cantera as ct
import json

gas = ct.Solution('gri30.yaml')
gas.TP = 300.0, 101325.0
gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1.0, N2:3.76')

f = ct.FreeFlame(gas, width=0.03)
f.set_refine_criteria(ratio=3, slope=0.1, curve=0.2)
f.solve(loglevel=0, auto=True)

result = {"flame_speed_m_s": float(f.velocity[0])}
print(json.dumps(result))
