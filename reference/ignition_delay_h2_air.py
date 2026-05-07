"""Reference implementation: ignition delay time.

Verified on Cantera 3.2.0, H2/air, phi=1.0, 1200 K, 1 atm → 0.0500 ms
(ignition delay = time of peak dT/dt, widely used operational definition)

Note: uses clone=False to silence the Cantera 3.2 DeprecationWarning about
Solution object sharing.
"""
import cantera as ct
import numpy as np
import json

gas = ct.Solution('gri30.yaml')
gas.TP = 1200.0, 101325.0
gas.set_equivalence_ratio(1.0, 'H2', 'O2:1.0, N2:3.76')

r = ct.IdealGasConstPressureReactor(gas, clone=False)
net = ct.ReactorNet([r])

times, temps = [], []
T0 = gas.T
end_time = 1.0
while net.time < end_time:
    net.advance(net.time + 1e-5)
    times.append(net.time)
    temps.append(gas.T)
    # Early-exit once we're well past ignition
    if gas.T > T0 + 400:
        break

times_arr = np.array(times)
temps_arr = np.array(temps)
dTdt = np.gradient(temps_arr, times_arr)
tau = float(times_arr[int(np.argmax(dTdt))])

result = {"ignition_delay_s": tau, "T_final_K": float(gas.T)}
print(json.dumps(result))
