"""Deterministic normalization of raw spec dicts into validated SimulationSpec.

Everything in here is a Python dict lookup or arithmetic operation. No LLM.

Rationale: in v1, we tried to make the LLM remember that 'air' must be
expanded to 'O2:1.0, N2:3.76' before calling set_equivalence_ratio. That rule
was sometimes followed and sometimes not, and worse, my prompt was initially
*wrong* about the direction. These transformations are deterministic, so they
belong in Python, not in a prompt.

The Planner agent is free to emit shorthand like ``oxidizer='air'`` or
``T_value=25, T_unit='C'``; this module is the single place where shorthand
gets turned into the explicit Cantera-ready form.

v2.1 adds ``normalize_fuel``, which:
  - Resolves common name aliases to species symbols (e.g. 'butane' → 'C4H10').
  - Selects the appropriate Cantera mechanism file based on the fuel, because
    gri30.yaml only covers CH4/H2/CO and C1–C2 species.  Heavier fuels require
    a different mechanism (e.g. n-heptane → nheptane.yaml).  Failing to switch
    mechanisms produces a silent Cantera species-not-found error that exhausts
    all CodeGen retries without a useful diagnosis.
"""
from __future__ import annotations
from typing import Any
from schemas import SimulationSpec


# ---------------------------------------------------------------------------
# Oxidizer aliases
# ---------------------------------------------------------------------------

# Oxidizer shorthands → Cantera composition strings.
# Air composition: 21% O2, 79% N2 by volume → mole ratio O2:N2 = 1:3.76
OXIDIZER_ALIASES: dict[str, str] = {
    "air": "O2:1.0, N2:3.76",
    "dry air": "O2:1.0, N2:3.76",
    "atmosphere": "O2:1.0, N2:3.76",
    "o2": "O2:1.0",
    "oxygen": "O2:1.0",
    "pure oxygen": "O2:1.0",
}


# ---------------------------------------------------------------------------
# Fuel aliases and mechanism selection
# ---------------------------------------------------------------------------

# Maps common names / LLM shorthand → canonical Cantera species symbol.
# The symbol is what goes into set_equivalence_ratio(fuel=...).
FUEL_ALIASES: dict[str, str] = {
    # C1
    "methane": "CH4",
    "ch4": "CH4",
    # C2
    "ethane": "C2H6",
    "c2h6": "C2H6",
    "ethylene": "C2H4",
    "ethene": "C2H4",
    "c2h4": "C2H4",
    "acetylene": "C2H2",
    "c2h2": "C2H2",
    # C3
    "propane": "C3H8",
    "c3h8": "C3H8",
    "propylene": "C3H6",
    "propene": "C3H6",
    "c3h6": "C3H6",
    # C4
    "butane": "C4H10",
    "n-butane": "C4H10",
    "nbutane": "C4H10",
    "c4h10": "C4H10",
    "isobutane": "IC4H10",
    "i-butane": "IC4H10",
    "ic4h10": "IC4H10",
    # C7
    "heptane": "NC7H16",
    "n-heptane": "NC7H16",
    "nheptane": "NC7H16",
    "nc7h16": "NC7H16",
    # H2 / CO / Syngas
    "hydrogen": "H2",
    "h2": "H2",
    "carbon monoxide": "CO",
    "co": "CO",
}

# Maps canonical species symbol → recommended Cantera mechanism file.
#
# Design notes:
#   - gri30.yaml: covers CH4, H2, CO, C2H2, C2H4, C2H6, C3H6, C3H8.
#     The GRI-Mech species list stops at C3.  C4+ species are absent.
#   - nDodecane_Lu.yaml: high-temperature n-alkane mechanism bundled with
#     Cantera 3.x; covers C4H10 and IC4H10 among others.
#   - nheptane.yaml: n-heptane reduced mechanism, also bundled.
#   - Fallback: gri30.yaml — safest default for unrecognized species because
#     Cantera will raise a clear "unknown species" error rather than silently
#     producing wrong results.
#
# Only the *default* mechanism is set here.  The Planner can still override
# ``mechanism`` explicitly (e.g. for a user that specifies their own .yaml),
# and normalize_spec will pass it through unchanged if the key is present.
_SPECIES_TO_MECHANISM: dict[str, str] = {
    # GRI-Mech species
    "CH4": "gri30.yaml",
    "H2": "gri30.yaml",
    "CO": "gri30.yaml",
    "C2H2": "gri30.yaml",
    "C2H4": "gri30.yaml",
    "C2H6": "gri30.yaml",
    "C3H6": "gri30.yaml",
    "C3H8": "gri30.yaml",
    # C4 — requires a mechanism that includes these species
    "C4H10": "nDodecane_Lu.yaml",
    "IC4H10": "nDodecane_Lu.yaml",
    # C7
    "NC7H16": "nheptane.yaml",
}

# Human-readable note shown in logs when mechanism is auto-selected.
_MECH_REASON: dict[str, str] = {
    "nDodecane_Lu.yaml": "gri30.yaml does not contain C4+ species; switched to nDodecane_Lu.yaml",
    "nheptane.yaml": "gri30.yaml does not contain C7 species; switched to nheptane.yaml",
}


# ---------------------------------------------------------------------------
# Phi aliases
# ---------------------------------------------------------------------------

# Phi shorthand. Users say "stoichiometric" / "lean" / "rich" more often than
# numeric values. Planner returns these; we map to canonical floats.
PHI_ALIASES: dict[str, float] = {
    "stoichiometric": 1.0,
    "stoich": 1.0,
    "lean": 0.7,
    "very lean": 0.5,
    "rich": 1.3,
    "very rich": 1.6,
}


# ---------------------------------------------------------------------------
# Pressure unit conversions → Pascals (Cantera's base unit)
# ---------------------------------------------------------------------------

PRESSURE_CONVERSIONS: dict[str, float] = {
    "pa": 1.0,
    "pascals": 1.0,
    "kpa": 1_000.0,
    "bar": 100_000.0,
    "atm": 101_325.0,
    "atmosphere": 101_325.0,
    "atmospheres": 101_325.0,
    "psi": 6_894.76,
    "torr": 133.322,
    "mmhg": 133.322,
}


# ---------------------------------------------------------------------------
# Public normalizer functions
# ---------------------------------------------------------------------------

def normalize_oxidizer(oxidizer: str) -> str:
    """Expand shorthand like 'air' to a Cantera composition string.

    Pass-through if already explicit (contains ':').

    Args:
        oxidizer: Raw oxidizer string from the Planner.

    Returns:
        A Cantera-ready composition string such as 'O2:1.0, N2:3.76'.
    """
    o = oxidizer.strip()
    if ":" in o:
        return o  # already a composition string
    return OXIDIZER_ALIASES.get(o.lower(), o)


def normalize_fuel(fuel: str) -> tuple[str, str | None]:
    """Resolve a fuel name to its Cantera species symbol and recommended mechanism.

    Accepts common names ('butane', 'methane') or direct species symbols
    ('C4H10', 'CH4') in any case.

    Args:
        fuel: Raw fuel string from the Planner.

    Returns:
        A 2-tuple of (species_symbol, mechanism_or_None).
        ``mechanism_or_None`` is the recommended .yaml file if the default
        gri30.yaml does not contain the species, otherwise None (caller
        should keep whatever mechanism was already set).

    Raises:
        ValueError: If the fuel string is empty after stripping.
    """
    raw = fuel.strip()
    if not raw:
        raise ValueError("fuel must not be empty")

    # Resolve alias (case-insensitive lookup, fall back to uppercased raw)
    symbol = FUEL_ALIASES.get(raw.lower(), raw.upper())

    # Recommend a mechanism only when the species needs a non-GRI mechanism
    recommended_mech = _SPECIES_TO_MECHANISM.get(symbol)
    # If it maps to gri30 we don't need to force an override
    if recommended_mech == "gri30.yaml":
        recommended_mech = None

    return symbol, recommended_mech


def normalize_phi(phi_value: Any) -> float:
    """Accept numeric phi or shorthand string and return a float.

    Args:
        phi_value: Equivalence ratio as a number or descriptive string
                   such as 'stoichiometric', 'lean', 'rich'.

    Returns:
        Equivalence ratio as a Python float.

    Raises:
        ValueError: If a string alias is unrecognized or cannot be parsed.
        TypeError: If the type is neither numeric nor string.
    """
    if isinstance(phi_value, (int, float)):
        return float(phi_value)
    if isinstance(phi_value, str):
        s = phi_value.strip().lower()
        if s in PHI_ALIASES:
            return PHI_ALIASES[s]
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"Unrecognized phi value: {phi_value!r}")
    raise TypeError(f"phi must be a number or string, got {type(phi_value).__name__}")


def normalize_temperature(value: float, unit: str = "K") -> float:
    """Convert a temperature value to Kelvin.

    Args:
        value: Numeric temperature in the given unit.
        unit:  Unit string — 'K'/'Kelvin', 'C'/'Celsius', or 'F'/'Fahrenheit'.
               Case-insensitive; degree symbols are stripped.

    Returns:
        Temperature in Kelvin as a float.

    Raises:
        ValueError: For unrecognized unit strings.
    """
    unit = unit.strip().lower().replace("°", "")
    if unit in ("k", "kelvin", ""):
        return float(value)
    if unit in ("c", "celsius"):
        return float(value) + 273.15
    if unit in ("f", "fahrenheit"):
        return (float(value) - 32) * 5.0 / 9.0 + 273.15
    raise ValueError(f"Unknown temperature unit: {unit!r}")


def normalize_pressure(value: float, unit: str = "Pa") -> float:
    """Convert a pressure value to Pascals.

    Args:
        value: Numeric pressure in the given unit.
        unit:  Unit string ('Pa', 'kPa', 'bar', 'atm', 'psi', 'Torr', 'mmHg').
               Case-insensitive.

    Returns:
        Pressure in Pascals as a float.

    Raises:
        ValueError: For unrecognized unit strings.
    """
    u = unit.strip().lower()
    if u in PRESSURE_CONVERSIONS:
        return float(value) * PRESSURE_CONVERSIONS[u]
    raise ValueError(f"Unknown pressure unit: {unit!r}")


def normalize_spec(raw: dict) -> SimulationSpec:
    """Convert a raw Planner output dict into a validated SimulationSpec.

    Expected keys in ``raw``:
      - ``sim_type``              : str  — 'equilibrium', 'ignition_delay', 'flame_speed'
      - ``fuel``                  : str  — species symbol or common name
      - ``oxidizer``              : str  — shorthand or explicit composition (optional, default 'air')
      - ``phi``                   : number or string
      - ``T_value`` + ``T_unit``  : preferred temperature input shape
        OR ``T``                  : number in Kelvin
      - ``P_value`` + ``P_unit``  : preferred pressure input shape
        OR ``P``                  : number in Pa
      - ``mechanism``             : str  — optional; auto-selected if absent
      - ``end_time``              : float — optional; ignition-delay end time (s)

    Args:
        raw: Dictionary produced by the Planner agent.

    Returns:
        A fully validated :class:`SimulationSpec`.

    Raises:
        KeyError: If a required key is missing from ``raw``.
        ValueError: If any value fails normalization or Pydantic validation.
    """
    out: dict[str, Any] = {}

    out["sim_type"] = raw["sim_type"]

    # Fuel: resolve alias → species symbol, auto-pick mechanism if needed
    fuel_symbol, auto_mech = normalize_fuel(raw["fuel"])
    out["fuel"] = fuel_symbol

    out["oxidizer"] = normalize_oxidizer(raw.get("oxidizer", "air"))
    out["phi"] = normalize_phi(raw["phi"])

    # Temperature — support two input shapes
    if "T_value" in raw:
        out["T"] = normalize_temperature(raw["T_value"], raw.get("T_unit", "K"))
    else:
        out["T"] = normalize_temperature(raw["T"], "K")

    # Pressure — same pattern
    if "P_value" in raw:
        out["P"] = normalize_pressure(raw["P_value"], raw.get("P_unit", "Pa"))
    else:
        out["P"] = normalize_pressure(raw["P"], "Pa")

    # Mechanism: explicit Planner choice > auto-selected > default gri30
    planner_mech = raw.get("mechanism")
    if planner_mech:
        out["mechanism"] = planner_mech  # user/Planner override — respect it
    elif auto_mech:
        out["mechanism"] = auto_mech
        # Surface the reason in logs (orchestrator will print the spec dict)
        reason = _MECH_REASON.get(auto_mech, f"auto-selected {auto_mech} for {fuel_symbol}")
        print(f"[normalize] mechanism auto-selected: {reason}")
    else:
        out["mechanism"] = "gri30.yaml"

    if "end_time" in raw and raw["end_time"] is not None:
        out["end_time"] = float(raw["end_time"])

    return SimulationSpec(**out)
