# Cantera Multi-Agent LLM Tool

> **Final project for CSC 7644: Applied LLM Development**

A natural-language interface to [Cantera](https://cantera.org/) combustion
simulations. Ask a question in plain English; a four-agent LLM pipeline plans
the simulation, generates and validates Cantera Python code, executes it in a
sandbox, and returns a concise interpreted result.

---

## Project Overview

Combustion simulation tools like Cantera are powerful but require expert
knowledge of the API, mechanism files, species naming conventions, and unit
systems. This project wraps that complexity in an LLM agent pipeline so that
researchers and students can request simulations conversationally:

```
"What is the adiabatic flame temperature of stoichiometric butane/air at 1 atm, 300 K?"
"Give me the ignition delay of H2/air at phi=0.5, 1200 K, 10 atm."
"Laminar flame speed of rich propane/air at 500 K?"
```

The system handles unit conversions, fuel/oxidizer normalization, mechanism
selection, and traceback-driven code repair automatically — without requiring
the user to know any Cantera API details.

---

## Key Features

- **Three simulation types**: adiabatic flame temperature (HP equilibrium),
  ignition delay (constant-pressure reactor), and laminar flame speed (1D
  freely-propagating flame).
- **Natural-language input** via a CLI — no Cantera knowledge required.
- **Deterministic normalization** of fuel names (`butane` → `C4H10`),
  oxidizer shorthands (`air` → `O2:1.0, N2:3.76`), equivalence ratio aliases
  (`stoichiometric` → `1.0`), and unit conversions (°C/°F → K, bar/atm → Pa).
- **Automatic mechanism selection** — switches from `gri30.yaml` to the
  appropriate bundled mechanism when a heavier fuel (C4+, C7) is requested.
- **RAG-augmented code generation** — relevant Cantera documentation chunks
  are retrieved via ChromaDB and injected into the CodeGen prompt.
- **Traceback-driven retry** — up to three CodeGen attempts, each informed by
  the previous execution error or plausibility-validation failure.
- **Physical plausibility validators** that catch silent order-of-magnitude
  errors (wrong equilibration mode, unit mixup) before they reach the user.
- **Full audit trail** saved per run: `record.json`, each generated script,
  and stderr — useful for debugging and evaluation.

---

## Tech Stack and Architecture

| Layer | Technology |
|---|---|
| LLM backend | [Ollama](https://ollama.com/) (local) — default model `llama3.1:8b` |
| Combustion engine | [Cantera 3.2](https://cantera.org/) |
| Vector store / RAG | [ChromaDB](https://www.trychroma.com/) + `BAAI/bge-small-en-v1.5` embeddings |
| Schema validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Testing | [pytest](https://pytest.org/) |

### Agent pipeline

```
User query (CLI)
     │
     ▼
 Planner agent          ← LLM, JSON mode; extracts sim_type, fuel, phi, T, P
     │
     ▼
 normalize_spec()       ← Pure Python; unit conversion, fuel/oxidizer/mechanism resolution
     │
     ▼
 RAG retriever          ← ChromaDB; fetches relevant Cantera doc chunks
     │
     ▼
 ┌── CodeGen agent ──┐  ← LLM; writes a self-contained Cantera Python script
 │   Executor         │  ← Sandboxed subprocess with import allowlist + timeout
 │   Validators       │  ← Physical-plausibility bounds check
 │   (retry × 3)      │  ← Each retry includes prior code + traceback/hint
 └───────────────────┘
     │
     ▼
 Analysis agent         ← LLM; interprets numeric result in plain English
     │
     ▼
 RunRecord (JSON log) + CLI output
```

---

## Setup Instructions

### Prerequisites

- **Python 3.11+** (tested on 3.11 and 3.12)
- **Ollama** installed and running locally — see <https://ollama.com/download>
- Operating system: Linux or macOS recommended; Windows works but Cantera
  installation may require conda.

### 1. Pull the LLM

```bash
ollama pull llama3.1:8b
```

Any Ollama-compatible model can be substituted by editing `MODEL` in
`config.py`.

### 2. Create and activate a virtual environment

```bash
python -m venv cantera_pilot
source cantera_pilot/bin/activate      # Windows: cantera_pilot\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: Cantera is easiest to install via conda if pip fails:
> ```bash
> conda install -c cantera cantera
> ```

### 4. Environment variables / API keys

This project uses **Ollama running locally** — no external API keys are
required. No `.env` file is needed.

If you swap the LLM backend to a remote provider (e.g. OpenAI), add your key
as an environment variable and update `llm_client.py`:

```bash
export OPENAI_API_KEY="sk-..."
```

### 5. Build the RAG index

```bash
python -m rag.build_index
```

This ingests the Cantera documentation in `data/docs/` and the reference
scripts in `reference/` into ChromaDB. The index is stored in `rag/chroma_db/`
(git-ignored). Retrieval still works if you skip this step — it will simply
return no context chunks.

---

## Running the Application

```bash
source cantera_pilot/bin/activate

python run.py --query "adiabatic flame temperature of stoichiometric CH4/air at 1 atm, 300 K"
python run.py -q "ignition delay of H2/air at phi=1, 1200 K, 1 atm"
python run.py -q "laminar flame speed of butane/air phi=1.0 at 1 atm, 300 K"
python run.py -q "flame speed of n-heptane/air stoichiometric 1 atm 400 K"

# Suppress per-step progress output
python run.py -q "..." --quiet
```

### Output

On success the CLI prints:

```
RESULT
======================================================================
Adiabatic flame temperature of CH4/air (phi=1.0) at 101325 Pa, 300 K: 2226 K
...
Key values:
  T_ad_K: 2226.3
CodeGen attempts: 1
Total time: 4.2s
```

On failure it prints the error and last stderr excerpt, and exits with code 1.

### Running the test suite

```bash
# Gate test — must pass before adding new agents
python -m pytest tests/test_executor.py -v

# Full suite
python -m pytest tests/ -v
```

---

## Repository Organization

```
.
├── agents/
│   ├── planner.py          # Planner agent: NL query → raw spec dict (LLM, JSON mode)
│   ├── codegen.py          # CodeGen agent: spec + docs → Cantera Python script (LLM)
│   └── analysis.py         # Analysis agent: numeric result → plain-English report (LLM)
│
├── rag/
│   ├── build_index.py      # One-time script: ingest docs into ChromaDB
│   ├── retriever.py        # retrieve(spec) → relevant doc chunks string
│   └── chroma_db/          # ChromaDB persistence directory (git-ignored)
│
├── sandbox/
│   └── executor.py         # Sandboxed subprocess runner with import allowlist + timeout
│
├── tests/
│   ├── test_executor.py    # Gate tests: three reference scripts produce correct values
│   ├── test_normalize.py   # Unit tests for all normalizer functions
│   └── test_validators.py  # Unit tests for plausibility validators
│
├── reference/              # Hand-verified Cantera 3.2 scripts (ground truth for CodeGen)
│   ├── equilibrium_ref.py
│   ├── ignition_delay_ref.py
│   └── flame_speed_ref.py
│
├── data/
│   └── docs/               # Cantera documentation excerpts used for RAG
│
├── logs/                   # Per-run audit trails (git-ignored)
│
├── config.py               # Central configuration: model, paths, RAG params
├── schemas.py              # Pydantic models: SimulationSpec, AnalysisReport, RunRecord
├── normalize.py            # Deterministic spec normalization (no LLM)
├── validators.py           # Physical-plausibility bounds checks
├── orchestrator.py         # End-to-end pipeline: Planner → CodeGen loop → Analysis
├── llm_client.py           # Ollama wrapper used by all agents
├── run.py                  # CLI entrypoint
└── requirements.txt
```

---

## Configuration

All tunable parameters are in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `llama3.1:8b` | Ollama model tag |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `MAX_CODEGEN_RETRIES` | `3` | Max CodeGen+execute attempts per query |
| `EXECUTION_TIMEOUT_SEC` | `180` | Subprocess timeout (1D flame solve can take 30–60 s) |
| `RETRIEVAL_K` | `8` | Number of doc chunks retrieved from ChromaDB |
| `RERANK_K` | `3` | Top-K after reranking |

---

## Supported Fuels and Mechanisms

| Fuel | Species symbol | Mechanism | flame_speed | equilibrium | ignition_delay |
|---|---|---|---|---|---|
| Methane | `CH4` | `gri30.yaml` | ✓ | ✓ | ✓ |
| Hydrogen | `H2` | `gri30.yaml` | ✓ | ✓ | ✓ |
| Ethane | `C2H6` | `gri30.yaml` | ✓ | ✓ | ✓ |
| Ethylene | `C2H4` | `gri30.yaml` | ✓ | ✓ | ✓ |
| Propane | `C3H8` | `gri30.yaml` | ✓ | ✓ | ✓ |
| n-Dodecane | `c12h26` | `nDodecane_Reitz.yaml` | ✗ | ✓ | ✓ |

> **Note on flame speed and transport data**: `ct.FreeFlame` requires a
> mechanism that includes transport properties (viscosity, thermal conductivity,
> diffusion coefficients). `nDodecane_Reitz.yaml` ships without transport data
> (`transport model: none`), so `flame_speed` queries for heavy fuels will fail.
> For flame speed simulations, use a GRI-Mech fuel (CH4, H2, C3H8, etc.), or
> supply a custom mechanism file that includes a transport section.
>
> Butane (C4H10) and n-heptane (NC7H16) are not present in any bundled Cantera
> mechanism on this install. Queries for those fuels are automatically redirected
> to n-dodecane (`c12h26`) as the nearest available heavy fuel. See
> [Using a Downloaded Mechanism File](#using-a-downloaded-mechanism-file) to
> add true butane or heptane support.

---

## Using a Downloaded Mechanism File

The bundled Cantera mechanisms cover a limited set of fuels. If you need a
fuel that isn't supported (e.g. true butane C4H10, iso-octane, ethanol), you
can download a mechanism and drop it in so Cantera can find it.

### Step 1 — Find a mechanism

Good sources:
- [Cantera Community Mechanisms](https://github.com/Cantera/cantera-examples)
- [NIST Chemical Kinetics Database](https://kinetics.nist.gov/)
- [LLNL Combustion Mechanisms](https://combustion.llnl.gov/mechanisms)
- Your institution's combustion lab (many publish `.yaml` or `.cti` files)

Make sure the mechanism is in Cantera's `.yaml` format (or `.cti` for older
versions). Some repositories distribute Chemkin-format files — use
`ck2yaml` (bundled with Cantera) to convert them:

```bash
python -m cantera.ck2yaml --input=mech.inp --thermo=therm.dat --output=mymech.yaml
```

### Step 2 — Find your Cantera data directory

```bash
python -c "import cantera as ct; print(ct.get_data_directories())"
```

This prints a list of directories. Copy your `.yaml` file into the **first**
one listed (usually inside your virtual environment):

```bash
cp mymech.yaml /path/to/cantera_pilot/lib/python3.x/site-packages/cantera/data/
```

### Step 3 — Verify Cantera can load it

```bash
python -c "
import cantera as ct
gas = ct.Solution('mymech.yaml')
print('Species count:', gas.n_species)
print('Has C4H10:', 'C4H10' in gas.species_names)
"
```

### Step 4 — Register it in `normalize.py`

Open `normalize.py` and add your fuel and mechanism to the two tables:

```python
# In FUEL_ALIASES — map any name variants to the species symbol
FUEL_ALIASES = {
    ...
    "butane": "C4H10",
    "n-butane": "C4H10",
    "c4h10": "C4H10",
}

# In _SPECIES_TO_MECHANISM — map the symbol to your mechanism file
_SPECIES_TO_MECHANISM = {
    ...
    "C4H10": "mymech.yaml",
}
```

After that, queries like `"flame speed of butane/air phi=1 at 1 atm 300 K"`
will automatically load your mechanism — no other changes needed.

---

## Attributions and Citations

- **Cantera** — D. G. Goodwin et al., *Cantera: An Object-oriented Software
  Toolkit for Chemical Kinetics, Thermodynamics, and Transport Processes*,
  v3.2, 2024. <https://cantera.org>
- **GRI-Mech 3.0** — Smith et al., UC Berkeley.
  <http://combustion.berkeley.edu/gri-mech/>
- **ChromaDB** — Chroma AI, <https://www.trychroma.com/>
- **BGE embeddings** — BAAI/bge-small-en-v1.5 via `sentence-transformers`,
  <https://huggingface.co/BAAI/bge-small-en-v1.5>
- **Ollama** — <https://ollama.com/>
- Agent architecture inspired by patterns discussed in course materials for
  CSC 7644: Applied LLM Development.
