# ExoLyn

ExoLyn computes the vertical cloud structure in exoplanet atmospheres. It solves for the steady-state distribution of condensates and vapor across a 1D atmospheric column, taking into account diffusion, sedimentation, condensation, nucleation, and coagulation. The underlying physical model follows Ormel & Min (2019).

## Requirements

- **Python 3** (developed with Python 3.10)
- **NumPy** and **SciPy**
- **Matplotlib**
- **Meson** and **Ninja** (for compiling the Fortran solver)
- **gfortran** (Fortran compiler)
- **f2py** (bundled with NumPy; used to compile the Fortran solver)

Optional:

- **[Optool](https://github.com/cdominik/optool)** — required for calculating cloud optical constants (absorption, scattering, extinction, asymmetry parameter)
- **[FastChem](https://exoclime.github.io/FastChem/)** — used by post-processing scripts in `userfuncs/`
- **[petitRADTRANS](https://petitradtrans.readthedocs.io/)** — used by post-processing scripts in `userfuncs/`

## Installation

1. Clone the repository.
2. Compile the Fortran linear solver:

```bash
bash build.sh
```

This uses f2py to compile `src/sol.f90` into a shared object (`src/fmatrixsol.so`). The compiled module is architecture-specific; rebuild it if you move to a different system.

## Running ExoLyn

From the `src/` directory:

```bash
cd src
python3 relaxation.py
```

This reads `parameters.txt` from the current working directory by default. To use a different parameter file:

```bash
python3 relaxation.py /path/to/parameters.txt
```

To restart from an existing solution (interpolates onto a new grid):

```bash
python3 relaxation.py --restart grid_output.txt
```

Three example parameter files are provided:

| Example | Planet | Location |
|---------|--------|----------|
| Hot Jupiter | generic hot Jupiter | `examples/hotjupiter/parameters.txt` |
| Self-luminous | HR 8799e | `examples/selfluminous/parameters.txt` |
| Sub-Neptune | GJ 1214b | `examples/subneptune/parameters.txt` |

To run an example, copy its `parameters.txt` into `src/` and execute `python3 relaxation.py`.

## How It Works

ExoLyn solves a system of coupled diffusion-sedimentation-condensation equations on a 1D pressure grid. The solver proceeds in three stages:

1. **Initialization** — Find the atmospheric pressure domain and compute initial equilibrium solid concentrations using Newton-Raphson.
2. **Relaxation** — Gradually ramp up the "fudge parameters" `fdif` (diffusion) and `fsed` (sedimentation) from 0 to 1, using a binary-search stepping algorithm. At each step, the full system is solved with Newton-Raphson iterations until convergence.
3. **Finishing** — Write the output file and optionally calculate cloud optical constants.

The Newton-Raphson linear system is solved by a custom Fortran block-tridiagonal solver (`sol.f90`) for speed.

## The Parameter File

The parameter file (`parameters.txt`) is divided into two sections separated by a line of `=====`:

- **Above the separator**: parameter definitions in `name = value` format.
- **Below the separator**: chemical reactions, one per line (e.g., `Mg + SiO + 2H2O -> MgSiO3(s) + 2H2`).

### Parameter format

The parameter file supports Python-like expressions:

```python
R_star = cnt.R_sun               # reference constants via cnt.*
Rp = 1.087 * cnt.Rj             # arithmetic expressions
mn0 = 4 / 3 * np.pi * rho_int * an ** 3  # numpy functions and previous parameters
gas = ['Mg', 'SiO', 'H2O']      # lists
rootdir = '../../'               # strings
doptical = {optooldir='/path/'; multiproc=True}  # dictionaries
```

Available constants (`cnt.*`): `au`, `R_sun`, `Rj`, `R_earth`, `mu`, `kb`, `sigma_sb`, `R`, `c`, `h` (all in CGS).

### Key parameters

#### Stellar and planet properties

| Parameter | Description | Unit |
|-----------|-------------|------|
| `T_star` | Stellar effective temperature | K |
| `R_star` | Stellar radius | cm |
| `Rp` | Planet radius | cm |
| `rp` | Orbital radius | cm |
| `g` | Surface gravity | cm s^-2 |

#### Simulation control

| Parameter | Description | Unit |
|-----------|-------------|------|
| `runname` | Run identifier (used in output filename) | string |
| `rootdir` | Root directory of the ExoLyn installation | string |
| `gibbsfile` | Path to Gibbs energy data file | string |
| `N` | Number of atmospheric grid points | — |
| `Pa` | Upper boundary pressure | dyn cm^-2 |
| `Pb` | Lower boundary pressure | dyn cm^-2 |
| `Pref` | Reference pressure for Gibbs energy | dyn cm^-2 |
| `autobdrylow` | Auto-adjust lower boundary | bool |
| `autobdrytop` | Auto-adjust upper boundary | bool |

#### Temperature-pressure profile (Guillot 2010 analytic model)

| Parameter | Description | Unit |
|-----------|-------------|------|
| `opa_IR` | IR opacity | cm^2 g^-1 |
| `firr` | Radiation distribution factor | — |
| `opa_vis_IR` | Visual-to-IR opacity ratio | — |
| `T_int` | Internal temperature | K |

#### Cloud model parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `mgas` | Mean gas molecular weight | g |
| `Kzz` | Eddy diffusion coefficient | cm^2 s^-1 |
| `Kp` | Particle diffusivity | cm^2 s^-1 |
| `cs_mol` | Molecular cross section | cm^2 |
| `f_stick` | Sticking probability (vapor on particle) | — |
| `f_coag` | Sticking probability (particle on particle) | — |
| `cs_com` | Combined cross section (vapor + H2) | cm^2 |
| `rho_int` | Particle internal density | g cm^-3 |
| `an` | Nucleation radius | cm |
| `mn0` | Nucleation mass | g |
| `nuc_pro` | Nuclei production rate | g cm^-2 s^-1 |
| `sigma_nuc` | Width of nucleation profile | — |
| `P_star` | Nucleation height | dyn cm^-2 |

#### Chemistry

| Parameter | Description | Unit |
|-----------|-------------|------|
| `gas` | List of gas species names | list |
| `xvb` | Bottom-boundary mixing ratios for each gas | list |

#### Output and plotting

| Parameter | Description |
|-----------|-------------|
| `verbose` | Plot frequency: `silent`, `quiet`, `default`, or `verbose` |
| `plotmode` | Plot behavior: `all`, `save`, `popup`, or `none` |
| `writeoutputfile` | Whether to write the output file |

#### Optical constants

| Parameter | Description |
|-----------|-------------|
| `calcoptical` | Whether to compute optical constants |
| `optooldir` | Path to Optool installation |
| `doptical` | Dictionary with `optooldir`, `multiproc`, `multi_nproc` |

### Chemical reactions

Reactions are written below the `=====` separator in standard stoichiometric format:

```
Mg + SiO + 2H2O -> MgSiO3(s) + 2H2
```

Solid products are marked with `(s)`. All species (gas reactants/products and solid products) must be listed in the `gas` parameter, and Gibbs formation energy data must be available for every molecule involved.

## Output

The main output file is `grid<runname>.txt` (or `grid.txt` if `runname` is empty). It contains:

- **Line 1**: Comment line (`# <comment>`)
- **Line 2**: Key simulation parameters
- **Line 3**: Chemical reactions
- **Line 4**: Column header (`logP T(K) rhop(gcm-3) ap(cm) Sn <solids>(s)... <gases>... nuclei`)
- **Data rows**: One row per pressure layer with all quantities in scientific notation

If `calcoptical=True`, two additional directories are created:

- `meff/` — effective refractive indices for each layer (Bruggeman mixing)
- `coeff/` — absorption, scattering, extinction opacities and asymmetry parameter (via Optool)

## Adding New Reactions

1. Add the reaction below the `=====` separator in `parameters.txt`.
2. Add Gibbs formation energy data for every molecule involved in the reaction to the Gibbs energy file (`tables/gibbs_test.txt`). Data can be sourced from [janaf.nist.gov](https://janaf.nist.gov). Use `tables/download_janaf_table.py` to download JANAF tables:
   ```bash
   python3 tables/download_janaf_table.py <janaf_code> <species_name>
   ```
3. Add the solid density to `tables/density.txt`.
4. Ensure all gas and solid species appear in the `gas` parameter list in `parameters.txt`.
5. If optical constants are needed, add refractive index data (n, k vs wavelength) to `tables/nk/`.

## Code Structure

```
src/
  relaxation.py      Main driver: convergence control, iteration loop, output
  parameters.py       Reads and parses the parameter file
  chemistry.py        Molecules, reactions, Gibbs energy data
  functions.py        Physics: residual equations E(), Jacobian dEdy(), microphysics
  atmosphere_class.py Atmosphere data structure (grid, concentrations, derived quantities)
  init.py             Domain finding and initial equilibrium concentrations
  output.py           Writes the grid output file
  constants.py        Physical constants (CGS)
  sol.f90             Fortran block-tridiagonal linear solver
  fmatrixsol.so       Compiled Fortran module
util/
  draw.py             Plotting functions
  opticalnew.py       Effective medium + Mie scattering via Optool
  opticalchris.py     Alternative optical calculation pipeline
  optical.py          Original optical setup (deprecated)
  calmeff.py          Standalone effective medium calculations
  calkappa.py         Standalone opacity calculations
  testpars.py         Parameter space exploration tool
tables/
  gibbs_test.txt      Gibbs formation energies (JANAF)
  gibbsfit.txt        Fitted Gibbs energy coefficients
  density.txt         Solid material densities
  janaf_tables/       Raw JANAF data for 32+ species
  nk/                 Refractive index data for 28+ condensates
examples/
  hotjupiter/         Hot Jupiter example
  selfluminous/       HR 8799e example
  subneptune/         GJ 1214b example
```

## References

- Ormel, C. W. & Min, M. (2019) — The underlying cloud microphysics model
- Guillot, T. (2010) — Analytic temperature-pressure profile
- Woitke, P. et al. (2018) — Solid material densities
- Kitzmann, D. & Heng, K. (2018) — Refractive index data
- JANAF Thermochemical Tables (janaf.nist.gov) — Gibbs formation energies