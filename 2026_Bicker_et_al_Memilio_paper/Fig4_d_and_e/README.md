# Figure 4: Fits of the ODE SECIR-type model for SARS-CoV-2 related ICU cases in Germany and Spain for national and district level

## Requirements

- **MEmilio**: See `git_tag.cmake` for the required version.
- **Python**: Required for all simulations and plots. See the requirements.txt for the required python packages.
- **Shapefiles**: The map plots of Germany and Spain require shapefiles. We use the geodata from the [Bundesamt für Kartographie und Geodäsie](https://gdz.bkg.bund.de/index.php/default/open-data/verwaltungsgebiete-1-2-500-000-stand-31-12-vg2500-12-31.html) (VG2500 31.12.) for Germany and [Instituto Geográfico Nacional](https://centrodedescargas.cnig.es/CentroDescargas/limites-municipales-provinciales-autonomicos#licencias) for Spain.

## Files

- **Data gathering**:
    - `getSimulationDataSpain.py`: Get the data for the simulations of Spain.
- **Python Simulations**:
    - `graph_germany_nuts0.py`: Simulation, fitting and plots of the Germany model without spatial resolution (Fig. 4d, top).
    - `graph_germany_nuts3.py`: Simulation, fitting and plots of the Germany model with spatial resolution on county level (Fig. 4d, bottom).
    - `graph_spain_nuts0.py`: Simulation, fitting and plots of the Spain model without spatial resolution (Fig. 4e, top).
    - `graph_spain_nuts3.py`: Simulation, fitting and plots of the Spain model with spatial resolution on provinces level (Fig. 4e, bottom).

## How to Run

1.  **Install the project**:
    Configure the project using CMake as explained [here](../../README.md#configuring-using-cmake). This already installs the necessary packages for the simulations, which takes approximately 15 minutes.

2. **Download the required data**:
    Download the data for Germany by running 
    ```bash
    source build/venv/bin/activate
    python build/_deps/memilio-src/pycode/memilio-epidata/memilio/epidata/getSimulationData.py -o "build/_deps/memilio-src/data"
    ```
    or 
    ```bash
    source build/venv/bin/activate
    python getSimulationDataSpain.py
    ```
    for the simulations of Spain.

    Then you can run the simulations by executing the corresponding python scripts:
    ```bash
    python graph_germany_nuts0.py
    python graph_germany_nuts3.py
    python graph_spain_nuts0.py
    python graph_spain_nuts0.py
    ```