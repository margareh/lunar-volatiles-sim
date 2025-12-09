# Lunar Volatiles Sim
Simulation for generating synthetic distributions of craters, lunar volatiles, and associated instrument measurements from a neutron spectrometer (NSS). This proceeds by cratering a surface over geologic time scales, implanting volatiles, and updating the surface model and volatiles distribution at each time step until the "present day."

Example output is shown below for a 200 x 200 m map, with the surface on the left and the PSR mask on the right:

<img width="400" height="416" alt="hillshade_dem_crop" src="https://github.com/user-attachments/assets/2de048ab-1eaa-49b5-bc83-a748363c3e3f" />
<img width="424" height="416" alt="final_psr_mask" src="https://github.com/user-attachments/assets/0fa5a542-925e-48c9-a419-89d60f91eb6f" />

The corresponding depth to lunar volatiles, volatile weight %, and observations for both detectors of the NSS are shown below:

<img width="1000" height="960" alt="nss_obs" src="https://github.com/user-attachments/assets/03b18a8f-86cc-488f-8680-82daaed8b910" />


# Installing the Sim
The ```install.sh``` script installs all packages that are included as part of this simulation: raytrace_cuda, illumination_cuda, diffusion_cuda, synthterrain, moonpies, lvsim

Alternatively, individual packages can be built with ```python3.9 -m pip install .``` in each directory.

## Dependencies
This simulation requires the following packages, tested with the provided version numbers:
- **Python == 3.9**
- **CUDA == 12.4**
- **PyTorch == 2.6.0+cu124**
- GPyTorch == 1.13
- numpy == 1.26.4
- matplotlib == 3.9.4
- pandas == 2.3.1
- pyproj == 3.6.1
- shapely == 2.0.7
- rasterio == 1.4.3
- scikit-image == 0.24.0
- imageio == 2.37.0
- richdem == 0.3.4
- tqdm == 4.67.1

All of the above requirements are found in ``requirements.txt`` **_except for the three in bold_**. These must be installed separately, after which running ```pip install -r requirements.txt``` should install the remainder of the requirements.

This simulation has been built with CUDA integration to speed up performance. CUDA 12.4 was used on an NVIDIA GeForce RTX 3060 GPU. Performance on other platforms has not been tested.

# Running the Sim
The sim can be run using the CLI command ```lvsim```. Simply running this command will produce a 1000 x 1000 pixel map at a resolution of 1 meter/pixel, starting at 3.8 Gyrs in the past and taking time steps of 100 Myr.

The default values can be altered to produce smaller maps using command line arguments. For instance, the command below produces a 200 x 200 pixel map, with other defaults kept the same:
```
lvsim --bbox 0 200 200 0 --max_range 0.04 --d_lim 2 200
```
The ```bbox``` option defines the bounding box of the map, while ```d_lim``` specifies the diameters of craters to generate and should be changed to reflect the size of the map. Likewise, ```max_range``` is a parameter that affects the horizon modeling and must be decreased from its default value (0.2 for a 1000 px map) for good performance on smaller maps.

The output directory can be changed with the ```outpath``` option.
