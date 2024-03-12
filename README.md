# downNO2_XGB
<img src="https://github.com/minsughim/downNO2_XGB/blob/main/figures/Graphical_abstract.png" width="100%">

 In this repository, we list all the scripts that were used for the project (published as Kim, Brunner, and Kuhlmann, 2021) from downloading, processing data to training a XGB model to predict high-resolution maps of near-surface NO<sub>2</sub> over an alpine region.

# Data download
This work includes multiple dataset 

1. TROPOMI data (satelite observation of NO<sub>2</sub> )
- Sentinel 5p data (Level 2) from http://www.temis.nl/airpollution/
- ./data_download/sentinel_data_down.py can be used
- 
2. Meteorological data (hourly sinlge level ERA5)
- https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
- ./data_download/ERA5_data_down.py can be used

3. Traffic data over Europe
- http://opentransportmap.info/
- ./data_download/opentf_data_down.py can be used

4. AIRBASE data (AQ stations) for near-surface NO<sub>2</sub> 
- http://discomap.eea.europa.eu/map/fme/AirQualityExport.htm
- ./data_download/airbase_data_down.py can be used
- This uses the already prepared list of download links saved in ./input/AIRBASE/Airbase_links.txt 
 
5. Land use data (CORINE Land Cover 2018 - vector/raster 100 m)
 - https://land.copernicus.eu/en/products/corine-land-cover/clc2018
 
6. Topography (The European Digital Elevation Model (EU-DEM) version 1.1)
- https://www.eea.europa.eu/en/datahub/datahubitem-view/d08852bc-7b5f-4835-a776-08362e2fbf4b
- 
# Data process 


# Model train and predict



# Near-surface NO<sub>2</sub> concentrations in Alpine region (Switzerland and Northern Italy) during March 2019 (100m, hourly resolution)


# Reference
**Minsu Kim**, Dominik Brunner, Gerrit Kuhlmann,(2021) Importance of satellite observations for high-resolution mapping of near-surface NO<sub>2</sub> by machine learning, _Remote Sensing of Environment_, __264__,112573,
[https://doi.org/10.1016/j.rse.2021.112573](https://doi.org/10.1016/j.rse.2021.112573 "Persistent link using digital object identifier")

**Minsu Kim**, Kuhlmann Gerrit, and Brunner Dominik. Dataset for: Importance of satellite observations for high-resolution mapping of near-surface NO<sub>2</sub> by machine learning. *Zenodo*, 14(23):5403-5424, DOI: http://doi.org/10.5281/zenodo.5036643, 2021. 
