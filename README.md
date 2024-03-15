# downNO2_XGB
<img src="https://github.com/minsughim/downNO2_XGB/blob/main/figures/Graphical_abstract.png" width="100%">

 In this repository, we list all the scripts that were used for the project (published as Kim, Brunner, and Kuhlmann, 2021) from downloading, processing data to training a XGB model to predict high-resolution maps of near-surface NO<sub>2</sub> over an alpine region.

# Data download
This work includes multiple dataset 

1. TROPOMI data (satelite observation of NO<sub>2</sub> )
- Sentinel 5p data (Level 2) from http://www.temis.nl/airpollution/
- ./data_download/sentinel_data_down.py can be used

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

7. Population data
- https://publications.jrc.ec.europa.eu/repository/handle/JRC100523
- For world populatio ndta, other data source can be used at higher resolution

8. NO<sub>x</sub> emissions of point sources (TNO/MACC-3)
- https://acp.copernicus.org/articles/14/10963/2014/

# Data process 
The data process in the model is crucial in terms of training and producing maps. Several approaches that are unique in this study compared to other studies such Land Use Regression (LUR) models. Here, the target grid is set as the same as land use data (CORINE Land Cover 2018 - vector/raster 100 m) and all the data are regridded and interpolated to match with the land use grid.

1. Gap-filling and hourly linear interpolation of daily satellite observation
- Gap-filling and re-gridding was based on (Kuhlmann et al, 2014) https://amt.copernicus.org/articles/7/451/2014/ (./data_process/Sentinel_down_gridding.py) 
- Here, different QA values can be assigned either 50 or 75 for gap-filling
- Alternatively, re-gridding can be simply achieved using reindexing function in Xarray (./data_process/Sentinel_reindexing.py) 
- After regridding and gap-filling, daily observations are linearly interpolated for every hour. (./data_process/sentinel_hourly_interpolation.py) 

2. Length-wise decomposition of topogrophic features (using 2D wavelet transformation)
- ./data_process/dem_features.py generates multiple topological features using 2D wavelet tranfomation

3. Spatial interpolation of ERA5 at higher resolution 
- ./data_process/meteo_interp.py  linearly interpolate meteological data from ERA5 (30km scale) to 100m scale

4. Rasterisation of traffic data (line, vector data)
- ./data_process/nut2_traffic_rasterise.py
 
5. Magnitude and distant decomposition of emission data using gaussian convolution 
- ./data_process/data_gaussian_convolution.py

5. AQ station data pre-processing (data cleaning) 
- ./data_process/airbase_data_processing.py

6. Combining all the data to a core dataset for training models
- ./data_process/generate_core_data.py



# Model train
An XGB model can be trained using a pre-processed data (gap-filled, interpolated, AQ station-wise sorted). For the stratified sampling, umap-clustering method is used in the script. 
- ./model_train/xgb_train_save.py can be used 

# Predict and make NO<sub>2</sub> maps 
- STEP1: ./model_predict_map/predict_maps_month.py can be used to make prediction for a sub-region within the region of interest (ROI). The division of the ROI to multiple subregions are required to resolve the memory issue.

- STEP2: ./model_predict_map/merge_predicted_ROI_maps.py can be used to merge predictions of sub-regions to a single netCDF file (using xarray) and make a plot as shown in the publication


# Near-surface NO<sub>2</sub> concentrations in Alpine region (Switzerland and Northern Italy) during March 2019 (100m, hourly resolution)

https://github.com/minsughim/downNO2_XGB/assets/38297771/1cb24f3e-e414-4c90-ad81-69464b4fe67d

# Reference
**Minsu Kim**, Dominik Brunner, Gerrit Kuhlmann. Importance of satellite observations for high-resolution mapping of near-surface NO<sub>2</sub> by machine learning, _Remote Sensing of Environment_, __264__,112573,
[https://doi.org/10.1016/j.rse.2021.112573](https://doi.org/10.1016/j.rse.2021.112573 "Persistent link using digital object identifier"), 2021.

**Minsu Kim**, Gerrit Kuhlmann, and Dominik Brunner. Dataset for: Importance of satellite observations for high-resolution mapping of near-surface NO<sub>2</sub> by machine learning. *Zenodo*, 14(23):5403-5424, DOI: http://doi.org/10.5281/zenodo.5036643, 2021. 
