# Mapping Mining Areas in the Tropics

Mining provides crucial materials for the global economy and the climate transition, but has potentially severe adverse environmental and social impacts. Currently, the analysis of such impacts is obstructed by the poor availability of data on mining activity --- particularly in regions most affected.

In this paper, we present a novel panel dataset of mining areas in the tropical belt from 2016 to 2024. We use a transformer-based segmentation model, trained on an extensive dataset of mining polygons from the literature, to automatically delineate mining areas in satellite imagery over time.

The resulting dataset features improved accuracy and reduced noise from human errors, and can readily be extended to cover new locations and points in time as they become available.
Our comprehensive dataset of mining areas can be used to assess local environmental, social, and economic impacts of mining activity in regions where conventional data is not available or incomplete.

This repository contains all code used to produce the dataset and an example of the required directory structure, as well as the required Python packages. Further, for model training and prediction, a directory for MMSegmentation (https://mmsegmentation.readthedocs.io/en/main/get_started.html) is required.
