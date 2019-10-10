# Harmonization

Here we perform harmonization with ComBat and train ml models to harmonize unseen data using image features.

## Organize data from our local server.

Combine data from different datasets the dataset and run harmonization.

```
bash prepare_data.sh
```

It outputs ```./data/combined/mriqc.csv```, which contains the relative volumes of ROIs and the metrics from [MRIQC](https://mriqc.readthedocs.io) tool.
