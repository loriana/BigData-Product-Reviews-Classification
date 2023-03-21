# BigData-Product-Reviews-Classification
Pipeline for data cleaning and running an ML classifier to decide if a review is helpful or not helpful.

## Usage

```
python -m pr_pipeline [-h] 
    --path-to-data PATH_TO_DATA
    [--preprocess]
    [--missing-val-report]
```
General args:
- **`--path-to-data`** (required): The absolute path where your data is stored
- **`--preprocess`**: Generates the preprocess train, validate and test files and save them in `./preprocessing_output`
- **`--missing-val-report`**: Generates a report on terminal to visualize potential missing values




