# Aftershock Forecasting Project

## Overview

This project builds machine learning models to forecast aftershocks using seismic data. The objective is to predict aftershock rates and magnitudes based on historical earthquake data. We focus on using the Southern California earthquake catalog from 1932 to 2025 (Mw ≥ 3), to train a compact ConvLSTM (Convolutional Long Short-Term Memory) model. The resulting tool can forecast the number of aftershocks expected within 30 days and their distribution across different magnitudes. This model is lightweight, operates in milliseconds on standard hardware, and can be used for real-time aftershock forecasting.

## Project Structure

The repository contains the following files and directories:

- **figures/**: Directory containing figures and plots generated during analysis.
- **README.md**: Project description and setup instructions.
- **aftershock_forecasting_full_analysis.ipynb**: Jupyter Notebook for full analysis.
- **convLSTM.ipynb**: **Main notebook for the project**, which includes all model training, evaluation, tuning, binning, graphing, and results analysis. This notebook is where the core ConvLSTM model is implemented and optimized.- **data_reader_and_PINNs_omori_experiments.ipynb**: Python script for reading the Southern California catalogue and converting it into a structured CSV file.
- **.DS_Store**: A system file for macOS (can be ignored).

## Installation Instructions

1. **Clone the Repository:**

git clone https://github.com/khalil-alhaj01/aftershock_forecast.git


2. **Install Dependencies:**

This project requires Python 3.x and several libraries. You can install the required dependencies by using the following command:

pip install -r requirements.txt

**Note**: If a `requirements.txt` file is not available, you can manually install the following libraries:

- TensorFlow
- NumPy
- pandas
- Matplotlib
- Scikit-learn
- Keras

3. **Dataset:**

The project uses the Southern California earthquake catalog (January 1932 – April 2025, Mw ≥ 3). The `data_reader_and_PINNs_omori_experiments.ipynb` script processes this data, extracting essential information such as event timestamps, latitude, longitude, depth, and magnitude. It then converts the data into a structured CSV file for further analysis and modeling.

## Instructions for Reproducing Results

1. **Run the Notebooks:**

- Open the `aftershock_forecasting_full_analysis.ipynb` notebook to start the full analysis. This includes data preprocessing, model training, and evaluation.
- For training the ConvLSTM model, use the `convLSTM.ipynb` notebook.
- For data preparation using the Southern California catalog, refer to `data_reader_and_PINNs_omori_experiments.ipynb`.

2. **Training the Model:**

Follow the instructions in each notebook to train the models. The notebooks include cells for loading the dataset, preprocessing it, training the models, and evaluating performance.

3. **Results:**

After running the models, visualize the results using the figures saved in the `figures/` directory.

## Key Contributions

- **End-to-End Catalogue Pipeline**: The entire pipeline, from data extraction to model forecasting, is built using only the earthquake catalog. The `data_reader_and_PINNs_omori_experiments.ipynb` script parses raw data files from the Southern California Seismic Network (SCSN) to extract necessary seismic event features.
- **Lightweight and Real-Time Forecasting**: The project uses a ConvLSTM model trained on the Southern California catalog (1932-2025) to predict aftershock rates and their distribution across magnitudes.
- **Sequence-Specific Forecasting**: The model forecasts aftershock occurrence based on the spatio-temporal sequence of prior events, without the need for waveform data or external physics-based inputs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

