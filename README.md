# Vehicle Trajectory Prediction Using Machine Learning

An end-to-end machine learning project that predicts future vehicle movements using historical trajectory data from the **Waymo Open Motion Dataset**. The project includes data preprocessing, trajectory prediction, model evaluation, and an interactive Streamlit dashboard.

## Project Overview

Understanding how vehicles move is an important part of autonomous driving systems. Predicting a vehicle's future trajectory can help autonomous vehicles anticipate movement and make better driving decisions.

The objective of this project is to use historical vehicle positions to predict where a vehicle may move over the next 8 seconds.

The project explores trajectory prediction using a Relative Linear Regression model and evaluates its predictions against actual vehicle movements.

## Key Features

- Processing vehicle trajectory data from Waymo scenarios.
- Extracting historical and future vehicle positions.
- Preparing training, validation, and test datasets.
- Predicting future vehicle trajectories using machine learning.
- Evaluating predictions using Average Displacement Error (ADE) and Final Displacement Error (FDE).
- Interactive Streamlit dashboard for exploring individual vehicle predictions.
- Visualization of observed, actual, and predicted trajectories.

## Dataset

**Source:** [Waymo Open Motion Dataset](https://waymo.com/open/)

The experiment uses one training TFRecord shard from the Waymo Open Motion Dataset.

| Dataset Statistic | Value |
|---|---:|
| Scenarios processed | 496 |
| Complete vehicle trajectories | 1,167 |
| Training vehicles | 823 |
| Validation vehicles | 174 |
| Test vehicles | 170 |
| Test scenarios | 69 |

The dataset was divided into training, validation, and test sets using scenario-based grouping to prevent scenarios from overlapping across the splits.

**Note:** The complete Waymo dataset was not used in this experiment.

## Methodology

### 1. Data Preprocessing

- Extracted vehicle trajectories from scenario TFRecord files.
- Selected vehicle agents with complete observation and future trajectories.
- Used 11 historical frames as input.
- Used the following 80 frames as prediction targets.
- Applied relative coordinates by centering positions around the last observed position.

Each frame represents 0.1 seconds.

- Historical observation: 1.1 seconds.
- Future prediction horizon: 8 seconds.

### 2. Model Development

**Selected Model: Relative Linear Regression**

The model uses the vehicle's historical movement to estimate its future positions.

Instead of directly predicting absolute coordinates, the model learns future positions relative to the vehicle's last observed position. This makes predictions less dependent on the vehicle's absolute location in a scenario.

Other approaches, including Random Forest and LSTM, were also explored. Relative Linear Regression achieved better validation performance among the approaches evaluated.

### 3. Evaluation Metrics

**Average Displacement Error (ADE)**

Measures the average Euclidean distance between predicted and actual positions across all future frames.

**Final Displacement Error (FDE)**

Measures the Euclidean distance between predicted and actual positions at the final prediction frame.

Lower values indicate more accurate predictions.

## Model Results

The selected Relative Linear Regression model was evaluated on the held-out test dataset.

| Metric | Test Result |
|---|---:|
| Average Displacement Error (ADE) | 7.03 meters |
| Final Displacement Error (FDE) | 20.68 meters |

The model was evaluated on 170 test vehicles across 69 scenarios.

### Prediction Error Over Time

The following results show average prediction error at different future horizons on the validation dataset.

| Future Horizon | Average Error |
|---|---:|
| 1 second | 0.36 m |
| 2 seconds | 1.33 m |
| 4 seconds | 5.57 m |
| 6 seconds | 12.48 m |
| 8 seconds | 21.60 m |

The results show that prediction error increases as the future prediction horizon becomes longer.

## Interactive Dashboard

The project includes an interactive Streamlit dashboard that allows users to explore model predictions.

Dashboard features:

- Dataset overview.
- Interactive test vehicle selection.
- Individual vehicle ADE and FDE.
- Observed vehicle trajectory visualization.
- Actual versus predicted future trajectory comparison.
- Overall test-set performance.
- Prediction error over different future time horizons.

## Technologies Used

- Python
- TensorFlow
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Waymo Open Motion Dataset

## Running the Dashboard

Clone the repository:

```bash
git clone https://github.com/VaibhavKela/waymo-vehicle-trajectory-prediction.git
cd waymo-vehicle-trajectory-prediction
```

Create and activate a Python virtual environment, then install the required dependencies.

For example, install Streamlit using:

```bash
uv pip install streamlit
```

Run the dashboard:

```bash
streamlit run app.py
```

The dashboard expects the prepared test arrays, prediction file, and prediction-error CSV to be available in the `outputs/` directory.

The original dataset and generated model artifacts are not included in this README. They must be obtained or generated separately.

## Limitations

- The experiment uses only one training TFRecord shard rather than the complete Waymo dataset.
- The selected model uses historical position information and does not explicitly model road geometry, traffic signals, or interactions between agents.
- Linear regression may struggle to predict complex vehicle behavior, such as sharp turns or sudden changes in direction.
- Prediction error increases over longer forecasting horizons.

## Future Improvements

- Improve trajectory prediction using additional motion features.
- Incorporate road geometry and map information.
- Explore interaction-aware trajectory prediction.
- Investigate more advanced sequence-based prediction architectures.
- Evaluate performance on a larger portion of the Waymo dataset.

## Author

**Vaibhav Kela**  
M.S. Computer Science  
University of Colorado Denver

GitHub: [VaibhavKela](https://github.com/VaibhavKela)
