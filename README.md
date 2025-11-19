# Battery RUL Prediction App 🔋

A Streamlit web application for predicting the Remaining Useful Life (RUL) of batteries using machine learning.

## Features

- Interactive web interface for battery RUL prediction
- Real-time predictions using Random Forest Regressor
- Input validation and user-friendly design
- Visual feedback on battery health status

## Prerequisites

Before running the application, ensure you have:

1. Python 3.8 or higher installed
2. The trained model files (`best_model.pkl` and `scaler.pkl`)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Generating Model Files

If you don't have the model files yet, run the Jupyter notebook to train and save the model:

1. Open `CI_project _battery.ipynb`
2. Run all cells to train the model
3. The notebook will generate:
   - `best_model.pkl` - The trained Random Forest model
   - `scaler.pkl` - The feature scaler

## Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Select Battery ID**: Choose from B5, B6, B7, or B18
2. **Enter Cycle Number**: Current cycle number of the battery
3. **Input Charging Parameters**:
   - Charging Current (chI)
   - Charging Voltage (chV)
   - Charging Temperature (chT)
4. **Input Discharging Parameters**:
   - Discharging Current (disI)
   - Discharging Voltage (disV)
   - Discharging Temperature (disT)
5. **Enter Battery State**:
   - Battery Capacity (BCt)
   - State of Health (SOH)
6. Click **"Predict RUL"** to get the prediction

## Model Information

- **Algorithm**: Random Forest Regressor with GridSearchCV
- **Features**: 10 input parameters
- **Target**: Remaining Useful Life (RUL) in cycles
- **Dataset**: Battery_dataset.csv

## Project Structure

```
.
├── app.py                          # Streamlit application
├── CI_project _battery.ipynb       # Model training notebook
├── Battery_dataset.csv             # Training dataset
├── best_model.pkl                  # Trained model (generated)
├── scaler.pkl                      # Feature scaler (generated)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Deployment

### Local Deployment

Follow the "Running the Application" section above.

### Cloud Deployment (Streamlit Cloud)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy the app

**Note**: Ensure `best_model.pkl` and `scaler.pkl` are included in your repository.

## Troubleshooting

**Model files not found error:**

- Make sure you've run the Jupyter notebook to generate the model files
- Verify that `best_model.pkl` and `scaler.pkl` are in the same directory as `app.py`

**Import errors:**

- Ensure all dependencies are installed: `pip install -r requirements.txt`

## License

This project is for educational purposes.
