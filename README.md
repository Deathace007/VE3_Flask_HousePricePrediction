# House Price Prediction System

## Overview
An advanced machine learning system that predicts house prices using a Random Forest model with extensive feature engineering and hyperparameter optimization. The system includes a user-friendly web interface built with Flask.

## Features
### Model Features
- Random Forest Regressor with optimized hyperparameters
- Comprehensive feature engineering including:
  - Time-based features (house age, renovation age)
  - Area-based features (total area, area per room)
  - Location features (schools nearby, airport distance)
- Automated data preprocessing pipeline
- Cross-validation and grid search for optimal performance
- Feature importance analysis

### Web Interface
- Clean, modern UI with Bootstrap styling
- Interactive form with input validation
- Helpful tooltips and value ranges
- Responsive design for all devices
- Detailed price prediction results

## Technical Stack
- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, Bootstrap
- **Data Processing**: Automated pipeline with StandardScaler

## Setup Instructions

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
   jupyter notebook Model_Training.ipynb
   ```
   Run all cells to generate `model.pkl`

4. **Start the Flask application:**
   ```bash
   python app.py
   ```

5. **Access the web interface:**
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser

## Project Structure
```
├── app.py                    # Flask application
├── Model_Training.ipynb      # Jupyter notebook for model training
├── House Price India.csv     # Dataset
├── model.pkl                 # Trained model and pipeline
├── requirements.txt          # Python dependencies
├── static/
│   └── style.css            # Custom CSS styles
└── templates/
    ├── index.html           # Main prediction form
    └── result.html          # Prediction results page
```

## Model Features Used
- Number of bedrooms and bathrooms
- Living area and lot area
- House condition and grade
- Number of floors and views
- Waterfront presence
- Built year and renovation year
- Location features (latitude, longitude)
- Proximity features (schools, airport)
- Engineered features (total area, room ratios, etc.)

## Performance Metrics
- RMSE (Root Mean Square Error)
- R² Score
- MAE (Mean Absolute Error)

## Future Improvements
- Support for GPU acceleration
- Additional feature engineering
- Model ensemble techniques
- API endpoint for predictions
- Extended visualization options

## Notes
- The model is trained on the House Price India dataset
- Prices are predicted in Indian Rupees (₹)
- Input validation ensures prediction accuracy
- The system uses an automated ML pipeline for reproducibility

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Notes
- You can use your own dataset by replacing `housing.csv` and updating the code accordingly.
- The model is a simple linear regression for demonstration purposes.
