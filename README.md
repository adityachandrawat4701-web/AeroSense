# AeroSense ✈️
## About the Project

I started AeroSense out of curiosity about Machine Learning. I wanted to understand how ML actually works, how models are trained, how datasets are created and how they make predictions.

For this, I explored a virtual flight simulation that generates data for different flight phases like Ground, Climb, Cruise, Descent and Landing.

The generated data is used to train a **Decision Tree Classifier**, which tries to identify the flight phase based on the given parameters.

The main purpose of this project was to learn ML practically. Flight simulation was just a way to generate data for the experiment.

## How it works

The project follows this basic workflow:

```text
       Flight Simulation
               |
               v
      Generated Flight Data
               |
               v
     Sensor Disturbances
               |
               v
        Dataset Creation
               |
               v
        Data Preparation
               |
          +----+----+
          |         |
          v         v
      Training    Testing
        Data        Data
          |         |
          v         |
     Model Training |
          |         |
          v         |
     Trained Model  |
          |         |
          +----+----+
               |
               v
       Model Evaluation
               |
               v
      Streamlit Dashboard
```

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## Why Sensor Disturbances?

In real life, sensor measurements are not always perfect. They can contain noise and variations.

I wanted to explore this, so synthetic disturbances were introduced into the simulated sensor data to make the dataset less idealized.

## Results and Limitations

The model achieved approximately 94–95% accuracy (to be verified).

Some limitations:

- The simulation is simplified.
- The dataset is synthetically generated.
- The model has not been tested on real aircraft data.
- The results cannot be considered representative of real-world flight conditions.

## AI Disclosure

**This entire project was generated using AI.**

I did not independently write the code. I used ChatGPT to generate the implementation, including the simulation, dataset generation, ML model and dashboard.

I started this project mainly to explore how Machine Learning works in practice. I provided the initial idea and directions, including the idea of introducing sensor disturbances, and used the generated project as a way to learn about the different components involved.

This is an educational experiment, not an independently developed or production-ready system.

## Final Note

AeroSense was simply my first attempt at exploring Machine Learning through a practical project. It helped me get a basic idea of how data generation, model training and prediction work together.
