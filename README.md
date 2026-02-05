# OcuTemp Backend — Occupancy-Based AC Control with ML

## Overview
OcuTemp is a web-based backend system that predicts **optimal air-conditioning setpoints** based on room temperature and humidity using a **Random Forest ML model**. It supports **multi-room inputs** and is designed to work with ESP32 sensors or simulated room data.  

This repository contains the **Flask backend** and **trained ML model**, ready for testing and deployment.  

---

## Features
- Accepts **temperature and humidity readings** from multiple rooms  
- Predicts **comfort status** (`too_hot`, `too_cold`, `comfortable`)  
- Returns **confidence level** of prediction  
- Suggests **recommended AC setpoint**  
- Multi-room support (via room IDs)  


---

## Project Structure
OcuTemp-Backend/
├── app.py # Main Flask server
├── model/
│ └── comfort_model.pkl # Trained Random Forest model
├── requirements.txt # Python dependencies
└── README.md