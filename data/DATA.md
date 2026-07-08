# Dataset Overview

**Airline Passenger Satisfaction** (Kaggle)

- **File**: `airline-passenger-satisfaction/train_clean_encoded_balanced.csv` (ALL features)
- **Size**: 103,904 training records (after missing-value removal and SMOTE balancing)
- **Target**: Binary classification (satisfied=1, neutral/dissatisfied=0)
- **Total Features**: 23 after one-hot encoding
  - 4 continuous: Age, Flight_Distance, Departure_Delay, Arrival_Delay
  - 14 service ratings (1–5): WiFi, Online_boarding, Seat_comfort, etc.
  - 5 categorical (one-hot encoded): Gender, Customer_Type, Type_of_Travel, Class

**Optimal 9-feature subset** (selected by XGBoost at 80% cumulative gain threshold):
```
['Online_boarding', 'Type_of_Travel_Personal Travel', 'Class_Eco', 
 'Inflight_wifi_service', 'On_board_service', 'Customer_Type_disloyal Customer', 
 'Inflight_entertainment', 'Checkin_service', 'Leg_room_service']
```

**Data usage**:
- **Model training (SPLIT/RESPLIT/PRAXIS)**: Use 9-feature subset (same features as DIMEX baseline)
- **Feature importance comparison**: Use all 23 features (compare XGBoost importance vs. PRAXIS RID on the 9 selected features)