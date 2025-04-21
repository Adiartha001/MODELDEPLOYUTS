#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model dan encoder
model = joblib.load('XG_booking_status.pkl')
booking_status_encode = joblib.load('booking_status_encode.pkl')
oneHot_encode_room = joblib.load('oneHot_encode_room.pkl')
oneHot_encode_meal = joblib.load('oneHot_encode_meal.pkl')
oneHot_encode_mark = joblib.load('oneHot_encode_mark.pkl')

def predict_booking_status(no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
                            type_of_meal_plan, required_car_parking_space, room_type_reserved, lead_time,
                            arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest,
                            no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
                            avg_price_per_room, no_of_special_requests):

    input_data = pd.DataFrame({
        'no_of_adults': [no_of_adults],
        'no_of_children': [no_of_children],
        'no_of_weekend_nights': [no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'required_car_parking_space': [required_car_parking_space],
        'lead_time': [lead_time],
        'arrival_year': [arrival_year],
        'arrival_month': [arrival_month],
        'arrival_date': [arrival_date],
        'repeated_guest': [repeated_guest],
        'no_of_previous_cancellations': [no_of_previous_cancellations],
        'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
        'avg_price_per_room': [avg_price_per_room],
        'no_of_special_requests': [no_of_special_requests]
    })

    meal_encoded = oneHot_encode_meal.transform([[type_of_meal_plan]])
    room_encoded = oneHot_encode_room.transform([[room_type_reserved]])
    market_encoded = oneHot_encode_mark.transform([[market_segment_type]])

    full_input = np.hstack([input_data.values, meal_encoded, room_encoded, market_encoded])
    prediction = model.predict(full_input)
    output = list(booking_status_encode.keys())[list(booking_status_encode.values()).index(prediction[0])]
    return output

def display_prediction_result(result, label):
    color = 'green' if result == 'Not_Canceled' else 'red'
    st.markdown(f"**{label}:** <span style='color:{color}; font-weight:bold;'>{result}</span>", unsafe_allow_html=True)

# Header
st.markdown("""<h2 style='text-align: center;'>Hotel Booking Status Prediction</h2>""", unsafe_allow_html=True)
st.markdown("""<p style='text-align: center;'>Nama: <b>Adiartha Wibisono Hasnan</b> | NIM: <b>2702315236</b></p>""", unsafe_allow_html=True)
st.markdown("---")

# Input Manual
st.subheader("Input Manual")
with st.form("manual_input_form"):
    no_of_adults = st.number_input("No of Adults", 0, 100)
    no_of_children = st.number_input("No of Children", 0, 100)
    no_of_weekend_nights = st.number_input('No of Weekend Night', 0, 2)
    no_of_week_nights = st.number_input('No of Week Night', 0, 5)
    type_of_meal_plan = st.selectbox('Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    required_car_parking_space = st.radio('Required Car Parking Space', [0, 1])
    room_type_reserved = st.selectbox('Room Type Reserved', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 
                                                            'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    lead_time = st.number_input("Lead Time (days)", 0, 360)
    arrival_year = st.selectbox("Arrival Year", [2017, 2018])
    arrival_month = st.selectbox('Arrival Month', list(range(1,13)))
    arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
    market_segment_type = st.selectbox('Market Segment Type', ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])
    repeated_guest = st.radio('Repeated Guest', [0, 1])
    no_of_previous_cancellations = st.number_input('Previous Cancellations', 0, 100)
    no_of_previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', 0, 100)
    avg_price_per_room = st.number_input('Average Price Per Room (in Euros)', 0.00, 10000.00)
    no_of_special_requests = st.number_input('Number of Special Requests', 0, 100)

    submitted = st.form_submit_button("Predict")
    if submitted:
        hasil = predict_booking_status(
            no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
            type_of_meal_plan, required_car_parking_space, room_type_reserved,
            lead_time, arrival_year, arrival_month, arrival_date, market_segment_type,
            repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
            avg_price_per_room, no_of_special_requests
        )
        display_prediction_result(hasil, "Prediction Result")

# Test Case 1
st.markdown("---")
st.subheader("Test Case 1")
st.markdown("""
<b>Input:</b><br>
- Adults: 2<br>
- Children: 0<br>
- Weekend Nights: 1<br>
- Week Nights: 2<br>
- Meal Plan: Meal Plan 1<br>
- Parking: Yes<br>
- Room Type: Room_Type 1<br>
- Lead Time: 10<br>
- Arrival: 2018-5-15<br>
- Market Segment: Offline<br>
- Repeated Guest: Yes<br>
- Prev Cancel: 0<br>
- Prev Not Cancel: 3<br>
- Price: 75.0<br>
- Special Requests: 2<br>
""", unsafe_allow_html=True)

if st.button("Run Test Case 1"):
    hasil = predict_booking_status(
        no_of_adults=2,
        no_of_children=0,
        no_of_weekend_nights=1,
        no_of_week_nights=2,
        type_of_meal_plan='Meal Plan 1',
        required_car_parking_space=1,
        room_type_reserved='Room_Type 1',
        lead_time=10,
        arrival_year=2018,
        arrival_month=5,
        arrival_date=15,
        market_segment_type='Offline',
        repeated_guest=1,
        no_of_previous_cancellations=0,
        no_of_previous_bookings_not_canceled=3,
        avg_price_per_room=75.0,
        no_of_special_requests=2
    )
    display_prediction_result(hasil, "Test Case 1 Result")

# Test Case 2
st.subheader("Test Case 2")
st.markdown("""
<b>Input:</b><br>
- Adults: 1<br>
- Children: 2<br>
- Weekend Nights: 2<br>
- Week Nights: 3<br>
- Meal Plan: Not Selected<br>
- Parking: No<br>
- Room Type: Room_Type 6<br>
- Lead Time: 200<br>
- Arrival: 2017-12-31<br>
- Market Segment: Online<br>
- Repeated Guest: No<br>
- Prev Cancel: 3<br>
- Prev Not Cancel: 0<br>
- Price: 300.0<br>
- Special Requests: 0<br>
""", unsafe_allow_html=True)

if st.button("Run Test Case 2"):
    hasil = predict_booking_status(
        no_of_adults=1,
        no_of_children=2,
        no_of_weekend_nights=2,
        no_of_week_nights=3,
        type_of_meal_plan='Not Selected',
        required_car_parking_space=0,
        room_type_reserved='Room_Type 6',
        lead_time=200,
        arrival_year=2017,
        arrival_month=12,
        arrival_date=31,
        market_segment_type='Online',
        repeated_guest=0,
        no_of_previous_cancellations=3,
        no_of_previous_bookings_not_canceled=0,
        avg_price_per_room=300.0,
        no_of_special_requests=0
    )
    display_prediction_result(hasil, "Test Case 2 Result")


# In[ ]:




