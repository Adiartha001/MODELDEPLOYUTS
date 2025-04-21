#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install joblib


# In[3]:


import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

class DataProcessor:
    def __init__(self):
        self.df = None
        self.room_enc = OneHotEncoder(sparse=False)
        self.meal_enc = OneHotEncoder(sparse=False)
        self.mark_enc = OneHotEncoder(sparse=False)

    def load_data(self):
        self.df = pd.read_csv('Dataset_B_hotel.csv')

    def handle_missing_data(self):
        self.df['avg_price_per_room'].fillna(103.51, inplace=True)
        self.df['required_car_parking_space'].fillna(0, inplace=True)
        self.df['type_of_meal_plan'].fillna('Meal Plan 1', inplace=True)

    def encode_categorical_data(self):
        room_enc = self.room_enc.fit_transform(self.df[['room_type_reserved']])
        meal_enc = self.meal_enc.fit_transform(self.df[['type_of_meal_plan']])
        mark_enc = self.mark_enc.fit_transform(self.df[['market_segment_type']])

        room_df = pd.DataFrame(room_enc, columns=self.room_enc.get_feature_names_out())
        meal_df = pd.DataFrame(meal_enc, columns=self.meal_enc.get_feature_names_out())
        mark_df = pd.DataFrame(mark_enc, columns=self.mark_enc.get_feature_names_out())

        self.df = pd.concat([self.df, room_df, meal_df, mark_df], axis=1)
        self.df.drop(['room_type_reserved', 'type_of_meal_plan', 'market_segment_type'], axis=1, inplace=True)

        pkl.dump(self.room_enc, open('oneHot_encode_room.pkl', 'wb'))
        pkl.dump(self.meal_enc, open('oneHot_encode_meal.pkl', 'wb'))
        pkl.dump(self.mark_enc, open('oneHot_encode_mark.pkl', 'wb'))

    def split_data(self):
        X = self.df.drop(['booking_status', 'Booking_ID'], axis=1)
        y = self.df['booking_status']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def handle_outliers_and_scale(self, x_train, x_test):
        exclude_cols = ['arrival_month', 'booking_status', 'arrival_year', 'repeated_guest']
        num_cols = [col for col in x_train.columns if col not in exclude_cols and np.issubdtype(x_train[col].dtype, np.number)]

        for col in num_cols:
            Q1 = x_train[col].quantile(0.25)
            Q3 = x_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = (x_train[col] < lower) | (x_train[col] > upper)

            if outliers.any():
                scaler = RobustScaler()
            elif abs(x_train[col].skew()) > 0.5:
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()

            x_train[col] = scaler.fit_transform(x_train[[col]])
            x_test[col] = scaler.transform(x_test[[col]])

        return x_train, x_test


class ModelTrainer:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None

    def train_random_forest(self, x_train, y_train):
        self.rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_split=2)
        self.rf_model.fit(x_train, y_train)

    def train_xgboost(self, x_train, y_train):
        self.xgb_model = xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=10,)
        self.xgb_model.fit(x_train, y_train)

    def evaluate_model(self, model, x_test, y_test, name):
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=['0', '1'])
        acc = model.score(x_test, y_test)
        print(f"\nClassification Report ({name})\n{report}")
        print(f"Accuracy ({name}): {acc:.2f}")


class ModelSaver:
    @staticmethod
    def save_model(model, filename):
        pkl.dump(model, open(filename, 'wb'))
        print(f"Model saved as {filename}")


class BookingPredictionSystem:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()

    def run(self):
        self.data_processor.load_data()
        self.data_processor.handle_missing_data()
        self.data_processor.encode_categorical_data()

        x_train, x_test, y_train, y_test = self.data_processor.split_data()

        # Encode target
        y_train = y_train.replace({"Canceled": 1, "Not_Canceled": 0})
        y_test = y_test.replace({"Canceled": 1, "Not_Canceled": 0})
        pkl.dump({"Canceled": 1, "Not_Canceled": 0}, open('booking_status_encode.pkl', 'wb'))

        x_train, x_test = self.data_processor.handle_outliers_and_scale(x_train, x_test)

        # Train models with the same data
        self.model_trainer.train_random_forest(x_train, y_train)
        self.model_trainer.train_xgboost(x_train, y_train)

        # Evaluate models
        self.model_trainer.evaluate_model(self.model_trainer.rf_model, x_test, y_test, "Random Forest")
        self.model_trainer.evaluate_model(self.model_trainer.xgb_model, x_test, y_test, "XGBoost")

        ModelSaver.save_model(self.model_trainer.xgb_model, 'XG_booking_status.pkl')


# Eksekusi
if __name__ == "__main__":
    system = BookingPredictionSystem()
    system.run()



# In[ ]:




