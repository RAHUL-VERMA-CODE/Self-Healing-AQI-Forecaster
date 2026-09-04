"""
Fetch real-time AQI data from data.gov.in/CPCB
and save city-level data to new_data/incoming.csv
"""

import os
import sys
import json
import socket
import requests
import pandas as pd
from datetime import date
from dotenv import load_dotenv

import urllib3.util.connection as urllib3_cn

from src.exception.exception import customException
from src.logging.logger import logging


load_dotenv()


# Force IPv4
def allowed_gai_family():
    return socket.AF_INET


urllib3_cn.allowed_gai_family = allowed_gai_family


API_KEY = os.getenv("DATA_GOV_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}

RATE_LIMIT_FILE = os.path.join("final", "api_usage_tracker.json")
MAX_DAILY_REQUESTS = 500

CITIES = ["Delhi", "Faridabad", "Ghaziabad", "Gurugram", "Noida"]

POLLUTANT_MAP = {
    "CO": "co",
    "NO2": "no2",
    "OZONE": "o3",
    "O3": "o3",
    "PM10": "pm10",
    "PM2.5": "pm25",
    "PM 2.5": "pm25",
    "SO2": "so2"
}

AVG_FIELD = "avg_value"

# API CO = ug/m3, training data CO = mg/m3
CO_UNIT_DIVISOR = 1000

AQI_BREAKPOINTS = {
    "pm25": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
              (91, 120, 201, 300), (121, 250, 301, 400), (251, float("inf"), 401, 500)],
    "pm10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
              (251, 350, 201, 300), (351, 430, 301, 400), (431, float("inf"), 401, 500)],
    "no2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
             (181, 280, 201, 300), (281, 400, 301, 400), (401, float("inf"), 401, 500)],
    "so2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
             (381, 800, 201, 300), (801, 1600, 301, 400), (1601, float("inf"), 401, 500)],
    "o3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
            (169, 208, 201, 300), (209, 748, 301, 400), (749, float("inf"), 401, 500)],
    "co": [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10.0, 101, 200),
            (10.1, 17.0, 201, 300), (17.1, 34.0, 301, 400), (34.1, float("inf"), 401, 500)]
}


def send_alert(message: str):
    try:
        logging.critical(message)
        if SLACK_WEBHOOK_URL:
            requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=10)
    except Exception as e:
        logging.error(f"Failed to send alert: {e}")


def check_and_increment_rate_limit(requests_needed: int):
    try:
        os.makedirs(os.path.dirname(RATE_LIMIT_FILE), exist_ok=True)
        today = str(date.today())

        tracker = {}
        if os.path.exists(RATE_LIMIT_FILE):
            with open(RATE_LIMIT_FILE, "r") as f:
                tracker = json.load(f)

        used_today = tracker.get(today, 0)

        if used_today + requests_needed > MAX_DAILY_REQUESTS:
            message = f"Daily API quota would be exceeded: {used_today}/{MAX_DAILY_REQUESTS}"
            send_alert(message)
            raise customException(message, sys)

        tracker[today] = used_today + requests_needed
        with open(RATE_LIMIT_FILE, "w") as f:
            json.dump(tracker, f, indent=4)

        logging.info(f"API usage today: {tracker[today]}/{MAX_DAILY_REQUESTS}")

    except customException:
        raise
    except Exception as e:
        raise customException(e, sys)


def calculate_sub_index(concentration, breakpoints):
    try:
        if pd.isna(concentration):
            return None

        concentration = float(concentration)
        if concentration < 0:
            return None

        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= concentration <= c_high:
                if c_high == float("inf"):
                    return float(i_high)
                return ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low

        return None

    except Exception as e:
        raise customException(e, sys)


def calculate_cpcb_aqi(row):
    try:
        sub_indices = []

        for pollutant, breakpoints in AQI_BREAKPOINTS.items():
            value = row.get(pollutant, float("nan"))
            if pd.isna(value):
                continue

            sub_index = calculate_sub_index(value, breakpoints)
            if sub_index is not None:
                sub_indices.append(sub_index)

        available_pm = (
            not pd.isna(row.get("pm25", float("nan")))
            or not pd.isna(row.get("pm10", float("nan")))
        )

        if len(sub_indices) < 3 or not available_pm:
            return None

        return round(max(sub_indices), 2)

    except Exception as e:
        raise customException(e, sys)


def fetch_city_data(city, max_retries=3):
    try:
        if not API_KEY:
            raise customException("DATA_GOV_API_KEY missing - check .env file", sys)

        params = {
            "api-key": API_KEY,
            "format": "json",
            "filters[city]": city,
            "limit": 100
        }

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(BASE_URL, params=params, headers=REQUEST_HEADERS, timeout=60)
                response.raise_for_status()
                records = response.json().get("records", [])
                logging.info(f"Fetched {len(records)} records for {city}")
                return pd.DataFrame(records)

            except requests.exceptions.RequestException as e:
                last_error = e
                logging.warning(f"Attempt {attempt}/{max_retries} failed for {city}: {e}")

        send_alert(f"Failed to fetch {city} after {max_retries} attempts: {last_error}")
        return pd.DataFrame()

    except customException:
        raise
    except Exception as e:
        raise customException(e, sys)


def build_city_dataframe(df, city):
    try:
        if df.empty:
            return pd.DataFrame()

        required_columns = ["last_update", "pollutant_id", AVG_FIELD]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise customException(f"{city}: missing API columns {missing_columns}", sys)

        df[AVG_FIELD] = pd.to_numeric(df[AVG_FIELD], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["last_update"], dayfirst=True, errors="coerce")

        # .copy() avoids SettingWithCopyWarning on the columns assigned below —
        # dropna() can return a view into the original dataframe
        df = df.dropna(subset=["timestamp", "pollutant_id", AVG_FIELD]).copy()
        if df.empty:
            return pd.DataFrame()

        df["pollutant_id"] = df["pollutant_id"].astype(str).str.strip().str.upper()
        df["pollutant"] = df["pollutant_id"].map(POLLUTANT_MAP)
        df = df.dropna(subset=["pollutant"])
        if df.empty:
            return pd.DataFrame()

        # station intentionally not used — all stations of one city are averaged
        pivot = df.pivot_table(
            index="timestamp", columns="pollutant", values=AVG_FIELD, aggfunc="mean"
        ).reset_index()
        pivot.columns.name = None

        pollutant_columns = ["co", "no2", "o3", "pm10", "pm25", "so2"]
        for col in pollutant_columns:
            if col not in pivot.columns:
                pivot[col] = float("nan")
            pivot[col] = pd.to_numeric(pivot[col], errors="coerce")

        # convert CO ug/m3 -> mg/m3 to match training data units
        pivot["co"] = pivot["co"] / CO_UNIT_DIVISOR

        pivot["location_name"] = "Gurgaon" if city == "Gurugram" else city

        # AQI calculated after city-level aggregation.
        # NOTE on NaNs: individual pollutant columns (co, no2, etc.) are left
        # as-is if missing — data_transformation.py's SimpleImputer(median)
        # handles those at training time, so no need to duplicate that logic
        # here. Only "aqi" itself is checked below, since a missing target/
        # feature value can't be meaningfully imputed.
        pivot["aqi"] = pivot.apply(calculate_cpcb_aqi, axis=1)

        # derive the same time features training expects —
        # location_lat/lon dropped, they were never used as model features
        pivot["hour"] = pivot["timestamp"].dt.hour
        pivot["month"] = pivot["timestamp"].dt.month
        pivot["day_of_week"] = pivot["timestamp"].dt.dayofweek
        pivot["is_weekend"] = (pivot["day_of_week"] >= 5).astype(int)

        final_columns = [
            "timestamp", "location_name", "co", "no2", "o3", "pm10", "pm25", "so2",
            "aqi", "hour", "month", "day_of_week", "is_weekend"
        ]
        pivot = pivot[final_columns]
        pivot = pivot.sort_values(by="timestamp").reset_index(drop=True)

        logging.info(f"{city}: {len(pivot)} city-level rows created")
        return pivot

    except customException:
        raise
    except Exception as e:
        raise customException(e, sys)


def build_incoming_dataframe():
    try:
        check_and_increment_rate_limit(requests_needed=len(CITIES))

        all_city_data = []
        for city in CITIES:
            raw_df = fetch_city_data(city)
            if raw_df.empty:
                logging.warning(f"No data found for {city}")
                continue

            city_df = build_city_dataframe(raw_df, city)
            if city_df.empty:
                logging.warning(f"No usable data for {city}")
                continue

            all_city_data.append(city_df)

        if not all_city_data:
            message = "No city data fetched. Every city failed."
            send_alert(message)
            raise customException(message, sys)

        combined = pd.concat(all_city_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp", "location_name"], keep="last")
        combined = combined.sort_values(by=["timestamp", "location_name"]).reset_index(drop=True)

        expected_columns = [
            "timestamp", "location_name", "co", "no2", "o3", "pm10", "pm25", "so2",
            "aqi", "hour", "month", "day_of_week", "is_weekend"
        ]

        if list(combined.columns) != expected_columns:
            raise customException(
                f"Final schema mismatch.\nExpected: {expected_columns}\nGot: {list(combined.columns)}",
                sys
            )

        # rows without enough pollutant info can't produce a valid AQI —
        # these are dropped rather than imputed, since AQI is both the
        # feature and derived from the target's basis (current-hour aqi)
        combined = combined.dropna(subset=["aqi"]).reset_index(drop=True)

        logging.info(f"Final rows: {len(combined)}")
        return combined

    except customException:
        raise
    except Exception as e:
        raise customException(e, sys)


if __name__ == "__main__":
    try:
        os.makedirs("new_data", exist_ok=True)
        incoming_df = build_incoming_dataframe()

        output_path = os.path.join("new_data", "incoming.csv")

        # append instead of overwrite, so runs accumulate data
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            existing_df = pd.read_csv(output_path)
            incoming_df = pd.concat([existing_df, incoming_df], ignore_index=True)

        incoming_df = incoming_df.drop_duplicates(subset=["timestamp", "location_name"], keep="last")
        incoming_df = incoming_df.sort_values(by=["timestamp", "location_name"]).reset_index(drop=True)
        incoming_df.to_csv(output_path, index=False)

        logging.info(f"Rows saved: {len(incoming_df)} | File: {output_path}")

    except Exception as e:
        send_alert(f"fetch_new_data.py failed: {e}")
        raise customException(e, sys)