import pandas as pd
import re

# Read the data
data = pd.read_csv("data.csv")


##Clean data and extract only the date in a string


def extract_time(text):
    time_pattern = r"\d{2}:\d{2}:\d{2} [APap][Mm]"
    match = re.search(time_pattern, text)

    if match:
        return match.group()
    else:
        return None


def extract_date(text):
    date_pattern = r"\d{1,2}/\d{1,2}/\d{4}"  # MM/DD/YYYY format
    match = re.search(date_pattern, text)

    if match:
        return match.group()
    else:
        return None


data["Time"] = data["Text"].apply(extract_time)
data["Date"] = data["Text"].apply(extract_date)
