import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

START_YEAR = 2010
END_YEAR = 2026

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

def extract_from_historical_page(year):
    url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
    r = requests.get(url)
    if r.status_code != 200:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text()

    meetings = []

    # Handles:
    # - January 26-27
    # - July 31-August 1
    pattern = r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d+)(?:-(?:([A-Za-z]+)\s+)?(\d+))?"

    matches = re.findall(pattern, text)

    for match in matches:
        start_month, day1, end_month, day2 = match

        if day2:
            # If cross-month
            if end_month:
                month = MONTHS[end_month]
            else:
                month = MONTHS[start_month]
            day = int(day2)
        else:
            month = MONTHS[start_month]
            day = int(day1)

        meetings.append((year, month, day))

    return meetings


def extract_from_modern_page():
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text()

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    meetings = []
    current_year = None
    current_month = None

    month_pattern = re.compile(
        r"^(January|February|March|April|May|June|July|August|September|October|November|December)$"
    )

    year_pattern = re.compile(r"^(20\d{2}) FOMC Meetings")

    day_pattern = re.compile(r"^(\d+)(?:-(\d+))?")

    for line in lines:

        # Detect year header
        year_match = year_pattern.match(line)
        if year_match:
            current_year = int(year_match.group(1))
            continue

        # Detect month
        month_match = month_pattern.match(line)
        if month_match:
            current_month = MONTHS[month_match.group(1)]
            continue

        # Detect day or day range
        day_match = day_pattern.match(line)
        if day_match and current_year and current_month:
            day1, day2 = day_match.groups()
            day = int(day2) if day2 else int(day1)
            meetings.append((current_year, current_month, day))

    return meetings


all_data = []

for y in range(START_YEAR, min(2021, END_YEAR + 1)):
    print(f"Processing historical {y}")
    all_data.extend(extract_from_historical_page(y))

if END_YEAR >= 2021:
    print("Processing modern consolidated page")
    all_data.extend(extract_from_modern_page())

df = pd.DataFrame(all_data, columns=["Year", "Month", "Day"])
df.drop_duplicates(inplace=True)
df.sort_values(["Year", "Month", "Day"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Save directory
wrkdir = os.getcwd()
savedir = os.path.join(wrkdir, "marketinputs")
os.makedirs(savedir, exist_ok=True)

csvfile = os.path.join(savedir, "fed_meeting_calendar.csv")
df.to_csv(csvfile, index=False)

print("Done.")
print(f"Saved to: {csvfile}")