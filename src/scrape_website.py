import re
import time

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options  # for headless mode; optional

ID = 51500
SID = 46834
TITLE = "289KW_PV_System_Hourly"
MAX_DAY = 164

year = 2019
month = 10
day = 25

# don't commit these by accident - sys variables are nice to use here
USERNAME = "Enter your Pvoutput.org username here"
PASSWORD = "Enter your Pvoutput.org password here"


def days_in_month():
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if month == 2 and year % 4 == 0:
        return 29
    return days[month - 1]


def subtract_day():
    global day, month, year

    day -= 1

    if day > 0:
        return

    month -= 1
    if month <= 0:
        year -= 1
        month = 12

    day = days_in_month()


def login(driver: webdriver):
    login_url = "https://pvoutput.org/login.jsp"
    driver.get(login_url)
    driver.implicitly_wait(100)

    username = driver.find_element_by_id("login")
    password = driver.find_element_by_id("password")
    username.send_keys(USERNAME)
    password.send_keys(PASSWORD)
    driver.find_elements_by_class_name("btn-primary")[0].click()


def get_row(driver: webdriver, d: int) -> list:
    url = f"https://pvoutput.org/intraday.jsp?id=51500&sid=46834&dt={year}{str(month).zfill(2)}{str(day).zfill(2)}&gs=0&m=0"
    subtract_day()

    driver.get(url)
    driver.implicitly_wait(100)
    html = driver.page_source

    soup = BeautifulSoup(html, "html.parser")
    return soup.find_all("tr", class_=["e", "o"])


def process_row(row: str) -> str:
    data = []

    for index in range(5):
        text = row.contents[index].text
        if 2 <= index <= 3:
            continue

        if text == "-":
            text = np.nan
        elif index == 4:
            text = text.replace(",", "")  # 1,230.5 -> 1230.5
            text = float(re.findall(r"\d*", text)[0])
        data.append(text)

    return data


if __name__ == "__main__":
    data = pd.DataFrame(
        columns=["Date (dd/mm/yy)", "Time (12 Hour)", "Generated (kWh)"]
    )

    options = Options()
    options.headless = True
    firefox_profile = webdriver.FirefoxProfile(
        r"C:/Users/Chris/AppData/Roaming/Mozilla/Firefox/Profiles/fh2hfyrj.default"
    )
    driver = webdriver.Firefox(firefox_profile=firefox_profile, options=options)
    login(driver)

    for d in range(MAX_DAY):
        rows = get_row(driver, d)

        for index in range(len(rows)):
            row = rows[index]
            row = process_row(row)
            data.loc[len(data)] = row

        print(f"Day {d} processed.")
        time.sleep(15)  # don't spam the website for data

    data.to_csv(f"../data/{TITLE}.csv", index=False)
