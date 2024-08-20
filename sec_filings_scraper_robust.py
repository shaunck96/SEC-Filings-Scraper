from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import time
import json
from mongodb_connection import MongoDBHandler

def init_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def safe_get(driver, url, retries=3, delay=5):
    while retries > 0:
        try:
            driver.get(url)
            return  
        except WebDriverException:
            time.sleep(delay)  
            retries -= 1
    raise WebDriverException(f"Failed to load {url} after multiple attempts")

def scroll_to_bottom(driver, pause_time=2, scroll_limit=10):
    last_height = driver.execute_script("return document.body.scrollHeight")
    count = 0
    while count < scroll_limit:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        count += 1

def scrape_current_page(driver):
    src = driver.page_source
    soup = BeautifulSoup(src, 'html.parser')
    data = []
    rows = soup.find_all('tr')[1:]
    for row in rows:
        try:
            tds = row.find_all('td')
            if len(tds) == 6:

                issuer_links = tds[2].find_all('a')
                issuer = ''
                for link in issuer_links:
                    if 'Issuer' in link.text:
                        issuer = link.text.split('(')[0].strip()
                data.append({
                    'Form Type': tds[0].get_text(strip=True),
                    'Links': "https://www.sec.gov" + tds[1].find('a', href=True)['href'],
                    'Description': tds[2].get_text(strip=True),
                    'Accepted Date': tds[3].get_text(strip=True),
                    'Filing Date': tds[4].get_text(strip=True),
                    'File/Film Number': tds[5].get_text(strip=True)#,
                    #'Filing Content': '',
                    #'issuer': issuer
                })
        except (IndexError, TypeError, KeyError, AttributeError) as e:
            print(f"Error processing row: {e}")
    return data


def main():
    driver = init_driver()
    base_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&datea=&dateb=&company=&type=&SIC=&State=&Country=&CIK=&owner=include&accno=&start={}&count=40"
    all_data = []

    try:
        for pagenum in range(0, 40, 40):
            safe_get(driver, base_url.format(pagenum))
            scroll_to_bottom(driver)
            data = scrape_current_page(driver)
            for entry in data:
                try:
                    safe_get(driver, entry['Links'])
                    scroll_to_bottom(driver, pause_time=2, scroll_limit=5)
                    src = driver.page_source
                    soup = BeautifulSoup(src, 'html.parser')
                    entry['Filing Content'] = soup.find('pre').get_text() if soup.find('pre') else 'Content not found'
                except Exception as e:
                    print(f"Error accessing filing details: {e}")
            all_data.extend(data)
    finally:
        driver.quit()

    with open("mongodb_credentials.json","r"):
        uri = json.loads("mongodb_credentials.json")

    handler = MongoDBHandler(uri)

    handler.insert_records("finance_database", 
                           "sec_filings", 
                           all_data, 
                           avoid_duplicates=True)


if __name__ == "__main__":
    main()
    