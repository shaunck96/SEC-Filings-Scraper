from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time
import json

# Set Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-notifications")
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

# Initialize the WebDriver with Chrome options
driver = webdriver.Chrome(options=chrome_options)
urls = ["https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&datea=&dateb=&company=&type=&SIC=&State=&Country=&CIK=&owner=include&accno=&start={}&count=40"]

data = []
for index in range(len(urls)):
    for pagenum in range(40,120,40):            
        driver.get(urls[index].format(pagenum))
        time.sleep(10)

        SCROLL_PAUSE_TIME = 5
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_limit = 10
        count = 0
        while True and count < scroll_limit:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            count += 1

        src = driver.page_source
        soup = BeautifulSoup(src, 'html.parser')  

        # Find all rows in the table, skipping the header row
        rows = soup.find_all('tr')[1:]

        # Iterate over the rows to extract data
        for row in rows:
            # Each 'td' in the row contains a piece of information
            tds = row.find_all('td')

            # Only process rows with the expected number of columns
            if len(tds) == 6:
                form_type = tds[0].get_text(strip=True)
                links = "https://www.sec.gov"+''.join([a['href'] for a in tds[1].find_all('a')][1])
                description = tds[2].get_text(strip=True)
                accepted_date = tds[3].get_text(strip=True)
                filing_date = tds[4].get_text(strip=True)
                file_film_number = tds[5].get_text(strip=True)
                
                driver.get(links)
                SCROLL_PAUSE_TIME = 2
                last_height = driver.execute_script("return document.body.scrollHeight")
                scroll_limit = 5
                count = 0
                while True and count < scroll_limit:
                    if driver.window_handles:  # Check if there are open windows
                        driver.switch_to.window(driver.window_handles[0])  # Switch to the first open window
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(SCROLL_PAUSE_TIME)
                        new_height = driver.execute_script("return document.body.scrollHeight")
                        if new_height == last_height:
                            break
                        last_height = new_height
                        count += 1

            src = driver.page_source
            soup = BeautifulSoup(src, 'html.parser') 
            # Find the <pre> tag that contains the text
            text_content = soup.find('pre').get_text()      

            # Append the extracted data to the list
            data.append({
                'Form Type': form_type,
                'Links': links,
                'Description': description,
                'Accepted Date': accepted_date,
                'Filing Date': filing_date,
                'File/Film Number': file_film_number,
                'Filing Content': text_content
            })

# Print the extracted data
for item in data:
    print(item)

with open('sec_filings.json', 'w') as file:
    file.write(json.dumps(data))
