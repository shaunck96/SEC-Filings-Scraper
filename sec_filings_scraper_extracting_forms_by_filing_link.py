from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time
import json
import os
import hashlib

with open('sec_filings.json', 'r') as f:
    sec_filings_info = json.load(f)

index = 6

url = sec_filings_info[index]['Links']
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
driver.get(url)

time.sleep(5)

SCROLL_PAUSE_TIME = 2
last_height = driver.execute_script("return document.body.scrollHeight")
scroll_limit = 5
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

links = []
# Iterate over the rows to extract data
for row in rows:
    # Each 'td' in the row contains a piece of information
    tds = row.find_all('td')
    
    # Iterate over each 'td' element to find 'a' elements
    for td in tds:
        a_tags = td.find_all('a')
        for a in a_tags:
            href = a.get('href')
            links.append("https://www.sec.gov"+href)
    
print(links)

#driver.get(links[1])
driver.get(links[0])

time.sleep(5)

SCROLL_PAUSE_TIME = 2
scroll_limit = 5
screenshot_folder = "screenshots"+sec_filings_info[0]['File/Film Number']  # Create a folder to save screenshots
sec_filings_info[index]['folder_path'] = screenshot_folder
os.makedirs(screenshot_folder, exist_ok=True)

count = 0
while count < scroll_limit:
    # Save a screenshot of the current view
    screenshot_path = os.path.join(screenshot_folder, f"screenshot_{count}"+sec_filings_info[0]['File/Film Number']+".png")
    driver.save_screenshot(screenshot_path)
    
    # Scroll down
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    # Wait for an element near the bottom of the page to appear
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//footer[@id='footer']")))
        break  # Exit the loop if the footer element is found
    except:
        pass
    
    time.sleep(SCROLL_PAUSE_TIME)
    
    count += 1

driver.quit()

hashes = set()

for filename in os.listdir(screenshot_folder):
    path = os.path.join(screenshot_folder, filename)
    digest = hashlib.sha1(open(path,'rb').read()).digest()
    if digest not in hashes:
        hashes.add(digest)
    else:
        os.remove(path)
