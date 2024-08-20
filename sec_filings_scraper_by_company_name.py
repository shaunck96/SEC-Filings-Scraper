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

# Navigate to the SEC Edgar website
driver.get('https://www.sec.gov/edgar/searchedgar/companysearch.html')

# Find the input field for the company name by its name attribute
company_name_input = driver.find_element(By.ID, 'edgar-company-person')


# Enter "Alphabet" into the input field
company_name_input.send_keys('0000320193')

time.sleep(5)

# Find and click the "Search" button
search_button = driver.find_element(By.CLASS_NAME, 'collapsed-submit')
search_button.click()

# Wait for some time to allow the search results to load
# You may need to adjust the waiting time as per your needs
time.sleep(5)  # Wait for 10 seconds

time.sleep(5)
# Find the <span> element with class "alt-title" and containing the text "Open document"
element = driver.find_element(By.XPATH, '//span[@class="alt-title" and contains(text(), "Open document")]')

# Click the element
element.click()

# Close the browser when done
driver.quit()

