from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Set up the WebDriver (make sure the path is correct for your system)
driver = webdriver.Chrome()  # or webdriver.Firefox() for Firefox

# Open the login URL
url = "https://x.com/login"  # Adjust to the login page if needed
driver.get(url)

# Wait for the page to load
time.sleep(5)

# Input username
username_field = driver.find_element(By.NAME, "text")  # Using the name attribute for username
username_field.send_keys("mukeshadar")  # Your username
time.sleep(2)  # Wait for a moment

# Click the "Next" button
next_button = driver.find_element(By.XPATH, "//span[text()='Next']")  # Using XPath to find the button
next_button.click()

# Wait for the password field to load
time.sleep(5)

# Input password using full XPath
password_field = driver.find_element(By.XPATH, "/html/body/div/div/div/div[1]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/label/div/div[2]/div[1]/input")
password_field.send_keys("mukeshadarone")  # Your password
time.sleep(2)

# Click the "Log in" button using full XPath
login_button = driver.find_element(By.XPATH, "/html/body/div/div/div/div[1]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div[1]/div/div/button/div")
login_button.click()

# Wait for the home page to load
time.sleep(5)

# Now navigate to the search URL after logging in
search_url = "https://x.com/search?q=SPRO%20biotech&src=typed_query&f=live"
driver.get(search_url)

# Wait for the page to load
time.sleep(5)

# Scroll down a few times to load more tweets
for _ in range(5):  # Adjust the range for more scrolling
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
    time.sleep(2)  # Wait for new content to load

# Collect tweets
tweets = driver.find_elements(By.CSS_SELECTOR, 'div.tweet')  # Adjust selector as needed
tweet_texts = [tweet.text for tweet in tweets]

# Close the browser
driver.quit()

# Print or save the tweet texts
for text in tweet_texts:
    print(text)

# from playwright.sync_api import sync_playwright
# import pandas as pd

# def intercept_response(response):
#     """Capture all background requests and save those containing tweet data."""
#     try:
#         if "TweetResultByRestId" in response.url:
#             return response.json()
#     except Exception as e:
#         print(f"Error in intercept_response: {e}")
#     return {}

# def scrape_tweet(url: str) -> dict:
#     """Scrape a single tweet page for tweet data."""
#     tweet_data = {}

#     with sync_playwright() as pw:
#         browser = pw.chromium.launch(headless=True)
#         context = browser.new_context(viewport={"width": 1920, "height": 1080})
#         page = context.new_page()

#         page.on("response", lambda response: tweet_data.update(intercept_response(response)))
#         page.goto(url, timeout=60000)
#         page.wait_for_selector("[data-testid='tweet']")

#         tweet_calls = [xhr for xhr in tweet_data if "TweetResultByRestId" in xhr.url]
#         for xhr in tweet_calls:
#             data = xhr.json()
#             return data['data']['tweetResult']['result']

# def parse_tweet(data: dict) -> dict:
#     """Parse X.com tweet JSON dataset for the most important fields."""
#     result = {
#         "created_at": data.get("legacy", {}).get("created_at"),
#         "attached_urls": [url["expanded_url"] for url in data.get("legacy", {}).get("entities", {}).get("urls", [])],
#         "attached_media": [media["media_url_https"] for media in data.get("legacy", {}).get("entities", {}).get("media", [])],
#         "tagged_users": [mention["screen_name"] for mention in data.get("legacy", {}).get("entities", {}).get("user_mentions", [])],
#         "tagged_hashtags": [hashtag["text"] for hashtag in data.get("legacy", {}).get("entities", {}).get("hashtags", [])],
#         "favorite_count": data.get("legacy", {}).get("favorite_count"),
#         "retweet_count": data.get("legacy", {}).get("retweet_count"),
#         "reply_count": data.get("legacy", {}).get("reply_count"),
#         "text": data.get("legacy", {}).get("full_text"),
#         "user_id": data.get("legacy", {}).get("user_id_str"),
#         "tweet_id": data.get("legacy", {}).get("id_str"),
#         "conversation_id": data.get("legacy", {}).get("conversation_id_str"),
#         "language": data.get("legacy", {}).get("lang"),
#         "source": data.get("source"),
#         "views": data.get("views", {}).get("count")
#     }
#     return result

# def save_to_csv(tweet_data: dict, filename: str):
#     """Save the parsed tweet data to a CSV file."""
#     df = pd.DataFrame([tweet_data])
#     df.to_csv(filename, index=False)

# if __name__ == "__main__":
#     tweet_url = "https://x.com/BillGates/status/1352662770416664577"
#     tweet_data = scrape_tweet(tweet_url)
#     parsed_data = parse_tweet(tweet_data)
#     save_to_csv(parsed_data, "tweet_data.csv")
#     print(f"Tweet data saved to tweet_data.csv")