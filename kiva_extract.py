import pandas as pd
import requests
import time

# Kiva's WAF didn't exist when she wrote this; without a browser UA you get 403.
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}

NUM_PAGES = 4   # 500 per page -> 2000 loans

all_loans = []
for page in range(1, NUM_PAGES + 1):
    url = 'https://api.kivaws.org/v1/loans/search.json?country_code=UG&per_page=500&page=%s' % page
    response = requests.get(url, headers=HEADERS)
    loans = response.json()['loans']
    all_loans += loans
    print('page %s: %s loans (%s total)' % (page, len(loans), len(all_loans)))
    if not loans:
        break
    time.sleep(1)

df = pd.json_normalize(all_loans)
df.to_csv('kiva_uganda.csv', index=False)
print(df['location.country'].value_counts().head())