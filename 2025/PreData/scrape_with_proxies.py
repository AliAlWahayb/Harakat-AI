import csv
import json
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


CSV_PATH = 'siwar.csv'          # Your input queries file
OUTPUT_ALL = 'all_data.jsonl'   # Full JSON data output
OUTPUT_DEFS = 'definitions.csv' # Definitions output as CSV
FAILED_QUERIES = 'failed_queries.csv'  # Failed queries output
PROXIES_FILE = 'proxies_list.txt'      # Your proxies file

API_URL_TEMPLATE = "https://siwar.ksaa.gov.sa/api/search/alt/global-public/{query}?lexiconIds=2202f51d-7d70-4472-9fc0-f178fb425463"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:112.0) Gecko/20100101 Firefox/112.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:111.0) Gecko/20100101 Firefox/111.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13.4; rv:114.0) Gecko/20100101 Firefox/114.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_7_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.7 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 13; Pixel 6 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Android 13; Mobile; rv:114.0) Gecko/114.0 Firefox/114.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.67",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.57",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 OPR/99.0.4844.51",
]

class ProxyRotator:
    def __init__(self, proxy_list):
        self.proxies = proxy_list
        self.index = 0
        self.count = len(proxy_list)
        self.fail_counts = {p: 0 for p in proxy_list}
        self.max_failures = 3  # Remove proxy after 3 failures

    def get_next_proxy(self):
        if self.count == 0:
            return None
        proxy = self.proxies[self.index]
        self.index = (self.index + 1) % self.count
        return proxy

    def report_failure(self, proxy):
        if proxy not in self.fail_counts:
            return
        self.fail_counts[proxy] += 1
        if self.fail_counts[proxy] >= self.max_failures:
            print(f"Removing proxy due to repeated failures: {proxy}")
            self.remove_proxy(proxy)

    def remove_proxy(self, proxy):
        if proxy in self.proxies:
            self.proxies.remove(proxy)
            del self.fail_counts[proxy]
            self.count = len(self.proxies)
            if self.index >= self.count:
                self.index = 0

def validate_proxy(proxy, timeout=5):
    test_url = "https://httpbin.org/ip"
    proxies = {
        "http": proxy,
        "https": proxy,
    }
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
    }
    try:
        resp = requests.get(test_url, headers=headers, proxies=proxies, timeout=timeout, verify=False)
        if resp.status_code == 200:
            print(f"Proxy validated: {proxy}")
            return True
        else:
            print(f"Proxy failed status {resp.status_code}: {proxy}")
            return False
    except Exception as e:
        print(f"Proxy validation failed for {proxy}: {e}")
        return False

def fetch_data(session, query, proxy=None, user_agent=None, verify_ssl=False):
    url = API_URL_TEMPLATE.format(query=requests.utils.quote(query))
    headers = {
        'User-Agent': user_agent if user_agent else USER_AGENTS[0],
        'Accept': 'application/json',
        'Accept-Language': 'ar,en-US;q=0.9,en;q=0.8',
    }
    proxies = {
        "http": proxy,
        "https": proxy,
    } if proxy else None

    try:
        response = session.get(url, headers=headers, proxies=proxies, timeout=10, verify=verify_ssl)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Warning: Status {response.status_code} for query '{query}'")
            return None
    except Exception as e:
        print(f"Exception for query '{query}' with proxy '{proxy}': {e}")
        return None

def load_proxies(file_path):
    proxies = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            proxy = line.strip()
            if proxy and not proxy.startswith('#'):
                # Optional: ensure proxies start with http:// or https://
                if not proxy.startswith("http://") and not proxy.startswith("https://"):
                    proxy = "http://" + proxy
                proxies.append(proxy)
    return proxies

def main():
    print("Loading proxies from file...")
    raw_proxies = load_proxies(PROXIES_FILE)
    print(f"Loaded {len(raw_proxies)} proxies from {PROXIES_FILE}.")

    print("Validating proxies before starting scraping...")
    valid_proxies = []
    for p in raw_proxies:
        if validate_proxy(p):
            valid_proxies.append(p)
    print(f"Proxy validation complete. {len(valid_proxies)} proxies are valid out of {len(raw_proxies)}.")

    proxy_rotator = ProxyRotator(valid_proxies)
    session = requests.Session()

    with open(CSV_PATH, newline='', encoding='utf-8') as csvfile, \
         open(OUTPUT_ALL, 'a', encoding='utf-8') as all_file, \
         open(OUTPUT_DEFS, 'a', encoding='utf-8', newline='') as def_file, \
         open(FAILED_QUERIES, 'a', encoding='utf-8', newline='') as fail_file:

        reader = csv.DictReader(csvfile)
        total_queries = sum(1 for _ in open(CSV_PATH, encoding='utf-8')) - 1
        print(f"Total queries to process: {total_queries}")

        def_writer = csv.writer(def_file)
        fail_writer = csv.writer(fail_file)

        # Write headers if files are empty
        def_file.seek(0, 2)
        fail_file.seek(0, 2)
        if def_file.tell() == 0:
            def_writer.writerow(['query', 'definition'])
        if fail_file.tell() == 0:
            fail_writer.writerow(['query'])

        for idx, row in enumerate(reader, 1):
            query = row['query']
            user_agent = random.choice(USER_AGENTS)

            success = False
            max_attempts = min(10, proxy_rotator.count)  # try up to 10 proxies max per query

            for attempt in range(max_attempts):
                proxy = proxy_rotator.get_next_proxy()
                if not proxy:
                    print("No proxies left to try.")
                    break

                print(f"[{idx}/{total_queries}] Trying '{query}' with proxy {proxy} and UA {user_agent[:50]} (Attempt {attempt+1}/{max_attempts})")
                data = fetch_data(session, query, proxy=proxy, user_agent=user_agent, verify_ssl=False)

                if data:
                    success = True
                    all_file.write(json.dumps({'query': query, 'data': data}, ensure_ascii=False) + "\n")
                    all_file.flush()

                    definitions = []
                    for entry in data.get('entries', []):
                        for sense in entry.get('senses', []):
                            def_text = sense.get('definition', '').strip()
                            if def_text:
                                definitions.append(def_text)

                    if definitions:
                        for d in definitions:
                            def_writer.writerow([query, d])
                        def_file.flush()
                    else:
                        print(f"No definitions found for query '{query}'")
                        fail_writer.writerow([query])
                        fail_file.flush()
                    break
                else:
                    print(f"Failed attempt {attempt+1} for query '{query}' with proxy {proxy}")
                    proxy_rotator.report_failure(proxy)
                    time.sleep(random.uniform(2, 6))

            if not success:
                print(f"All attempts failed for query '{query}'")
                fail_writer.writerow([query])
                fail_file.flush()

            time.sleep(random.uniform(2, 12))

    print("Scraping completed!")

if __name__ == "__main__":
    main()
