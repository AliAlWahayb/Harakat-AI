import csv
import json
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

CSV_PATH = 'siwar.csv'
PROCESSED_QUERIES_FILE = 'processed_queries.txt'  # Track processed queries here
OUTPUT_ALL = 'all_data.jsonl'
OUTPUT_DEFS = 'definitions.csv'
FAILED_QUERIES = 'failed_queries.csv'
PROXIES_FILE = 'proxies_list.txt'

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

write_lock = Lock()
processed_lock = Lock()


class ProxyRotator:
    def __init__(self, proxy_list):
        self.proxies = proxy_list.copy()
        self.index = 0
        self.count = len(proxy_list)
        self.fail_counts = {p: 0 for p in proxy_list}
        self.max_failures = 3

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


def load_processed_queries(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()


def save_processed_query(filename, query):
    with processed_lock:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(query + '\n')


def validate_proxy(proxy, timeout=5):
    test_url = "https://httpbin.org/ip"
    proxies = {"http": proxy, "https": proxy}
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    try:
        resp = requests.get(test_url, headers=headers, proxies=proxies, timeout=timeout, verify=False)
        if resp.status_code == 200:
            print(f"Proxy validated: {proxy}")
            return proxy
        else:
            print(f"Proxy failed status {resp.status_code}: {proxy}")
            return None
    except Exception as e:
        print(f"Proxy validation failed for {proxy}: {e}")
        return None


def validate_proxies_concurrently(proxies, max_workers=20):
    valid_proxies = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(validate_proxy, proxy): proxy for proxy in proxies}
        for future in futures:
            result = future.result()
            if result:
                valid_proxies.append(result)
    return valid_proxies


def fetch_data(session, query, proxy=None, user_agent=None, verify_ssl=False):
    url = API_URL_TEMPLATE.format(query=requests.utils.quote(query))
    headers = {
        'User-Agent': user_agent if user_agent else USER_AGENTS[0],
        'Accept': 'application/json',
        'Accept-Language': 'ar,en-US;q=0.9,en;q=0.8',
    }
    proxies = {"http": proxy, "https": proxy} if proxy else None
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


def process_query(session, query, proxy_rotator):
    max_attempts = 3
    for attempt in range(max_attempts):
        proxy = proxy_rotator.get_next_proxy()
        if not proxy:
            print("No proxies left to try in this thread.")
            break
        user_agent = random.choice(USER_AGENTS)
        data = fetch_data(session, query, proxy=proxy, user_agent=user_agent)
        if data:
            return data
        else:
            proxy_rotator.report_failure(proxy)
            time.sleep(random.uniform(2, 6))
    return None


def worker(queries, proxies, output_all_path, output_defs_path, failed_queries_path):
    session = requests.Session()
    proxy_rotator = ProxyRotator(proxies)

    with open(output_all_path, 'a', encoding='utf-8') as all_file, \
            open(output_defs_path, 'a', encoding='utf-8', newline='') as def_file, \
            open(failed_queries_path, 'a', encoding='utf-8', newline='') as fail_file:

        def_writer = csv.writer(def_file)
        fail_writer = csv.writer(fail_file)

        def_file.seek(0, 2)
        fail_file.seek(0, 2)
        if def_file.tell() == 0:
            def_writer.writerow(['query', 'definition'])
        if fail_file.tell() == 0:
            fail_writer.writerow(['query'])

        for query in queries:
            print(f"Thread {proxies[0]} processing query: {query[:30]}...")
            data = process_query(session, query, proxy_rotator)

            with write_lock:
                if data:
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

                    # Mark query as processed
                    save_processed_query(PROCESSED_QUERIES_FILE, query)

                else:
                    print(f"Failed all attempts for query '{query}'")
                    fail_writer.writerow([query])
                    fail_file.flush()

            time.sleep(random.uniform(2, 10))


def load_proxies(file_path):
    proxies = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            proxy = line.strip()
            if proxy and not proxy.startswith('#'):
                if not proxy.startswith("http://") and not proxy.startswith("https://"):
                    proxy = "http://" + proxy
                proxies.append(proxy)
    return proxies


def main():
    print("Loading proxies from file...")
    raw_proxies = load_proxies(PROXIES_FILE)
    print(f"Loaded {len(raw_proxies)} proxies from {PROXIES_FILE}.")

    print("Validating proxies concurrently before starting scraping...")
    valid_proxies = validate_proxies_concurrently(raw_proxies)
    print(f"Proxy validation complete. {len(valid_proxies)} proxies are valid out of {len(raw_proxies)}.")

    if not valid_proxies:
        print("No valid proxies found. Exiting.")
        return

    # Load processed queries set
    processed_queries = load_processed_queries(PROCESSED_QUERIES_FILE)

    # Load all queries from CSV, skipping processed ones
    with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        queries = [row['query'] for row in reader if row['query'] not in processed_queries]

    total_queries = len(queries)
    print(f"Total queries to process (excluding processed): {total_queries}")

    # Split proxies into 2 roughly equal parts
    half_proxies = len(valid_proxies) // 2
    proxies_thread_1 = valid_proxies[:half_proxies]
    proxies_thread_2 = valid_proxies[half_proxies:]

    print(f"Thread 1 proxies ({len(proxies_thread_1)}): {proxies_thread_1}")
    print(f"Thread 2 proxies ({len(proxies_thread_2)}): {proxies_thread_2}")

    # Split queries into 2 disjoint parts
    half_queries = total_queries // 2
    queries_thread_1 = queries[:half_queries]
    queries_thread_2 = queries[half_queries:]

    print(f"Thread 1 queries: {len(queries_thread_1)}")
    print(f"Thread 2 queries: {len(queries_thread_2)}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(worker, queries_thread_1, proxies_thread_1, OUTPUT_ALL, OUTPUT_DEFS, FAILED_QUERIES)
        future2 = executor.submit(worker, queries_thread_2, proxies_thread_2, OUTPUT_ALL, OUTPUT_DEFS, FAILED_QUERIES)

        future1.result()
        future2.result()

    print("Scraping completed!")


if __name__ == "__main__":
    main()
