import requests
from bs4 import BeautifulSoup
import os
import json
import time
from urllib.parse import urljoin, urlparse

# Function to fetch and parse HTML content
def fetch_html(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

# Function to extract and clean text from HTML
def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    # Extract text
    text = soup.get_text(separator=' ')
    # Collapse multiple spaces
    text = ' '.join(text.split())
    return text

# Function to extract all relevant links from the documentation
def extract_links(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Resolve relative URLs
        full_url = urljoin(base_url, href)
        # Ensure the link is within the same domain
        if urlparse(full_url).netloc == urlparse(base_url).netloc:
            links.add(full_url)
    return links

# Function to save extracted data to a JSON file
def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Main function to process the website
def process_website(base_url, output_dir='cobalt_docs_data', delay=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    visited = set()
    to_visit = {base_url}
    all_data = []

    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue

        try:
            print(f'Fetching {current_url}')
            html = fetch_html(current_url)
            text = extract_text(html)
            all_data.append({'url': current_url, 'text': text})

            # Extract and queue new links
            links = extract_links(html, base_url)
            to_visit.update(links - visited)

            # Save individual page data
            page_filename = os.path.join(output_dir, f"{urlparse(current_url).path.strip('/').replace('/', '_')}.json")
            save_to_json({'url': current_url, 'text': text}, page_filename)

            # Respectful crawling: delay between requests
            time.sleep(delay)
        except Exception as e:
            print(f'Error fetching {current_url}: {e}')

        visited.add(current_url)

    # Save all data combined
    combined_filename = os.path.join(output_dir, 'all_data.json')
    save_to_json(all_data, combined_filename)

# URL of the website to process
base_url = 'https://docs.gocobalt.io/'

# Process the website and store the data
process_website(base_url)
