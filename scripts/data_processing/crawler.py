import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
from collections import deque
import logging

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, WebDriverException

from policypilot.services.database import get_db_connection, add_discovered_source
from policypilot.config import settings
from policypilot.services.source_policy import is_trusted_source_url
from .crawler_utils import get_content_hash, sanitize_filename

logger = logging.getLogger(__name__)

def get_session():
    """Creates a requests session with a user agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
    return session

def download_and_save_content(session: requests.Session, url: str, dest_folder: Path, source_id: int):
    """Downloads a file (HTML, PDF), saves it using the new naming convention, and updates the database."""
    if not is_trusted_source_url(url):
        logger.warning("Rejected download outside the trusted government allowlist: %s", url)
        return None
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return

    content_type = response.headers.get('content-type', '').lower()
    file_ext = '.pdf' if 'pdf' in content_type else '.html'

    sanitized_name = sanitize_filename(url)
    final_filename = f"source_{source_id}_{sanitized_name}{file_ext}"
    save_path = dest_folder / final_filename

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        f.write(response.content)

    content_hash = get_content_hash(response.content)

    with get_db_connection() as conn:
        conn.cursor().execute(
            "UPDATE data_sources SET local_path = ?, content_hash = ?, status = ?, updated_at = ? WHERE id = ?",
            (str(save_path), content_hash, 'processed', time.strftime('%Y-%m-%d %H:%M:%S'), source_id)
        )
        conn.commit()
    logger.info(f"Successfully processed and saved {url} to {save_path}")
    return response.content if file_ext == '.html' else None

def crawl_with_requests(job: dict):
    """Crawls a domain using the requests library for static sites."""
    session = get_session()
    dest_folder = Path("data/raw")

    queue = deque([(job['start_url'], 0)])
    visited_urls = {job['start_url']}

    logger.info(f"Starting REQUESTS crawl for '{job['name']}'")

    while queue:
        current_url, current_depth = queue.popleft()
        if current_depth > job['crawl_depth']:
            continue

        logger.info(f"Crawling (depth {current_depth}): {current_url}")

        source_id = add_discovered_source(current_url, job['domain_lock'], 'html')
        html_content = download_and_save_content(session, current_url, dest_folder, source_id)

        if not html_content or current_depth >= job['crawl_depth']:
            continue

        soup = BeautifulSoup(html_content, 'lxml')
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(current_url, href).split('#')[0]

            if not full_url.startswith(('http', 'https')):
                continue

            if full_url not in visited_urls and urlparse(full_url).netloc.endswith(job['domain_lock']):
                visited_urls.add(full_url)
                if full_url.lower().endswith('.pdf'):
                    pdf_id = add_discovered_source(full_url, job['domain_lock'], 'pdf')
                    download_and_save_content(session, full_url, dest_folder, pdf_id)
                else:
                    queue.append((full_url, current_depth + 1))
        time.sleep(1)

def crawl_with_selenium(driver: webdriver.Chrome, job: dict):
    """Crawls a domain using Selenium for dynamic sites."""
    session = get_session()
    dest_folder = Path("data/raw")

    queue = deque([(job['start_url'], 0)])
    visited_urls = {job['start_url']}

    logger.info(f"Starting SELENIUM crawl for '{job['name']}'")

    while queue:
        current_url, current_depth = queue.popleft()
        if current_depth > job['crawl_depth']:
            continue

        logger.info(f"Crawling (depth {current_depth}): {current_url}")
        try:
            driver.get(current_url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(3)
            page_source = driver.page_source
        except (TimeoutException, WebDriverException) as e:
            logger.error(f"Selenium failed to get {current_url}: {e}")
            continue

        source_id = add_discovered_source(current_url, job['domain_lock'], 'html')

        sanitized_name = sanitize_filename(current_url)
        final_filename = f"source_{source_id}_{sanitized_name}.html"
        save_path = dest_folder / final_filename

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(page_source, encoding='utf-8')

        content_hash = get_content_hash(page_source.encode('utf-8'))
        with get_db_connection() as conn:
            conn.cursor().execute(
                "UPDATE data_sources SET local_path = ?, content_hash = ?, status = ?, updated_at = ? WHERE id = ?",
                (str(save_path), content_hash, 'processed', time.strftime('%Y-%m-%d %H:%M:%S'), source_id)
            )
            conn.commit()
        logger.info(f"Successfully processed and saved {current_url} to {save_path}")

        if current_depth >= job['crawl_depth']:
            continue

        soup = BeautifulSoup(page_source, 'lxml')
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(current_url, href).split('#')[0]

            if not full_url.startswith(('http', 'https')):
                continue

            if full_url not in visited_urls and urlparse(full_url).netloc.endswith(job['domain_lock']):
                visited_urls.add(full_url)
                if full_url.lower().endswith('.pdf'):
                    pdf_id = add_discovered_source(full_url, job['domain_lock'], 'pdf')
                    download_and_save_content(session, full_url, dest_folder, pdf_id)
                else:
                    queue.append((full_url, current_depth + 1))
