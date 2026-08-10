import logging
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from policypilot.config import settings
from policypilot.services.database import setup_database, initialize_crawl_jobs
from scripts.data_processing.crawler import crawl_with_requests, crawl_with_selenium

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_selenium_driver():
    """Initializes a headless Chrome WebDriver."""
    logger.info("Setting up Selenium WebDriver...")
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        driver_path = os.getenv("CHROMEDRIVER_PATH")
        service = ChromeService(driver_path or ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        logger.info("WebDriver setup complete.")
        return driver
    except Exception as e:
        logger.error(f"Failed to setup Selenium driver: {e}")
        return None

def main():
    """Main function to run the data crawling jobs."""
    logger.info("--- Starting PolicyPilot AI Data Acquisition ---")

    # Setup DB and initialize starting URLs
    setup_database()
    initialize_crawl_jobs()

    driver = None
    try:
        # Initialize Selenium driver only if needed
        if any(job.get('method') == 'selenium_crawl' and job.get('status') == 'active' for job in settings.CRAWLING_JOBS):
            driver = setup_selenium_driver()

        for job in settings.CRAWLING_JOBS:
            if job.get('status') != 'active':
                logger.info(f"--- Skipping inactive job: {job['name']} ---")
                continue

            logger.info(f"--- Processing job: {job['name']} ---")
            if job['method'] == 'selenium_crawl':
                if driver:
                    crawl_with_selenium(driver, job)
                else:
                    logger.error(f"Selenium method required for {job['name']} but driver failed to initialize. Skipping.")
            elif job['method'] == 'requests_crawl':
                crawl_with_requests(job)
            else:
                logger.warning(f"Method '{job['method']}' not implemented for job {job['name']}. Skipping.")

    finally:
        if driver:
            driver.quit()
            logger.info("Selenium WebDriver closed.")

    logger.info("--- Data Acquisition Process Finished ---")

if __name__ == "__main__":
    main()
