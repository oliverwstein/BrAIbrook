import requests
import json
import os
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def create_directory(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def sanitize_filename(filename):
    """Sanitize a filename to remove invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def download_image(url, filepath):
    """Download an image from a URL to a given filepath."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filepath, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def extract_metadata_and_iiif(driver, manuscript_url):
    """Extract metadata and IIIF manifest ID from the page description using Selenium."""
    try:
        driver.get(manuscript_url)

        # Wait for both the metadata section and the IIIF manifest link to be present
        WebDriverWait(driver, 90).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'dl.document-metadata'))
        )
        WebDriverWait(driver, 90).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a.hl__viewer-link'))
        )

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract metadata
        metadata = {}
        description_dl = soup.find('dl', class_='document-metadata')
        if description_dl:
            for dt in description_dl.find_all('dt'):
                dd = dt.find_next_sibling('dd')
                if dd:
                    key = dt.text.strip()
                    value = dd.text.strip()
                    # Handle multiple values within a single dd
                    if dd.find_all('a'):
                        values = [a.text.strip() for a in dd.find_all('a')]
                        metadata[key] = values
                    else:
                        metadata[key] = value

        # Extract IIIF manifest ID using regex, and then by string matching for ids:
        iiif_manifest_id = None
        ids_system = False
        match = re.search(r'manifestId=(https://iiif\.lib\.harvard\.edu/manifests/drs:(\d+))', soup.prettify())
        if match:
            iiif_manifest_id = match.group(1)
        else:
            ids_match = re.search(r'manifests/ids:(\d+)', soup.prettify())
            if ids_match:
                iiif_manifest_id = ids_match.group(0) # manifest/ids:####
                ids_system = True

        # Add the IIIF manifest ID to the metadata
        if iiif_manifest_id:
            metadata['iiif_manifest_id'] = iiif_manifest_id
            metadata['ids_system'] = ids_system #indicate if the ids system is used
        else:
            print(f"Warning: No IIIF manifest found for {manuscript_url}")

        return metadata

    except TimeoutException:
        print(f"Timeout waiting for elements to load on page: {manuscript_url}")
        return {}, None

def download_iiif_images(manifest_id, manuscript_dir, ids_system):
    """Download images from a IIIF manifest."""
    if ids_system:
         manifest_url = f"https://iiif.lib.harvard.edu/{manifest_id}"
    else:
        manifest_url = manifest_id
    
    try:
        response = requests.get(manifest_url)
        response.raise_for_status()
        manifest = response.json()

        image_urls = []
        if 'sequences' in manifest and manifest['sequences']:
            canvases = manifest['sequences'][0].get('canvases', [])
            for canvas in canvases:
                if 'images' in canvas and canvas['images']:
                    for image in canvas['images']:
                        if 'resource' in image and '@id' in image['resource']:
                            image_url = image['resource']['@id']
                            if image_url.endswith('default.jpg'):
                                image_urls.append(image_url)

        with ThreadPoolExecutor(max_workers=5) as executor:
            image_paths = [os.path.join(manuscript_dir, f"{i:04}.jpg") for i in range(len(image_urls))]
            list(tqdm(executor.map(download_image, image_urls, image_paths), total=len(image_urls), desc=f"Downloading images from IIIF manifest"))
    except requests.exceptions.RequestException as e:
        print(f"Error downloading IIIF manifest: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from IIIF manifest: {e}")


def scrape_houghton_manuscripts(base_url, num_pages, output_dir):
    """
    Scrape images and metadata from the Houghton Library's digital manuscripts.

    Args:
        base_url (str): The base URL for the list of manuscripts.
        num_pages (int): The number of pages of manuscripts to scrape.
        output_dir (str): The directory to save the scraped data.
    """
    raw_dir = os.path.join(output_dir, "raw")
    create_directory(raw_dir)

    # Configure Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode (no browser window)
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--blink-settings=imagesEnabled=false')  # Disable images to speed up
    driver = webdriver.Chrome(options=options)

    manuscript_urls = set()

    # Scrape list of manuscript URLs
    for page_num in tqdm(range(4, num_pages + 1), desc="Scraping manuscript list"):
        url = f"{base_url}&page={page_num}&per_page=96"

        try:
            driver.get(url)

            # Wait for a more reliable element to be present (e.g., the search results container)
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '#documents.documents-list'))
            )
            
            # Now that the page is loaded, parse it with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Select a tags with title links
            for a in soup.select('h3.index_title a'):
                link = a['href']
                full_url = "https://curiosity.lib.harvard.edu" + link
                manuscript_urls.add(full_url)

        except TimeoutException:
            print(f"Timeout waiting for page to load: {url}")

    # Scrape metadata and images for each manuscript
    for manuscript_url in tqdm(manuscript_urls, desc="Scraping manuscripts"):
        # Extract title for directory name
        driver.get(manuscript_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        title_element = soup.select_one('h1.item-title')
        if title_element:
            title = title_element.text.strip()
        else:
            title = os.path.basename(manuscript_url)  # Use URL as fallback
        
        # Create directory for manuscript
        manuscript_dir = os.path.join(raw_dir, sanitize_filename(title))

        # Skip scraping if images/folder already exist
                # Create directory for manuscript
        manuscript_dir = os.path.join(raw_dir, sanitize_filename(title))

        # Skip scraping if images already exist
        if os.path.exists(manuscript_dir):
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) for f in os.listdir(manuscript_dir)):
                continue
        else:
            print(f"Create directory for {manuscript_url}")
        create_directory(manuscript_dir)


        # Extract metadata from page description using Selenium
        metadata = extract_metadata_and_iiif(driver, manuscript_url)

        # Save metadata
        metadata_path = os.path.join(manuscript_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        # Download images from IIIF manifest
        if metadata.get('iiif_manifest_id'):
            download_iiif_images(metadata.get('iiif_manifest_id'), manuscript_dir, metadata.get('ids_system'))
        else:
            print(f"Warning: No IIIF manifest found for {manuscript_url}")

    driver.quit()  # Close the browser
    return manuscript_urls

def main():
    base_url = "https://curiosity.lib.harvard.edu/medieval-renaissance-manuscripts/catalog?f%5Bseries_ssim%5D%5B%5D=Digital+Medieval+Manuscripts+at+Houghton+Library"
    num_pages = 4
    output_dir = "data"

    manuscript_urls = scrape_houghton_manuscripts(base_url, num_pages, output_dir)
    print(len(manuscript_urls))
    print(len([item for item in os.listdir("data/raw") if os.path.isdir(os.path.join("data/raw", item))]))

if __name__ == "__main__":
    main()