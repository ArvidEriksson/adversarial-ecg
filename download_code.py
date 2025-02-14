import requests
import os
from urllib.parse import urlparse
from pathlib import Path
import argparse
from tqdm import tqdm

def get_record_files(record_id):
    """
    Fetch the list of files associated with a Zenodo record.
    """
    base_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(base_url)
    response.raise_for_status()
    
    record_data = response.json()
    return record_data.get('files', [])

def download_file(url, destination, filename):
    """
    Download a file with progress bar.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    file_size = int(response.headers.get('content-length', 0))
    
    filepath = os.path.join(destination, filename)
    
    with open(filepath, 'wb') as f, tqdm(
        desc=filename,
        total=file_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            progress_bar.update(size)

def download_zenodo_repository(record_id, destination=None):
    """
    Download all files from a Zenodo repository.
    
    Args:
        record_id (str): The Zenodo record ID
        destination (str, optional): Directory to save files. Defaults to current directory.
    """
    if destination is None:
        destination = os.getcwd()
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination, exist_ok=True)
    
    try:
        files = get_record_files(record_id)
        print(f"Found {len(files)} files in repository")
        
        for file_info in files:
            filename = file_info['key']
            download_url = file_info['links']['self']
            
            print(f"\nDownloading: {filename}")
            download_file(download_url, destination, filename)
            
        print("\nDownload completed successfully!")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to access Zenodo repository: {e}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download files from a Zenodo repository')
    parser.add_argument('record_id', help='Zenodo record ID')
    parser.add_argument('--destination', '-d', help='Destination directory for downloaded files')
    
    args = parser.parse_args()
    
    download_zenodo_repository(args.record_id, args.destination)

if __name__ == '__main__':
    main()