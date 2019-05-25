import sys
import requests
import urllib.request
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
	if "cub" in destination:
		print("Info:downloading bird captions")
	elif "flowers" in destination:
		print("Info:downloading flower captions")

	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params={ 'id': id }, stream=True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params=params, stream=True)
	save_response_content(response, destination)

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value
	return None

def save_response_content(response, destination, chunk_size=32*1024):
	with open(destination, "wb") as f:
		for chunk in response.iter_content(chunk_size):
			if chunk:f.write(chunk)

def downloder( url, save_path=None):
    file_name = url.split("/")[-1]
    print(f"Info:downloading {file_name}")
    urllib.request.urlretrieve(url, save_path, reporthook=progress)
    print("")

def progress( block_count, block_size, total_size ):
    ''' コールバック関数 '''
    if total_size < 0:total_size = 1
    percentage = block_count * block_size / total_size
    n_sharp = int(50 * percentage)
    n_dot = 50 - n_sharp
    log_text = "progress  [" + "#"*n_sharp + "."*n_dot + "] "
    sys.stdout.write(log_text + f"{percentage:.2%} ( {int(total_size / (1024*1024)):,d} MB )\r")
