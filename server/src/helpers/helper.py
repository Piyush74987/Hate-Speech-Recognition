import requests
from bs4 import BeautifulSoup
import os


def get_visible_text(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        visible_text = soup.get_text(separator="\n", strip=True)
        return visible_text
    else:
        return None


def get_cur_dir():
    return os.path.dirname(os.path.abspath(__file__))
