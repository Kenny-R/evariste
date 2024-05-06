import requests
import pandas as pd
from bs4 import BeautifulSoup

tags = ['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a','span', 'li', 'td', 'th', 'label']

def extract_html(url):
    """Extracts html code from a page

    Args:
        url (string): Page's link

    Returns:
        string: Page's html code
    """
    request = requests.get(url)

    if request.status_code == 200:
        return request.text
    
    return None

def extract_text(html):
    """Extracts text contents from a html code

    Args:
        html (string): Html code

    Returns:
        list: List of text contents
    """
    soup = BeautifulSoup(html, 'html.parser')

    p_tags = soup.find(id='bodyContent').find_all(tags)
    p_contents = [tag.text for tag in p_tags if tag.parent.name not in tags]

    return p_contents

def extract_pages_info(filename):
    """Extracts information from a page

    Args:
        filename (string): Path of the CSV with countries and pages

    Returns:
        _type_: _description_
    """
    countries = pd.read_csv(filename, sep=';')
    countries_dic = countries.set_index(countries.columns[0])[countries.columns[1]].to_dict()
    countries_info = {}

    for country in countries_dic.keys():
        url = countries_dic[country]  
        html = extract_html(url)

        if html is not None:
            countries_info[country] = extract_text(html)
        else:
            print(f"Could not get the HTML of the page {url}.")

    return countries_info
