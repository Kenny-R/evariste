import requests
import pandas as pd
from lxml import html

tags = ['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a','span', 'label', 'li']

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

def extract_text(html_str):
    """Extracts text contents from a html code

    Args:
        html (string): Html code

    Returns:
        list: List of text contents
    """
    root = html.fromstring(html_str)

    content = []
    stack = [root.xpath('//div[@id="bodyContent"]')[0]]
    
    while stack:
        node = stack.pop()
        if node.tag == 'tr' or node.tag == 'td' or node.tag == 'th' or node.tag == 'tbody':
            continue
        if node.tag in tags:
            if node.tag in ['h1','h2','h3','h4','h5','h6']:
                content.append('\n')
                content.append(node.text_content())
                content.append('\n')
            else: 
                content.append(node.text_content())

            continue

        stack.extend(reversed(node))
    
    return content

    
def extract_pages_info(filename):
    """Extracts information from a page

    Args:
        filename (string): Path of the CSV with countries and pages

    Returns:
        dictionary: Dictionary with countries as keys and information as value
    """
    countries = pd.read_csv(filename, sep=';')
    countries_dic = countries.set_index(countries.columns[0])[countries.columns[1]].to_dict()
    countries_info = {}

    for country in countries_dic.keys():
        url = countries_dic[country]  
        html = extract_html(url)

        countries_info[country] = extract_text(html) if html is not None else f"Could not get the HTML of the page {url}."

    return countries_info