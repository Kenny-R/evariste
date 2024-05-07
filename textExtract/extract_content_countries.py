import requests
import pandas as pd
from lxml import html

TAGS = ['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a','span', 'label', 'li']

def extract_html(url: str) -> str | None:
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

def extract_text(html_str: str, content_xpath: str) -> list[str]:
    """Extracts text contents from a html code

    Args:
        html (string): Html code
        
        content_xpath: A string with the xpath to the content main
                       label

    Returns:
        list: List of text contents
    """
    root = html.fromstring(html_str)

    content = []
    stack = [root.xpath(content_xpath)[0]]
    
    while stack:
        node = stack.pop()
        if node.tag == 'tr' or node.tag == 'td' or node.tag == 'th' or node.tag == 'tbody':
            continue
        if node.tag in TAGS:
            if node.tag in ['h1','h2','h3','h4','h5','h6']:
                content.append('\n')
                content.append(node.text_content())
                content.append('\n')
            else: 
                content.append(node.text_content())

            continue

        stack.extend(reversed(node))
    
    return content

    
def extract_pages_info(filename: str, 
                       content_xpath:str = '//div[@id="bodyContent"]') -> dict[str, str]:
    """Extracts information from a page

    Args:
        filename (string): Path of the CSV with countries and pages

        content_xpath: A string with the xpath to the content main
                       label

    Returns:
        dictionary: Dictionary with countries as keys and information as value
    """
    countries = pd.read_csv(filename, sep=';')
    countries_dic = countries.set_index(countries.columns[0])[countries.columns[1]].to_dict()
    countries_info = {}

    for country in countries_dic.keys():
        url = countries_dic[country]  
        html = extract_html(url)

        countries_info[country] = extract_text(html, content_xpath) if html is not None else f"Could not get the HTML of the page {url}."

    return countries_info