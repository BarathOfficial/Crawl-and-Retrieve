import requests
import trafilatura
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin
import logging


def crawler(url: str, visited: set, data: list, urls: list, depth: int = 0, max_depth: int = 0):
    
    try:
        if url in visited:
            return
        
        visited.add(url)
        
        html = requests.get(url, headers={'User-Agent':"Mozilla/5.0"}, timeout=10).text
        extracted_text = trafilatura.extract(html)
        if extracted_text:
            data.append(extracted_text)
            urls.append(url)
            
        if depth < max_depth:
            soup = bs(html, 'html.parser')
            for a in soup.find_all('a', href=True):
                next_url = urljoin(url, a['href'])
                crawler(next_url, visited, data, urls, depth + 1, max_depth)
        logging.info(f'crawler successfully gathered data from: {url}')
        return data

    except Exception as e:
        logging.error(f'error: {e}')

