import argparse
import os
import random
import re
import time
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://books.toscrape.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    """Descarga una URL y devuelve el árbol BeautifulSoup parseado.
    Params:
        session: Sesión HTTP reutilizable de requests.
        url: URL absoluta a solicitar.
    Returns:
        Objeto BeautifulSoup del HTML; lanza error si el código HTTP no es 200.
    """
    resp = session.get(url, headers=HEADERS, timeout=(10, 30))
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_category_links(session: requests.Session) -> List[Dict[str, str]]:
    """Extrae desde la página principal todas las categorías con su nombre y URL absoluta."""
    soup = get_soup(session, urljoin(BASE_URL, "index.html"))
    links = []
    for a in soup.select("div.side_categories ul li ul li a"):
        name = a.get_text(strip=True)
        href = urljoin(BASE_URL, a.get("href"))
        links.append({"name": name, "url": href})
    return links


def iterate_category_pages(session: requests.Session, category_url: str) -> Iterable[str]:
    """Generador que recorre la paginación de una categoría y va devolviendo las URLs de cada página."""
    url = category_url
    while True:
        yield url
        soup = get_soup(session, url)
        next_a = soup.select_one("li.next a")
        if not next_a:
            break
        url = urljoin(url, next_a.get("href"))


def parse_book_links(session: requests.Session, page_url: str) -> List[str]:
    """Dada una URL de listado, extrae y devuelve las URLs absolutas de cada libro listado."""
    soup = get_soup(session, page_url)
    links = []
    for a in soup.select("article.product_pod h3 a"):
        href = a.get("href")
        abs_url = urljoin(page_url, href)
        links.append(abs_url)
    return links


def parse_product_page(session: requests.Session, product_url: str, category_name: Optional[str] = None) -> Optional[Dict]:
    """Parsea la página de un producto y construye un diccionario de características.
    Devuelve None si hay errores al solicitar o parsear.
    """
    try:
        soup = get_soup(session, product_url)
    except Exception:
        return None

    # Title
    title_el = soup.select_one("div.product_main h1")
    title = title_el.get_text(strip=True) if title_el else None

    # Price (e.g., '£51.77')
    price_el = soup.select_one("p.price_color")
    price = None
    if price_el:
        price_text = price_el.get_text(strip=True)
        m = re.search(r"[0-9]+\.?[0-9]*", price_text)
        price = float(m.group()) if m else None

    # Availability (e.g., 'In stock (22 available)')
    avail_el = soup.select_one("p.instock.availability")
    availability = 0
    if avail_el:
        m = re.search(r"(\d+)", avail_el.get_text())
        availability = int(m.group(1)) if m else 0

    # Rating from classes: 'star-rating Three'
    rating_el = soup.select_one("p.star-rating")
    rating = None
    if rating_el and rating_el.has_attr("class"):
        for cls in rating_el["class"]:
            if cls in RATING_MAP:
                rating = RATING_MAP[cls]
                break

    # Category from breadcrumb
    cat_el = soup.select_one("ul.breadcrumb li:nth-of-type(3) a")
    category = cat_el.get_text(strip=True) if cat_el else category_name

    # Description length
    desc_el = soup.select_one("#product_description ~ p")
    description = desc_el.get_text(strip=True) if desc_el else ""
    description_len = len(description)
    has_desc = 1 if description else 0

    # Number of reviews (from product info table)
    n_reviews = 0
    for row in soup.select("table.table.table-striped tr"):
        th = row.select_one("th")
        td = row.select_one("td")
        if not th or not td:
            continue
        if th.get_text(strip=True).lower() == "number of reviews":
            try:
                n_reviews = int(td.get_text(strip=True))
            except Exception:
                n_reviews = 0
            break

    # Title length
    title_len = len(title) if title else 0

    return {
        "title": title,
        "price": price,
        "category": category,
        "rating": rating,
        "availability": availability,
        "description_len": description_len,
        "has_desc": has_desc,
        "n_reviews": n_reviews,
        "title_len": title_len,
        "url": product_url,
    }


def scrape_books(max_items: Optional[int] = None, min_delay: float = 0.2, max_delay: float = 0.6) -> pd.DataFrame:
    """Scrapea el sitio completo (o hasta max_items) y devuelve un DataFrame con los registros.
    Respeta un retardo aleatorio entre peticiones para ser cortés.
    """
    session = requests.Session()
    categories = parse_category_links(session)

    data: List[Dict] = []
    seen: Set[str] = set()

    for cat in categories:
        for page_url in iterate_category_pages(session, cat["url"]):
            book_links = parse_book_links(session, page_url)
            for link in book_links:
                if link in seen:
                    continue
                item = parse_product_page(session, link, category_name=cat["name"]) 
                if item and item["price"] is not None:
                    data.append(item)
                    seen.add(link)
                # polite delay
                time.sleep(random.uniform(min_delay, max_delay))
                if max_items and len(data) >= max_items:
                    return pd.DataFrame(data)
    return pd.DataFrame(data)


def main():
    """CLI del scraper: ejecuta la recolección y guarda data/dataset.csv."""
    parser = argparse.ArgumentParser(description="Scraper: Books to Scrape → dataset.csv")
    parser.add_argument("--out", type=str, default=os.path.join("data", "dataset.csv"), help="Ruta de salida del CSV")
    parser.add_argument("--max-items", type=int, default=0, help="Máximo de items a recolectar (0 = todos)")
    parser.add_argument("--min-delay", type=float, default=0.2, help="Retardo mínimo entre requests (s)")
    parser.add_argument("--max-delay", type=float, default=0.6, help="Retardo máximo entre requests (s)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    max_items = args.max_items if args.max_items and args.max_items > 0 else None
    df = scrape_books(max_items=max_items, min_delay=args.min_delay, max_delay=args.max_delay)

    # Limpieza básica: eliminar nulos y mantener tipos correctos
    df = df.dropna(subset=["price", "rating", "category"]).reset_index(drop=True)

    df.to_csv(args.out, index=False)
    print(f"Guardado: {args.out} ({len(df)} registros)")


if __name__ == "__main__":
    main()
