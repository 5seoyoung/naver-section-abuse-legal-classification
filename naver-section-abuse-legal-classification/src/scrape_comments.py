import os
import time
import argparse
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException


def setup_driver(headless: bool = True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    return driver


def open_comments_panel(driver):
    time.sleep(2)
    selectors = [
        "a.u_cbox_btn_view_comment",
        "a.c_cmt_btn",
    ]
    for sel in selectors:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            btn.click()
            time.sleep(2)
            return
        except NoSuchElementException:
            continue
    return


def load_more_comments(driver, max_clicks: int = 30, wait_sec: float = 1.0):
    for i in range(max_clicks):
        try:
            more_btn = driver.find_element(By.CSS_SELECTOR, "a.u_cbox_btn_more")
            driver.execute_script("arguments[0].click();", more_btn)
            time.sleep(wait_sec)
        except NoSuchElementException:
            break
        except ElementClickInterceptedException:
            time.sleep(wait_sec)
            continue


def extract_comments(driver):
    comment_elements = driver.find_elements(By.CSS_SELECTOR, "span.u_cbox_contents")
    comments = []
    for idx, elem in enumerate(comment_elements):
        text = elem.text.strip()
        if text:
            comments.append(text)
    return comments


def scrape_naver_comments(article_url: str, max_clicks: int = 30, headless: bool = True):
    driver = setup_driver(headless=headless)
    driver.get(article_url)
    time.sleep(2)

    open_comments_panel(driver)
    load_more_comments(driver, max_clicks=max_clicks)
    comments = extract_comments(driver)

    driver.quit()
    return comments


def save_comments_to_csv(comments, section: str, out_dir: str = "data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(
        {
            "id": [f"{section}_{i+1:03d}" for i in range(len(comments))],
            "section": section,
            "comment_text": comments,
        }
    )
    out_path = os.path.join(out_dir, f"comments_{section}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} comments to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="Naver news article URL")
    parser.add_argument(
        "--section",
        type=str,
        required=True,
        choices=["politics", "society", "entertainment"],
        help="news section",
    )
    parser.add_argument(
        "--max_clicks",
        type=int,
        default=30,
        help="maximum number of 'more comments' clicks",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run chrome in headless mode",
    )
    args = parser.parse_args()

    comments = scrape_naver_comments(
        article_url=args.url,
        max_clicks=args.max_clicks,
        headless=args.headless,
    )
    print(f"Collected {len(comments)} comments")

    save_comments_to_csv(comments, section=args.section, out_dir="data/raw")


if __name__ == "__main__":
    main()
