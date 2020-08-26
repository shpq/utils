import argparse
import time
from ..core.selenium_core import Bot
from random import random


class CelebrityScraper:
    def __init__(self, chromedriver_path):
        self.bot = Bot(chromedriver_path=chromedriver_path)
        self.search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    def scroll_to_end(self):
        self.bot.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")

    def get_thumbnail_results(self):
        return self.bot.browser.find_element_by_css_selector("img.Q4LuWd")

    def get_actual_images(self):
        return self.bot.browser.find_element_by_css_selector("img.n3VNCb")

    def get_image_urls(self, query, num_urls, sleep_time=2):
        image_urls = set()
        image_count = 0
        results_start = 0
        while image_count < num_urls:
            self.scroll_to_end()
            thumbnail_results = self.get_thumbnail_results()
            number_results = len(thumbnail_results)
            for img in thumbnail_results[results_start:number_results]:
                try:
                    img.click()
                    time.sleep(sleep_time + 2 * random())
                except Exception:
                    continue

                actual_images = self.get_actual_images()
                for actual_image in actual_images:
                    src = actual_image.get_attribute("src")
                    if src and "http" in src:
                        image_urls.add(src)

                image_count = len(image_urls)

                if len(image_urls) >= num_urls:
                    break

            results_start = len(thumbnail_results)

        return image_urls
