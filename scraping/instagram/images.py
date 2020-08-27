import argparse
from time import sleep
from ..core.selenium_core import Bot
from random import random
import sys
import os


script = """
    function get_pics(){
        d = []
        for (i=0; i<document.getElementsByClassName('FFVAD').length; i++){
        d.push({"url" : document.getElementsByClassName('FFVAD')[i].src,
                "text" : document.getElementsByClassName('FFVAD')[i].alt})
        }
        return d
}
    return get_pics()
"""


class InstagramScraper:

    def __init__(self, chromedriver_path, cookies=None, save_path=None):

        self.bot = Bot(chromedriver_path=chromedriver_path)
        self.bot.browser.get("https://www.instagram.com")
        if cookies:
            cookies = self.clear_cookies(cookies)
            self.bot.add_cookies(cookies, reload=True)

    def clear_cookies(self, cookies):

        for c in cookies:
            if "expiry" in c.keys():
                del c["expiry"]

        return cookies

    def find_body(self):
        return self.bot.browser.find_element_by_css_selector("body")

    def is_banned(self):
        return self.bot.browser.find_element_by_class_name("gxNyb")

    def get_image_urls(self, url, num_urls, filename):
        pics_chunk = []
        s = 0
        r = 0
        self.bot.browser.get(url)
        while len(pics_chunk) > num_urls:
            if r == len(pics_chunk):
                s += 1
            else:
                s = 0

            if s >= 50:
                break

            sleep(0.5 + random())
            body = self.find_body()
            self.bot.send_key(body, "down")
            result = self.bot.execute_script(script)
            result = [x for x in result if x["text"]]
            for el in result:
                el["src"] = self.bot.browser.current_url

            pics_chunk += result
            pics_chunk = [dict(t)
                          for t in {tuple(d.items()) for d in pics_chunk}]
            sys.stdout.write(f"{len(pics_chunk)}\n{url}")
            sys.stdout.flush()
            try:
                if self.is_banned():
                    self.bot.send_key(body, "home")
                    sleep(20 + 3*random())
                    self.bot.send_key(body, "end")
                    s -= 0.5
            except:
                pass
        if not save_path:
            pd.DataFrame(pics_chunk).to_csv(filename + ".csv")
        else:
            if not os.path.exists(save_path):
                os.mkdirs(save_path)

            pd.DataFrame(pics_chunk).to_csv(os.path.join(save_path, filename + ".csv"))
