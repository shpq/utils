from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


class Bot:
    def __init__(self, chromedriver_path, login=None, password=None):
        self.chrome_options = Options()
        self.chrome_options.add_argument("--window-size=1024x1400")
        self.browser = webdriver.Chrome(
            options=self.chrome_options, executable_path=chromedriver_path)
        self.login = login
        self.password = password

    def reload(self):
        self.browser.execute_script("location.reload()")

    def add_cookies(self, cookies, reload=True):
        for c in cookies:
            self.browser.add_cookies(c)
        self.reload()

    def send_key(self, element, key):
        get_key = {"down": Keys.PAGE_DOWN,
                   "home": Keys.HOME,
                   "end": Keys.END}
        element.send_keys(get_key[key])

    def execute_script(self, script):
        return self.browser.execute_script(script)
