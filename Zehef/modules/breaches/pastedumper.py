import random
from bs4 import BeautifulSoup
from lib.Requests import Request

class Pastebin_Dumper:
    def __init__(self, target: str) -> None:
        self.target = target
        self.dork = f"site:pastebin.com \"{target}\""
        self.links = []
        self.ua = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/123.0.6312.52 Mobile/15E148 Safari/604.1"
        ]

    async def google_dorks_scraper(self):
        """Scrapes Google search results for Pastebin links"""
        try:
            r = await Request(f"https://www.google.com/search?q={self.dork}", headers={"User-Agent": random.choice(self.ua)}).get()
            soup = BeautifulSoup(r.text, 'html.parser')
            search_results = soup.find_all('div', class_='tF2Cxc')

            for result in search_results:
                link = result.find('a')['href']
                if "pastebin.com" in link:
                    self.links.append(link)

        except Exception as e:
            return {"source": "Pastebin", "status": "error", "message": f"Google scraping failed: {str(e)}"}

    async def paste_check(self):
        """Checks Pastebin links for the target email"""
        await self.google_dorks_scraper()

        found_pastes = []
        for link in self.links:
            raw_link = link.replace('https://pastebin.com/', 'https://pastebin.com/raw/')

            try:
                r = await Request(raw_link).get()
                if str(self.target).lower() in r.text.lower():
                    found_pastes.append({"paste_url": link, "raw_url": raw_link})

            except Exception as e:
                return {"source": "Pastebin", "status": "error", "message": f"Failed to check pastebin link: {str(e)}"}

        if found_pastes:
            return {"source": "Pastebin", "status": "found", "pastes": found_pastes}
        else:
            return {"source": "Pastebin", "status": "not found"}
