import asyncio
import logging
from playwright.async_api import async_playwright, Page
from playwright_stealth import stealth_async
from datetime import datetime
from src.database.dependencies import session
from src.schemas.article import ArticleCreate
from src.services.article_service import ArticleService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class StackExchangeScraper:
    API_URL = "https://stackexchange.com/sites"

    def __init__(self):
        pass

    async def crawl_article(self, page: Page, url: str):
        try:
            await page.goto(url)
            await stealth_async(page)
            title = await page.text_content("h1[itemprop='name']")
            author = await page.text_content(".user-details a")
            content = await page.text_content(".question .post-text")
            abstract = content[:200] + "…" if content else ""
            return {
                "title": title.strip() if title else "No Title",
                "author": author.strip() if author else "Unknown",
                "abstract": abstract.strip(),
                "content": content.strip() if content else "",
                "source_url": url,
                "created_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error crawling article at {url}: {e}")
            return None

    async def scrape(self, site_limit=5, question_limit=10):
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage"
            ])
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)…",
                locale="en-US",
                timezone_id="Europe/Paris",
            )
            page = await context.new_page()
            try:
                await page.goto(self.API_URL)
                await stealth_async(page)
                hrefs = await page.eval_on_selector_all(
                    ".grid-layout .site-container a.site-link",
                    f"els => els.slice(0,{site_limit}).map(e => e.href)"
                )
                for site_url in hrefs:
                    try:
                        await page.goto(site_url)
                        await stealth_async(page)
                        q_links = await page.eval_on_selector_all(
                            ".question-summary .question-hyperlink",
                            f"els => els.slice(0,{question_limit}).map(e => e.href)"
                        )
                        for q in q_links:
                            data = await self.crawl_article(page, q)
                            if data:
                                self.save_direct(data)
                                logger.info(f"Saved: {data['title']}")
                    except Exception as e:
                        logger.error(f"Error scraping site {site_url}: {e}")
            except Exception as e:
                logger.error(f"Error scraping StackExchange sites: {e}")
            finally:
                await browser.close()

    def save_direct(self, data: dict):
        db = session()
        try:
            dto = ArticleCreate(**data)
            ArticleService(db).create_article(dto)
        except Exception as e:
            logger.error(f"Error saving article to DB: {e}")
        finally:
            db.close()

# Utility function to run in background
async def run_scraper():
    scraper = StackExchangeScraper()
    await scraper.scrape()

if __name__ == "__main__":
    import sys
    import asyncio
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_scraper())
