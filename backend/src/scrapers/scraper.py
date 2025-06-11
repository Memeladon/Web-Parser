import sys
import time
import random
import logging
from playwright.sync_api import sync_playwright, Page
from playwright_stealth import stealth_sync
from src.database.dependencies import session
from src.schemas.article import ArticleCreate
from datetime import datetime
from src.services.article_service import ArticleService
from src.database.repositories.article_repository import ArticleRepository
import json
import os

logger = logging.getLogger(__name__)

# Список современных user-agent'ов
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
]

# Список URL-адресов для парсинга вопросов
QUESTION_LIST_URLS = [
    "https://scifi.stackexchange.com/questions?sort=MostVotes&edited=true",
    "https://cstheory.stackexchange.com/questions?sort=MostVotes&edited=true",
    "https://movies.stackexchange.com/questions?sort=MostVotes&edited=true",
    "https://math.stackexchange.com/questions?sort=MostVotes&edited=true",
    "https://physics.stackexchange.com/questions?sort=MostVotes&edited=true",
    "https://chemistry.stackexchange.com/questions?sort=MostVotes&edited=true",
    "https://biology.stackexchange.com/questions?sort=MostVotes&edited=true",
    "https://history.stackexchange.com/questions?sort=MostVotes&edited=true",
    "https://philosophy.stackexchange.com/questions?sort=MostVotes&edited=true",
    "https://linguistics.stackexchange.com/questions?sort=MostVotes&edited=true"
]

class StackExchangeSiteScraper:
    def __init__(self, post_limit=10, question_list_urls=None, proxy=None):
        self.saved_count = 0
        self.post_limit = post_limit
        self.scraped_posts = 0
        self.question_list_urls = question_list_urls or QUESTION_LIST_URLS
        self.proxy = proxy
        self.current_user_agent = random.choice(USER_AGENTS)

    def get_browser_context(self, playwright):
        browser_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
            "--disable-site-isolation-trials"
        ]

        browser = playwright.chromium.launch(
            headless=True,
            args=browser_args
        )

        context_options = {
            "user_agent": self.current_user_agent,
            "locale": "en-US",
            "timezone_id": "Europe/Paris",
            "viewport": {"width": 1920, "height": 1080},
            "device_scale_factor": 1,
            "has_touch": False,
            "is_mobile": False,
            "java_script_enabled": True,
            "ignore_https_errors": True
        }

        if self.proxy:
            context_options["proxy"] = {
                "server": self.proxy,
                "username": os.getenv("PROXY_USERNAME"),
                "password": os.getenv("PROXY_PASSWORD")
            }

        context = browser.new_context(**context_options)
        
        # Добавляем дополнительные заголовки, чтобы выглядеть как реальный браузер
        context.set_extra_http_headers({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0"
        })

        return browser, context

    def handle_captcha(self, page):
        try:
            # Проверка на наличие элементов капчи
            captcha_present = page.query_selector("iframe[src*='captcha']") or \
                            page.query_selector("div[class*='captcha']") or \
                            page.query_selector("form[action*='captcha']")
            
            if captcha_present:
                logger.warning("Обнаружена капча! Ожидание ручного вмешательства...")
                # Ожидание ручного вмешательства
                page.wait_for_timeout(30000)  # Ждем 30 секунд
                return True
        except Exception as e:
            logger.error(f"Ошибка при обработке капчи: {e}")
        return False

    def extract_text(self, page, selector):
        # Извлекаем только текст, игнорируя изображения, код и т.д.
        try:
            elements = page.query_selector_all(selector)
            texts = [el.inner_text().strip() for el in elements if el.inner_text()]
            return "\n\n".join(texts)
        except Exception:
            return ""

    def crawl_post(self, page: Page, url: str):
        try:
            print(f"Парсинг поста: {url}")
            page.goto(url, timeout=60000)
            stealth_sync(page)
            time.sleep(random.uniform(1.5, 2.5))
            # Заголовок
            title = page.text_content("h1", timeout=20000)
            # Текст вопроса
            question_text = self.extract_text(page, ".question .js-post-body")
            # Текст первого ответа (если есть)
            answer_text = ""
            answer_cells = page.query_selector_all(".answer .js-post-body")
            if answer_cells:
                answer_text = answer_cells[0].inner_text().strip()
            # Объединяем текст вопроса и ответа
            content = f"ВОПРОС:\n{question_text}\n\nОТВЕТ:\n{answer_text}" if answer_text else f"ВОПРОС:\n{question_text}"
            # Теги для аннотации
            tags = page.eval_on_selector_all(
                ".post-tag",
                "els => els.map(e => e.textContent.trim())"
            )
            abstract = ", ".join(tags)
            # Автор
            author = None
            author_links = page.eval_on_selector_all(
                ".user-details a",
                "els => els.map(e => e.textContent.trim())"
            )
            if author_links:
                author = author_links[0]
            else:
                author = "Неизвестно"
            return {
                "title": title.strip() if title else "Без заголовка",
                "author": author.strip() if author else "Неизвестно",
                "abstract": abstract,
                "content": content.strip(),
                "tags": tags,
                "source_url": url,
                "created_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Ошибка при парсинге поста по адресу {url}: {e}")
            print(f"Ошибка при парсинге поста по адресу {url}: {e}")
            return None

    def scrape(self):
        print("Запуск парсинга сайтов StackExchange...")
        with sync_playwright() as pw:
            browser, context = self.get_browser_context(pw)
            page = context.new_page()
            try:
                for site_url in self.question_list_urls:
                    print(f"\nПарсинг сайта: {site_url}")
                    try:
                        page.goto(site_url, timeout=60000)
                        stealth_sync(page)
                        
                        # Проверка на капчу
                        if self.handle_captcha(page):
                            # Смена user-agent и повторная попытка
                            self.current_user_agent = random.choice(USER_AGENTS)
                            context.set_extra_http_headers({"User-Agent": self.current_user_agent})
                            page.goto(site_url, timeout=60000)
                            stealth_sync(page)
                        
                        time.sleep(random.uniform(2.0, 4.0))  # Увеличенная задержка
                        post_links = []
                        page_num = 1
                        while len(post_links) < self.post_limit:
                            # Собираем ссылки на посты с текущей страницы
                            links = page.eval_on_selector_all(
                                ".js-post-summary .s-post-summary--content-title a.s-link",
                                "els => els.map(e => e.href)"
                            )
                            for link in links:
                                if link not in post_links:
                                    post_links.append(link)
                                if len(post_links) >= self.post_limit:
                                    break
                            # Пытаемся перейти на следующую страницу, если нужно
                            if len(post_links) < self.post_limit:
                                next_btn = page.query_selector("a[rel='next']")
                                if next_btn:
                                    next_btn.click()
                                    time.sleep(random.uniform(1.5, 2.5))
                                    stealth_sync(page)
                                else:
                                    break
                            else:
                                break
                        print(f"Найдено {len(post_links)} ссылок на посты на этом сайте.")
                        db = session()
                        scraped = 0
                        idx = 0
                        while scraped < self.post_limit and idx < len(post_links):
                            post_url = post_links[idx]
                            exists = ArticleRepository.get_by_source_url(db, post_url)
                            if exists:
                                print(f"[ПРОПУСК] Уже существует: {post_url}")
                                idx += 1
                                continue
                            data = self.crawl_post(page, post_url)
                            if data:
                                self.save_direct(data, db=db)
                                print(f"[{scraped+1}/{self.post_limit}] Сохранено: {data['title']}")
                                self.saved_count += 1
                                scraped += 1
                                time.sleep(random.uniform(1.5, 3.5))
                            idx += 1
                        db.close()
                    except Exception as e:
                        logger.error(f"Ошибка при парсинге сайта {site_url}: {e}")
                        print(f"Ошибка при парсинге сайта {site_url}: {e}")
            except Exception as e:
                logger.error(f"Ошибка при парсинге сайтов StackExchange: {e}")
                print(f"Ошибка при парсинге сайтов StackExchange: {e}")
            finally:
                browser.close()
        print(f"Парсинг завершен. Всего сохранено постов: {self.saved_count}")

    def save_direct(self, data: dict, db=None):
        close_db = False
        if db is None:
            db = session()
            close_db = True
        try:
            dto = ArticleCreate(**data)
            ArticleService(db).create_article(dto)
        except Exception as e:
            logger.error(f"Ошибка при сохранении поста в БД: {e}")
            print(f"Ошибка при сохранении поста в БД: {e}")
        finally:
            if close_db:
                db.close()

def run_scraper(post_limit=10, question_list_urls=None):
    scraper = StackExchangeSiteScraper(post_limit=post_limit, question_list_urls=question_list_urls)
    scraper.scrape()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape questions and answers from StackExchange sites.")
    parser.add_argument("--limit", type=int, default=10, help="Number of posts to scrape per site")
    parser.add_argument("--sites", nargs="*", help="List of site URLs to scrape (from QUESTION_LIST_URLS)")
    args = parser.parse_args()
    if args.sites:
        # Only use the sites specified by the user, matching from QUESTION_LIST_URLS
        selected_sites = [url for url in QUESTION_LIST_URLS if any(site in url for site in args.sites)]
        if not selected_sites:
            print("No matching sites found for the given --sites argument. Scraping all sites.")
            selected_sites = QUESTION_LIST_URLS
    else:
        selected_sites = QUESTION_LIST_URLS
    print(f"Sites to scrape: {selected_sites}")
    run_scraper(post_limit=args.limit, question_list_urls=selected_sites)
