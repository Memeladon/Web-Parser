import re
import string
import unicodedata
from sqlalchemy.orm import Session
from src.database.dependencies import session
from src.services.article_service import ArticleService
from src.database.repositories.article_repository import ArticleRepository
from src.database.models import Article
import contractions
import inflect
import spacy
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка необходимых данных NLTK
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Инициализация компонентов NLP
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
p = inflect.engine()
nlp = spacy.load('en_core_web_sm')

def get_synonyms(word):
    """Получить синонимы для слова с помощью WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def normalize_word(word):
    """Нормализовать слово с помощью лемматизации spaCy."""
    doc = nlp(word)
    if doc:
        return doc[0].lemma_
    return word

def clean_math_content(text):
    """Очистка математических обозначений и содержимого LaTeX."""
    # Удаление разделителей LaTeX
    text = re.sub(r'\$.*?\$', ' ', text)  # Встроенная математика
    text = re.sub(r'\$\$.*?\$\$', ' ', text)  # Отдельная математика
    text = re.sub(r'\\[a-zA-Z]+{.*?}', ' ', text)  # Команды LaTeX
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)  # Команды LaTeX без фигурных скобок
    return text

def number_to_words(text):
    """Преобразовать числа в их словесное представление."""
    words = text.split()
    for i, word in enumerate(words):
        if word.isdigit():
            words[i] = p.number_to_words(word)
    return ' '.join(words)

def clean_abstract(text):
    if not text:
        return ""
    
    # Разделение по запятой или пробелу, удаление пробелов, дедупликация с сохранением порядка
    if "," in text:
        topics = [t.strip() for t in text.split(",") if t.strip()]
    else:
        topics = [t.strip() for t in text.split() if t.strip()]
    
    # Удаление дубликатов с сохранением порядка
    seen = set()
    deduped = []
    for topic in topics:
        if topic not in seen:
            seen.add(topic)
            deduped.append(topic)
    
    # Объединение обратно с тем же разделителем, который использовался
    if "," in text:
        return ",".join(deduped)
    else:
        return " ".join(deduped)

def clean_text(text):
    if not text:
        return ""
    
    try:
        # Сначала очистка математического содержимого
        text = clean_math_content(text)
        
        # Нормализация Unicode и удаление диакритических знаков
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        # Преобразование в нижний регистр
        text = text.lower()
        
        # Расширение сокращений с помощью библиотеки contractions
        text = contractions.fix(text)
        
        # Удаление HTML-тегов
        text = re.sub(r"<[^>]+>", " ", text)
        
        # Преобразование чисел в слова
        text = number_to_words(text)
        
        # Удаление пунктуации
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Нормализация пробелов
        text = re.sub(r"\s+", " ", text)
        
        # Разделение на слова
        words = text.split()
        
        # Обработка каждого слова
        processed_words = []
        for word in words:
            if word in STOPWORDS or word == 'join':  # Пропуск 'join' и стоп-слов
                continue
            
            # Стемминг слова
            stemmed = STEMMER.stem(word)
            if stemmed:
                processed_words.append(stemmed)
        
        # Удаление пустых токенов
        processed_words = [w for w in processed_words if w]
        
        return " ".join(processed_words)
    except Exception as e:
        logger.error(f"Ошибка при очистке текста: {e}")
        return text

def process_article(article, db):
    """Обработка одной статьи и сохранение изменений."""
    try:
        if getattr(article, 'cleaned', False):
            return False, "Уже очищено"
        orig_title = article.title
        orig_abstract = article.abstract
        orig_content = article.content
        article.title = clean_text(article.title)
        article.abstract = clean_abstract(article.abstract)
        article.content = clean_text(article.content)
        article.cleaned = True
        db.commit()
        return True, "Успешно"
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка при обработке статьи {article.id}: {e}")
        return False, str(e)

def clean_all_articles():
    db: Session = session()
    try:
        articles = db.query(Article).all()
        total = len(articles)
        logger.info(f"Загружено {total} статей из базы данных.")
        
        updated = 0
        skipped = 0
        failed = 0
        
        for i, article in enumerate(articles):
            success, message = process_article(article, db)
            
            if success:
                updated += 1
                if i < 3:  # Показать примеры для первых 3 статей
                    logger.info(f"\nПример {i+1} до/после:")
                    logger.info(f"Заголовок: {article.title}")
                    logger.info(f"Аннотация: {article.abstract}")
                    logger.info(f"Содержимое: {article.content[:100]}...")
            elif message == "Уже очищено":
                skipped += 1
            else:
                failed += 1
                logger.error(f"Не удалось обработать статью {article.id}: {message}")
            
            # Логирование прогресса каждые 10 статей
            if (i + 1) % 10 == 0:
                logger.info(f"Прогресс: {i + 1}/{total} статей обработано")
        
        logger.info(f"\nОбработка завершена:")
        logger.info(f"- Обновлено: {updated} статей")
        logger.info(f"- Пропущено: {skipped} статей")
        logger.info(f"- Ошибок: {failed} статей")
        
    except Exception as e:
        logger.error(f"Ошибка в clean_all_articles: {e}")
        db.rollback()
    finally:
        db.close()

def reset_cleaned_flag():
    """Сбросить флаг cleaned в False для всех статей."""
    db: Session = session()
    try:
        articles = db.query(Article).all()
        total = len(articles)
        logger.info(f"Сброс флага cleaned для {total} статей...")
        
        for article in articles:
            article.cleaned = False
        
        db.commit()
        logger.info(f"Успешно сброшен флаг cleaned для {total} статей")
    except Exception as e:
        logger.error(f"Ошибка при сбросе флага cleaned: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Очистка статей в базе данных или сброс флага cleaned",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Очистить все статьи
  python clean_articles.py
  
  # Сбросить флаг cleaned для всех статей
  python clean_articles.py -r
  
  # Сбросить и затем очистить все статьи
  python clean_articles.py -r && python clean_articles.py
"""
    )
    parser.add_argument("-r", "--reset", 
                       action="store_true",
                       help="Сбросить флаг cleaned в False для всех статей")
    args = parser.parse_args()
    
    if args.reset:
        reset_cleaned_flag()
    else:
        clean_all_articles() 