from collections import Counter
from sqlalchemy.orm import Session
from src.database.dependencies import session
from src.database.models import Article
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TagAnalyzer:
    def __init__(self, min_frequency=2, similarity_threshold=0.7):
        self.min_frequency = min_frequency
        self.similarity_threshold = similarity_threshold
        self.tag_groups = {}
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2)
        )

    def analyze_tags(self):
        """Анализ частоты тегов и группировка похожих тегов."""
        db: Session = session()
        try:
            articles = db.query(Article).filter(Article.cleaned == True).all()
            
            # Сбор всех тегов
            all_tags = []
            for article in articles:
                if article.abstract:
                    tags = article.abstract.split()
                    all_tags.extend(tags)
            
            # Подсчет частоты тегов
            tag_counts = Counter(all_tags)
            
            # Фильтрация редких тегов
            common_tags = {tag for tag, count in tag_counts.items() 
                         if count >= self.min_frequency}
            
            # Создание векторов тегов
            tag_texts = []
            for tag in common_tags:
                # Получение всех статей с этим тегом
                tag_articles = [a for a in articles 
                              if tag in a.abstract.split()]
                # Объединение их содержимого
                combined_text = " ".join(f"{a.title} {a.content}" 
                                       for a in tag_articles)
                tag_texts.append(combined_text)
            
            # Создание TF-IDF векторов для тегов
            tag_vectors = self.vectorizer.fit_transform(tag_texts)
            
            # Вычисление сходства между тегами
            similarity_matrix = cosine_similarity(tag_vectors)
            
            # Группировка похожих тегов
            self.tag_groups = {}
            processed_tags = set()
            
            for i, tag1 in enumerate(common_tags):
                if tag1 in processed_tags:
                    continue
                    
                similar_tags = []
                for j, tag2 in enumerate(common_tags):
                    if i != j and similarity_matrix[i, j] > 0.5:  # Порог сходства
                        similar_tags.append(tag2)
                        processed_tags.add(tag2)
                
                if similar_tags:
                    self.tag_groups[tag1] = similar_tags
                    processed_tags.add(tag1)
            
            # Вывод результатов анализа
            logger.debug("\nРезультаты анализа тегов:")
            logger.debug(f"Всего уникальных тегов: {len(tag_counts)}")
            logger.debug(f"Теги с частотой >= {self.min_frequency}: {len(common_tags)}")
            
            logger.debug("\nСамые частые теги:")
            for tag, count in tag_counts.most_common(10):
                logger.debug(f"{tag}: {count}")
            
            logger.debug("\nГруппы тегов (похожие теги):")
            for main_tag, similar_tags in self.tag_groups.items():
                logger.debug(f"{main_tag}: {', '.join(similar_tags)}")
            
            return self.tag_groups
            
        finally:
            db.close()

    def get_tag_mapping(self):
        """Получение отображения из оригинальных тегов в сгруппированные теги."""
        mapping = {}
        for main_tag, similar_tags in self.tag_groups.items():
            mapping[main_tag] = main_tag
            for tag in similar_tags:
                mapping[tag] = main_tag
        return mapping

    def get_tag_counts(self):
        """Возвращает Counter всех тегов из последнего анализа."""
        db = session()
        try:
            articles = db.query(Article).filter(Article.cleaned == True).all()
            all_tags = []
            for article in articles:
                if article.abstract:
                    tags = article.abstract.split()
                    all_tags.extend(tags)
            return Counter(all_tags)
        finally:
            db.close()

if __name__ == "__main__":
    analyzer = TagAnalyzer(min_frequency=2)
    analyzer.analyze_tags() 