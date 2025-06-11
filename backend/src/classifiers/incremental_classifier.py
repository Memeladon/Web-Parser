import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from pathlib import Path
import logging
from src.database.dependencies import session
from src.database.models import Article

logger = logging.getLogger(__name__)

class IncrementalArticleClassifier:
    def __init__(self, model_dir='classifier_models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Инициализация компонентов
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        self.model = SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=1e-3,
            random_state=42,
            max_iter=1000,
            tol=1e-3
        )
        self.mlb = MultiLabelBinarizer()
        self.classes_ = None
        
    def load_existing_model(self):
        """Загрузка существующей модели и компонентов, если они существуют."""
        try:
            self.tfidf = joblib.load(self.model_dir / 'tfidf_vectorizer.joblib')
            self.model = joblib.load(self.model_dir / 'incremental_model.joblib')
            self.mlb = joblib.load(self.model_dir / 'mlb.joblib')
            self.classes_ = self.mlb.classes_
            logger.info("Загружена существующая модель и компоненты")
            return True
        except FileNotFoundError:
            logger.info("Существующая модель не найдена. Будет обучена новая модель.")
            return False
            
    def partial_fit(self, new_texts, new_labels, multilabel=False):
        """Обновление модели новыми данными."""
        if self.classes_ is None:
            # Инициализируем классами из новых меток
            all_labels = []
            for tags in new_labels:
                if isinstance(tags, str):
                    tags = [tags]
                all_labels.extend(tags)
            self.classes_ = np.unique(all_labels)
        
        # Преобразование текста
        new_texts_tfidf = self.tfidf.transform(new_texts)
        if multilabel:
            # Оставляем только частые теги для multilabel
            filtered_labels = []
            for tags in new_labels:
                if isinstance(tags, str):
                    tags = [tags]
                filtered = [t for t in tags if t in self.mlb.classes_]
                filtered_labels.append(filtered)
            new_labels_bin = self.mlb.transform(filtered_labels)
            self.model.partial_fit(new_texts_tfidf, new_labels_bin, classes=self.mlb.classes_)
        else:
            # Single-label: берём только первый частый тег
            filtered_labels = []
            for tags in new_labels:
                if isinstance(tags, str):
                    tags = [tags]
                filtered = [t for t in tags if t in self.classes_]
                if filtered:
                    filtered_labels.append(filtered[0])
                else:
                    filtered_labels.append(self.classes_[0])  # запасной вариант
            self.model.partial_fit(new_texts_tfidf, filtered_labels, classes=self.classes_)
            
        # Сохранение обновленной модели
        self.save_model()
        
    def predict(self, texts):
        """Предсказание для новых текстов."""
        texts_tfidf = self.tfidf.transform(texts)
        predictions = self.model.predict(texts_tfidf)
        return self.mlb.inverse_transform(predictions)
        
    def save_model(self):
        """Сохранение модели и компонентов."""
        joblib.dump(self.model, self.model_dir / 'incremental_model.joblib')
        joblib.dump(self.tfidf, self.model_dir / 'tfidf_vectorizer.joblib')
        joblib.dump(self.mlb, self.model_dir / 'mlb.joblib')
        logger.info("Сохранены модель и компоненты")
        
    def get_latest_articles(self, limit=100):
        """Получение последних статей из базы данных для инкрементального обучения."""
        db = session()
        try:
            articles = db.query(Article).filter(
                Article.cleaned == True
            ).order_by(
                Article.created_at.desc()
            ).limit(limit).all()
            
            texts = []
            labels = []
            for article in articles:
                if article.abstract:
                    texts.append(f"{article.title} {article.abstract} {article.content}")
                    tags = article.abstract.split()
                    if tags:
                        labels.append(tags[0])
            return texts, labels
        finally:
            db.close()

def update_model_incrementally(limit=100):
    """Обновление модели последними статьями."""
    classifier = IncrementalArticleClassifier()
    classifier.load_existing_model()
    
    # Получение последних статей
    texts, labels = classifier.get_latest_articles(limit)
    if texts:
        logger.info(f"Обновление модели {len(texts)} новыми статьями")
        classifier.partial_fit(texts, labels)
    else:
        logger.info("Нет новых статей для обновления модели")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Обновление классификатора инкрементально с новыми данными")
    parser.add_argument("--limit", type=int, default=100,
                      help="Количество последних статей для обновления")
    args = parser.parse_args()
    update_model_incrementally(args.limit) 