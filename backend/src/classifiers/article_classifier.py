import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, multilabel_confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, chi2
from pathlib import Path
from sqlalchemy.orm import Session
from src.database.dependencies import session
from src.database.models import Article, DatasetSplit
import logging
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from src.classifiers.tag_analyzer import TagAnalyzer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import warnings
import pickle
from collections import Counter
from sklearn.linear_model import LogisticRegression
from numpy import linspace

# Фильтрация определенных предупреждений sklearn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.multiclass')

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ArticleClassifier:
    def __init__(self, min_tag_frequency=20, top_n_tags=15, multilabel=False):
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 1),
            min_df=2,  # Увеличиваем минимальную частоту документа
            max_df=0.95,  # Уменьшаем максимальную частоту документа
            stop_words=None,
            strip_accents='unicode',
            lowercase=True,
            analyzer='word',
            token_pattern=r'(?u)\b[\w.-]+\b'
        )
        self.naive_bayes = MultinomialNB(alpha=0.5)  # Уменьшаем alpha для более строгой классификации
        self.random_forest = RandomForestClassifier(
            n_estimators=200,  # Увеличиваем количество деревьев
            max_depth=8,  # Уменьшаем глубину для предотвращения переобучения
            min_samples_split=10,  # Увеличиваем минимальное количество образцов для разделения
            min_samples_leaf=4,  # Увеличиваем минимальное количество образцов в листе
            random_state=42,
            class_weight='balanced'
        )
        self.models = {
            'naive_bayes': self.naive_bayes,
            'random_forest': self.random_forest,
            'lr': OneVsRestClassifier(
                LogisticRegression(C=0.5, max_iter=1000, class_weight='balanced')  # Уменьшаем C для более строгой регуляризации
            )
        }
        self.model_dir = Path('classifier_models')
        self.model_dir.mkdir(exist_ok=True)
        self.multilabel = multilabel
        self.mlb = None
        self.top_n_tags = top_n_tags
        self.min_tag_frequency = min_tag_frequency
        
        # Инициализация анализатора тегов и получение частых тегов
        self.tag_analyzer = TagAnalyzer(min_frequency=min_tag_frequency)
        self.tag_analyzer.analyze_tags()
        self.tag_mapping = self.tag_analyzer.get_tag_mapping()
        
        # Получение количества тегов и их группировка с использованием маппинга
        tag_counts = self.tag_analyzer.get_tag_counts()
        grouped_counts = Counter()
        for tag, count in tag_counts.items():
            mapped_tag = self.tag_mapping.get(tag, tag)
            grouped_counts[mapped_tag] += count
        
        # Получение топ-N тегов из сгруппированных подсчетов
        most_common = [tag for tag, _ in grouped_counts.most_common(self.top_n_tags)]
        # Удаляем UNASSIGNED из частых тегов, если вдруг попал
        self.frequent_tags = set([tag for tag in most_common if tag != 'UNASSIGNED'])
        
        logger.info(f"\n[DATA] Топ {self.top_n_tags} тегов после группировки (мин. частота {self.min_tag_frequency}):")
        for tag in self.frequent_tags:
            logger.info(f"  {tag}: {grouped_counts[tag]}")
        
        self.feature_selector = SelectKBest(chi2, k='all')
        self.class_weights = None

    def _preprocess_text(self, text):
        """Предобработка текста для обеспечения его валидности для TF-IDF."""
        if not text:
            return ""
        # Удаление лишних пробелов
        text = ' '.join(text.split())
        # Убеждаемся, что текст не пустой после предобработки
        return text if text.strip() else ""

    def _process_split(self, articles):
        texts, labels = [], []
        kept_tags, filtered_tags = set(), set()

        for article in articles:
            # 1) Предобработка текста
            title = self._preprocess_text(article.title or "")
            content = self._preprocess_text(article.content or "")
            combined_text = f"{title} {content}".strip()
            if not combined_text:
                continue  # пропускаем статьи без текста

            # 2) Маппинг тегов
            original_tags = article.abstract.split() if article.abstract else []
            mapped = []
            for t in original_tags:
                mt = self.tag_mapping.get(t, t)
                if mt in self.frequent_tags:
                    mapped.append(mt)
                    kept_tags.add(mt)
                else:
                    filtered_tags.add(mt)
            mapped = list(dict.fromkeys(mapped))

            # 3) Если ничего не осталось — устанавливаем UNASSIGNED
            if not mapped:
                mapped = ['UNASSIGNED']

            # 4) Добавляем текст и метку(и) вместе
            texts.append(combined_text)
            if self.multilabel:
                labels.append(mapped)
            else:
                labels.append(mapped[0])

        # После сбора: удаляем примеры, где метка только UNASSIGNED
        if self.multilabel:
            filtered = [(t, l) for t, l in zip(texts, labels) if not (len(l) == 1 and l[0] == 'UNASSIGNED')]
            if filtered:
                texts, labels = zip(*filtered)
                texts, labels = list(texts), list(labels)
            else:
                texts, labels = [], []
        else:
            filtered = [(t, l) for t, l in zip(texts, labels) if l != 'UNASSIGNED']
            if filtered:
                texts, labels = zip(*filtered)
                texts, labels = list(texts), list(labels)
            else:
                texts, labels = [], []

        # Логирование распределения тегов после фильтрации
        if self.multilabel:
            filtered_tag_counts = Counter([tag for label in labels for tag in label])
        else:
            filtered_tag_counts = Counter(labels)
        
        logger.info(f"\n[DATA] Распределение тегов в {articles[0].dataset_split.value} (после фильтрации):")
        for tag in sorted(self.frequent_tags):
            count = filtered_tag_counts.get(tag, 0)
            if count > 0:
                logger.info(f"  {tag}: {count}")
        
        logger.info(f"\nОбработано {len(texts)} статей с {len(kept_tags)} сохраненными тегами и {len(filtered_tags)} отфильтрованными тегами")
        if texts:
            logger.info(f"Средняя длина текста: {sum(len(t) for t in texts) / len(texts):.1f} символов")
            logger.info(f"Минимальная длина текста: {min(len(t) for t in texts)} символов")
            logger.info(f"Максимальная длина текста: {max(len(t) for t in texts)} символов")
        
        return texts, labels, kept_tags, filtered_tags

    def process_articles(self, articles):
        """Обработка статей и подготовка обучающих данных."""
        logger.info("Обработка статей...")
        
        # Логирование всех уникальных значений dataset_split для отладки
        unique_splits = set(a.dataset_split.value for a in articles)
        logger.info(f"Найдены значения dataset_split в базе данных: {unique_splits}")
        
        # Разделение статей на обучающую/валидационную/тестовую выборки (без учета регистра)
        train_articles = [a for a in articles if a.dataset_split.value.upper() == 'TRAIN']
        val_articles = [a for a in articles if a.dataset_split.value.upper() == 'VALIDATION']
        test_articles = [a for a in articles if a.dataset_split.value.upper() == 'TEST']
        
        logger.info(f"Найдено {len(train_articles)} обучающих статей")
        logger.info(f"Найдено {len(val_articles)} валидационных статей")
        logger.info(f"Найдено {len(test_articles)} тестовых статей")
        
        if not train_articles:
            raise ValueError("Не найдено обучающих статей! Проверьте значения dataset_split в базе данных.")
        
        # Обработка каждой выборки
        train_texts, train_labels, train_kept_tags, train_filtered_tags = self._process_split(train_articles)
        val_texts, val_labels, val_kept_tags, val_filtered_tags = self._process_split(val_articles)
        test_texts, test_labels, test_kept_tags, test_filtered_tags = self._process_split(test_articles)
        
        logger.info(f"Обработанные тексты - Обучающая: {len(train_texts)}, Валидационная: {len(val_texts)}, Тестовая: {len(test_texts)}")
        
        if not train_texts:
            raise ValueError("Не найдено валидных обучающих текстов после предобработки")
        
        # Логирование примера обучающих текстов
        logger.info("\nПример обучающих текстов:")
        for i, text in enumerate(train_texts[:5]):
            logger.info(f"Текст {i}: {text[:200]}...")
        
        # Преобразование текстовых данных с помощью TF-IDF
        logger.info("\nПреобразование текстовых данных...")
        
        try:
            # Обучение TF-IDF только на обучающих данных
            X_train = self.tfidf.fit_transform(train_texts)
            logger.info(f"Размер словаря TF-IDF: {len(self.tfidf.vocabulary_)}")
            logger.info(f"Количество признаков: {X_train.shape[1]}")
            
            if X_train.shape[1] == 0:
                logger.error("Не найдено признаков после преобразования TF-IDF!")
                logger.error("Пример обучающих текстов:")
                for i, text in enumerate(train_texts[:5]):
                    logger.error(f"Текст {i}: {text[:200]}...")
                raise ValueError("Не найдено признаков после преобразования TF-IDF")
            
            # Логирование некоторых элементов словаря
            logger.info("\nПример элементов словаря:")
            vocab_items = list(self.tfidf.vocabulary_.items())[:10]
            for term, idx in vocab_items:
                logger.info(f"Термин: {term}, Индекс: {idx}")
            
            # Преобразование валидационных и тестовых данных с помощью того же векторизатора
            X_val = self.tfidf.transform(val_texts)
            X_test = self.tfidf.transform(test_texts)
            
            logger.info(f"\nРазмеры матриц признаков - Обучающая: {X_train.shape}, Валидационная: {X_val.shape}, Тестовая: {X_test.shape}")
            
        except Exception as e:
            logger.error(f"Ошибка при преобразовании TF-IDF: {str(e)}")
            raise
        
        # Преобразование меток в соответствующий формат
        if self.multilabel:
            self.mlb = MultiLabelBinarizer(classes=sorted(self.frequent_tags))
            y_train = self.mlb.fit_transform(train_labels)
            y_val = self.mlb.transform(val_labels)
            y_test = self.mlb.transform(test_labels)
            logger.info(f"\nМульти-классификация (multilabel)...")
            logger.info(f"Binary label matrix shapes: train={y_train.shape}, val={y_val.shape}, test={y_test.shape}")
        else:
            y_train = train_labels
            y_val = val_labels
            y_test = test_labels
            logger.info(f"\nSingle-label classes: {sorted(set(y_train))}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _log_tag_distribution(self, articles, split_name):
        """Log distribution of tags across articles in a split."""
        tag_counts = Counter()
        for article in articles:
            if article.abstract:
                tags = article.abstract.split()
                tag_counts.update(tags)
        
        logger.info(f"\n[DATA] Tag distribution in {split_name} (before filtering):")
        for tag, count in tag_counts.most_common():
            if tag in self.frequent_tags:
                logger.info(f"  {tag}: {count}")

    def prepare_data(self):
        """Load and prepare data from the database."""
        db: Session = session()
        try:
            logger.info("[DATA] Loading and preparing data from database...")
            train_articles = db.query(Article).filter(
                Article.cleaned == True,
                Article.dataset_split == DatasetSplit.TRAIN
            ).all()
            val_articles = db.query(Article).filter(
                Article.cleaned == True,
                Article.dataset_split == DatasetSplit.VALIDATION
            ).all()
            test_articles = db.query(Article).filter(
                Article.cleaned == True,
                Article.dataset_split == DatasetSplit.TEST
            ).all()
            
            logger.info(f"[DATA] Articles: train={len(train_articles)}, val={len(val_articles)}, test={len(test_articles)}")
            
            if not train_articles:
                logger.error("[DATA] No training articles found in database!")
                raise ValueError("No training articles found in the database")
            
            # Process each split
            train_texts, train_labels, train_kept_tags, train_filtered_tags = self._process_split(train_articles)
            val_texts, val_labels, val_kept_tags, val_filtered_tags = self._process_split(val_articles)
            test_texts, test_labels, test_kept_tags, test_filtered_tags = self._process_split(test_articles)
            
            logger.info(f"[DATA] After filtering: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")
            
            if self.multilabel:
                all_train_tags = set([tag for label in train_labels for tag in label])
            else:
                all_train_tags = set(train_labels)
            
            logger.info(f"[DATA] Classes in training: {sorted(all_train_tags)}")
            
            if len(train_texts) == 0:
                logger.error("[DATA] No training articles after filtering!")
                exit(1)
            
            return {
                'train': (train_texts, train_labels),
                'val': (val_texts, val_labels),
                'test': (test_texts, test_labels)
            }
        finally:
            db.close()

    def evaluate_model(self, model, X_test, y_test, model_name, custom_preds=None):
        logger.info(f"[EVAL] Оценка {model_name}...")
        if custom_preds is not None:
            y_pred = custom_preds
        else:
            y_pred = model.predict(X_test)
        if self.multilabel:
            classes = list(self.mlb.classes_)
            if 'UNASSIGNED' in classes:
                idx = classes.index('UNASSIGNED')
                y_test_filtered = np.delete(y_test, idx, axis=1)
                y_pred_filtered = np.delete(y_pred, idx, axis=1)
                classes_filtered = [c for c in classes if c != 'UNASSIGNED']
            else:
                y_test_filtered = y_test
                y_pred_filtered = y_pred
                classes_filtered = classes
            report = classification_report(y_test_filtered, y_pred_filtered, target_names=classes_filtered, zero_division=0, output_dict=True)
            logger.info(f"[EVAL] {model_name} Classification Report (без UNASSIGNED):\n" + classification_report(y_test_filtered, y_pred_filtered, target_names=classes_filtered, zero_division=0))
            logger.info(f"[EVAL] {model_name} TEST metrics (без UNASSIGNED): macro f1={report['macro avg']['f1-score']:.3f}, micro f1={report['micro avg']['f1-score']:.3f}, weighted f1={report['weighted avg']['f1-score']:.3f}")
            logger.info(f"[EVAL] {model_name} Классы: {classes_filtered}")
            mcm = multilabel_confusion_matrix(y_test, y_pred)
            labels_to_plot = [l for l in self.mlb.classes_ if l != 'UNASSIGNED']
            indices = [i for i, l in enumerate(self.mlb.classes_) if l != 'UNASSIGNED']
            n_labels = len(labels_to_plot)
            n_cols = 6 if n_labels > 6 else n_labels
            n_rows = (n_labels + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
            axes = axes.flatten() if n_labels > 1 else [axes]
            for plot_idx, idx in enumerate(indices):
                label = labels_to_plot[plot_idx]
                ax = axes[plot_idx]
                sns.heatmap(mcm[idx], annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_title(f'{label}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
            for j in range(len(indices), len(axes)):
                axes[j].axis('off')
            plt.tight_layout()
            plt.suptitle(f'Confusion Matrices - {model_name} (без UNASSIGNED)', y=1.02)
            plt.savefig(self.model_dir / f'confusion_matrix_{model_name}_all.png')
            plt.close()
            return {
                'classification_report': report,
                'multilabel_confusion_matrix': mcm
            }
        else:
            if hasattr(model, 'classes_'):
                class_labels = list(model.classes_)
            else:
                class_labels = sorted(list(set(list(y_test) + list(y_pred))))
            if 'UNASSIGNED' in class_labels:
                class_labels_filtered = [c for c in class_labels if c != 'UNASSIGNED']
                y_test_filtered = [y for y in y_test if y != 'UNASSIGNED']
                y_pred_filtered = [y for y, yt in zip(y_pred, y_test) if yt != 'UNASSIGNED']
            else:
                class_labels_filtered = class_labels
                y_test_filtered = y_test
                y_pred_filtered = y_pred
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_filtered, y_pred_filtered, average='weighted', zero_division=0
            )
            accuracy = np.mean(np.array(y_test_filtered) == np.array(y_pred_filtered))
            logger.info(f"[EVAL] {model_name} TEST Performance Metrics (без UNASSIGNED): accuracy={accuracy:.3f}, precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
            logger.info(f"[EVAL] {model_name} Классы: {class_labels_filtered}")
            cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=class_labels_filtered)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels_filtered, yticklabels=class_labels_filtered)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(self.model_dir / f'confusion_matrix_{model_name}.png')
            plt.close()
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'confusion_matrix': cm
            }

    def _find_best_thresholds_per_label(self, clf, X_val, y_val_bin):
        """Grid search for best threshold per label on validation set for multilabel."""
        val_probs = clf.predict_proba(X_val)
        best_thresh = {}
        for i, tag in enumerate(self.mlb.classes_):
            best_f1, best_t = 0, 0
            # Расширяем диапазон порогов и увеличиваем их количество
            for t in linspace(0.1, 0.8, 70):
                preds = (val_probs[:, i] >= t).astype(int)
                p, r, f, _ = precision_recall_fscore_support(
                    y_val_bin[:, i], preds, zero_division=0, average='binary'
                )
                # Добавляем вес для precision
                weighted_f1 = f * (0.7 * p + 0.3 * r)  # Отдаем предпочтение precision
                if weighted_f1 > best_f1:
                    best_f1, best_t = weighted_f1, t
            best_thresh[tag] = best_t
        return best_thresh

    def train(self):
        logger.info("\n========== [TRAINING] ==========")
        data = self.prepare_data()
        train_texts, train_labels = data['train']
        val_texts, val_labels = data['val']
        test_texts, test_labels = data['test']
        
        logger.info(f"[TRAINING] Размеры выборок: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")
        logger.info("[TRAINING] TF-IDF векторизация...")
        
        X_train_tfidf = self.tfidf.fit_transform(train_texts)
        X_val_tfidf = self.tfidf.transform(val_texts)
        X_test_tfidf = self.tfidf.transform(test_texts)
        
        logger.info(f"[TRAINING] TF-IDF vocabulary size: {len(self.tfidf.vocabulary_)}")
        logger.info(f"[TRAINING] Feature matrix shapes: train={X_train_tfidf.shape}, val={X_val_tfidf.shape}, test={X_test_tfidf.shape}")
        
        # Усиливаем feature selection
        n_features = X_train_tfidf.shape[1]
        k = min(500, n_features)  # Уменьшаем количество признаков
        self.feature_selector = SelectKBest(chi2, k=k)
        logger.info(f"[TRAINING] Feature selection: {k} из {n_features}")

        if self.multilabel:
            mlb_classes = sorted([t for t in self.frequent_tags if t != 'UNASSIGNED'])
            self.mlb = MultiLabelBinarizer(classes=mlb_classes)
            y_train_bin = self.mlb.fit_transform(train_labels)
            y_val_bin = self.mlb.transform(val_labels)
            y_test_bin = self.mlb.transform(test_labels)
            
            # Добавляем балансировку классов
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_bin),
                y=y_train_bin.ravel()
            )
            
            X_train_selected = self.feature_selector.fit_transform(X_train_tfidf, y_train_bin)
            X_val_selected = self.feature_selector.transform(X_val_tfidf)
            X_test_selected = self.feature_selector.transform(X_test_tfidf)
            
            results = {}
            for name, model in self.models.items():
                logger.info(f"[TRAINING] Обучение {name}...")
                clf = OneVsRestClassifier(model)
                clf.fit(X_train_selected, y_train_bin)
                
                logger.info(f"[TRAINING] Подбор порогов по валидации (по каждому классу)...")
                best_thresh = self._find_best_thresholds_per_label(clf, X_val_selected, y_val_bin)
                logger.info(f"[TRAINING] Лучшие thresholds по классам: {best_thresh}")
                
                with open(self.model_dir / f'thresholds_{name}_multilabel.pkl', 'wb') as f:
                    pickle.dump(best_thresh, f)
                
                # Оценка на валидации
                val_probs = clf.predict_proba(X_val_selected)
                val_preds = np.zeros_like(val_probs, dtype=int)
                for i, tag in enumerate(self.mlb.classes_):
                    val_preds[:, i] = (val_probs[:, i] >= best_thresh.get(tag, 0.3)).astype(int)  # Увеличиваем базовый порог
                
                val_metrics = self.evaluate_model(clf, X_val_selected, y_val_bin, f"{name}_validation", custom_preds=val_preds)
                
                # Оценка на тесте
                test_probs = clf.predict_proba(X_test_selected)
                test_preds = np.zeros_like(test_probs, dtype=int)
                for i, tag in enumerate(self.mlb.classes_):
                    test_preds[:, i] = (test_probs[:, i] >= best_thresh.get(tag, 0.3)).astype(int)
                
                test_metrics = self.evaluate_model(clf, X_test_selected, y_test_bin, f"{name}_test", custom_preds=test_preds)
                
                logger.info(f"[TRAINING] Сохраняю модель {name}...")
                joblib.dump(clf, self.model_dir / f'{name}_multilabel.joblib')
                
                results[name] = {
                    'model': clf,
                    'validation_metrics': val_metrics,
                    'test_metrics': test_metrics
                }
            
            joblib.dump(self.tfidf, self.model_dir / 'tfidf_vectorizer.joblib')
            joblib.dump(self.mlb, self.model_dir / 'mlb.joblib')
            joblib.dump(self.feature_selector, self.model_dir / 'feature_selector.joblib')
            
        else:
            for name, model in self.models.items():
                logger.info(f"[TRAINING] Обучение {name}...")
                model.fit(X_train_selected, train_labels)
                logger.info(f"[TRAINING] Оценка {name} на валидации...")
                val_metrics = self.evaluate_model(model, X_val_selected, val_labels, f"{name}_validation")
                logger.info(f"[TRAINING] Оценка {name} на тесте...")
                test_metrics = self.evaluate_model(model, X_test_selected, test_labels, f"{name}_test")
                logger.info(f"[TRAINING] Сохраняю модель {name}...")
                self.save_model(name)
                results[name] = {
                    'model': model,
                    'validation_metrics': val_metrics,
                    'test_metrics': test_metrics
                }
            joblib.dump(self.tfidf, self.model_dir / 'tfidf_vectorizer.joblib')
            joblib.dump(self.feature_selector, self.model_dir / 'feature_selector.joblib')
        
        logger.info(f"[TRAINING] Классы, которые может предсказывать классификатор: {self.get_classes('random_forest')}")
        logger.info(f"[TRAINING] Классы, которые может предсказывать классификатор: {self.get_classes('naive_bayes')}")
        logger.info("========== [TRAINING END] ==========")
        return results

    def save_model(self, model_name):
        """Save a trained model to disk."""
        model_path = self.model_dir / f'{model_name}.joblib'
        joblib.dump(self.models[model_name], model_path)
        logger.info(f"Saved {model_name} model to {model_path}")

    def load_model(self, model_name, multilabel=False):
        """Load a trained model from disk."""
        if multilabel:
            model_path = self.model_dir / f'{model_name}_multilabel.joblib'
            self.mlb = joblib.load(self.model_dir / 'mlb.joblib')
            self.feature_selector = joblib.load(self.model_dir / 'feature_selector.joblib')
        else:
            model_path = self.model_dir / f'{model_name}.joblib'
            self.feature_selector = joblib.load(self.model_dir / 'feature_selector.joblib')
        if model_path.exists():
            self.models[model_name] = joblib.load(model_path)
            self.tfidf = joblib.load(self.model_dir / 'tfidf_vectorizer.joblib')
            logger.info(f"Loaded {model_name} model from {model_path}")
        else:
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")

    def predict(self, text, model_name='naive_bayes', multilabel=False):
        """Predict the category of a new text."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        text_tfidf = self.tfidf.transform([text])
        text_selected = self.feature_selector.transform(text_tfidf)
        
        if multilabel:
            # Ансамблирование моделей
            all_preds = []
            all_probs = []
            
            for name in self.models:
                clf = joblib.load(self.model_dir / f'{name}_multilabel.joblib')
                probs = clf.predict_proba(text_selected)[0]
                all_probs.append(probs)
                
                try:
                    with open(self.model_dir / f'thresholds_{name}_multilabel.pkl', 'rb') as f:
                        thresholds = pickle.load(f)
                except Exception:
                    thresholds = {cls: 0.3 for cls in self.mlb.classes_}
                
                preds = [cls for cls, prob in zip(self.mlb.classes_, probs) 
                        if prob >= thresholds.get(cls, 0.3)]
                all_preds.append(set(preds))
            
            # Усредняем вероятности
            avg_probs = np.mean(all_probs, axis=0)
            
            # Принимаем тег только если он предсказан большинством моделей
            final_preds = set.intersection(*all_preds)
            
            # Добавляем теги с высокой средней вероятностью
            for cls, prob in zip(self.mlb.classes_, avg_probs):
                if prob >= 0.4 and cls not in final_preds:  # Увеличиваем порог для добавления
                    final_preds.add(cls)
            
            # Гибридные правила
            text_lower = text.lower()
            if 'star wars' in text_lower:
                if 'star-wars' in self.mlb.classes_:
                    final_preds.add('star-wars')
                if 'the-force' in self.mlb.classes_:
                    final_preds.add('the-force')
            if 'the one ring' in text_lower or 'ring of power' in text_lower:
                if 'the-one-ring' in self.mlb.classes_:
                    final_preds.add('the-one-ring')
            
            final_preds = list(final_preds)
            prob_dict = {label: float(prob) for label, prob in zip(self.mlb.classes_, avg_probs)}
            
            return {'predicted_tags': final_preds, 'probabilities': prob_dict}
        else:
            # Make prediction
            prediction = self.models[model_name].predict(text_selected)
            probabilities = self.models[model_name].predict_proba(text_selected)[0]
            # Получаем имена классов
            class_labels = self.get_classes(model_name)
            prob_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
            return {
                'prediction': prediction[0],
                'probabilities': prob_dict
            }

    def get_classes(self, model_name='random_forest'):
        """Вернуть список всех классов, на которые обучен классификатор."""
        model = self.models.get(model_name)
        if hasattr(model, 'classes_'):
            return list(model.classes_)
        elif hasattr(self, 'mlb') and self.mlb:
            return list(self.mlb.classes_)
        elif hasattr(self, 'frequent_tags'):
            return list(self.frequent_tags)
        else:
            return []

def train_classifiers(multilabel=False, min_tag_frequency=10, top_n_tags=20):
    """Train all classifiers and save them."""
    classifier = ArticleClassifier(multilabel=multilabel, min_tag_frequency=min_tag_frequency, top_n_tags=top_n_tags)
    # Get articles from database
    db: Session = session()
    try:
        articles = db.query(Article).filter(Article.cleaned == True).all()
        if not articles:
            raise ValueError("No cleaned articles found in the database")
        return classifier.train()
    finally:
        db.close()

def predict_category(text, model_name='naive_bayes', multilabel=False):
    """Predict the category of a new text using a specific model."""
    classifier = ArticleClassifier(multilabel=multilabel)
    classifier.load_model(model_name, multilabel=multilabel)
    return classifier.predict(text, model_name, multilabel=multilabel)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train classifiers (single or multi-label)")
    parser.add_argument('--multilabel', action='store_true', help='Enable multi-label classification')
    parser.add_argument('--min_tag_frequency', type=int, default=10, help='Minimum frequency for tags to be included')
    parser.add_argument('--top_n_tags', type=int, default=20, help='Number of top tags to use as labels')
    args = parser.parse_args()
    train_classifiers(multilabel=args.multilabel, min_tag_frequency=args.min_tag_frequency, top_n_tags=args.top_n_tags) 