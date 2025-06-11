import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from src.classifiers.article_classifier import ArticleClassifier
from src.classifiers.tag_analyzer import TagAnalyzer
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparison:
    def __init__(self, min_tag_frequency=2, top_n_classes=20):
        self.classifier = ArticleClassifier(top_n_tags=top_n_classes)
        self.tag_analyzer = TagAnalyzer(min_frequency=min_tag_frequency)
        self.results_dir = Path('classifier_results')
        self.results_dir.mkdir(exist_ok=True)
        self.top_n_classes = top_n_classes

    def train_and_compare(self, params=None):
        """Train models with different parameters and compare results."""
        if params is None:
            params = {
                'naive_bayes': {'alpha': [0.1, 1.0, 10.0]},
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20]
                }
            }
        
        # Analyze and group tags
        tag_groups = self.tag_analyzer.analyze_tags()
        tag_mapping = self.tag_analyzer.get_tag_mapping()
        
        # Prepare data with grouped tags
        data = self.classifier.prepare_data()
        train_texts, train_labels = data['train']
        val_texts, val_labels = data['val']
        test_texts, test_labels = data['test']
        
        # Group labels using tag mapping
        train_labels = [tag_mapping.get(label, label) for label in train_labels]
        val_labels = [tag_mapping.get(label, label) for label in val_labels]
        test_labels = [tag_mapping.get(label, label) for label in test_labels]
        
        # Train and evaluate models with different parameters
        results = {}
        
        for model_name, param_grid in params.items():
            results[model_name] = []
            
            if model_name == 'naive_bayes':
                for alpha in param_grid['alpha']:
                    self.classifier.naive_bayes = MultinomialNB(alpha=alpha)
                    result = self._evaluate_model(model_name, train_texts, train_labels, val_texts, val_labels)
                    results[model_name].append({
                        'params': {'alpha': alpha},
                        'metrics': result
                    })
            
            elif model_name == 'random_forest':
                for n_estimators in param_grid['n_estimators']:
                    for max_depth in param_grid['max_depth']:
                        self.classifier.random_forest = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                        result = self._evaluate_model(model_name, train_texts, train_labels, val_texts, val_labels)
                        results[model_name].append({
                            'params': {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth
                            },
                            'metrics': result
                        })
        
        # Visualize results
        self._plot_results(results)
        return results

    def _evaluate_model(self, model_name, train_texts, train_labels, val_texts, val_labels):
        """Evaluate a single model configuration."""
        # Transform text
        X_train_tfidf = self.classifier.tfidf.fit_transform(train_texts)
        X_val_tfidf = self.classifier.tfidf.transform(val_texts)
        
        # Apply feature selection
        X_train_selected = self.classifier.feature_selector.fit_transform(X_train_tfidf, train_labels)
        X_val_selected = self.classifier.feature_selector.transform(X_val_tfidf)
        
        # Train model
        self.classifier.models[model_name].fit(X_train_selected, train_labels)
        
        # Make predictions
        y_pred = self.classifier.models[model_name].predict(X_val_selected)
        
        # Get all unique classes
        all_classes = sorted(set(train_labels) | set(val_labels) | set(y_pred))
        
        # Calculate metrics with all known labels
        report = classification_report(val_labels, y_pred, output_dict=True, labels=all_classes)
        conf_matrix = confusion_matrix(val_labels, y_pred, labels=all_classes)
        logger.info(f"[INFO] Классы для {model_name}: {all_classes}")
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'accuracy': report['accuracy'],
            'classes': all_classes
        }

    def _plot_results(self, results):
        """Create visualizations of model comparison results for top N classes."""
        # Plot accuracy comparison
        plt.figure(figsize=(12, 6))
        for model_name, model_results in results.items():
            accuracies = [r['metrics']['accuracy'] for r in model_results]
            param_values = [str(r['params']) for r in model_results]
            plt.plot(param_values, accuracies, marker='o', label=model_name)
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Parameters')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / 'accuracy_comparison.png')
        plt.close()

        # Plot confusion matrices for best parameters (top N classes only)
        for model_name, model_results in results.items():
            best_result = max(model_results, key=lambda x: x['metrics']['accuracy'])
            conf_matrix = best_result['metrics']['confusion_matrix']
            class_labels = best_result['metrics']['classes']
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
            plt.title(f'Confusion Matrix - {model_name}\nBest Parameters: {best_result["params"]}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(self.results_dir / f'confusion_matrix_{model_name}_best.png')
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare and visualize classifier models")
    parser.add_argument("--min-tag-freq", type=int, default=2,
                      help="Minimum frequency for tags to be included")
    parser.add_argument("--top-n", type=int, default=20,
                      help="Number of top classes to visualize in confusion matrix")
    args = parser.parse_args()
    
    comparison = ModelComparison(min_tag_frequency=args.min_tag_freq, top_n_classes=args.top_n)
    results = comparison.train_and_compare()
    
    logger.info("\nBest results for each model:")
    for model_name, model_results in results.items():
        best_result = max(model_results, key=lambda x: x['metrics']['accuracy'])
        logger.info(f"\n{model_name}:")
        logger.info(f"Parameters: {best_result['params']}")
        logger.info(f"Accuracy: {best_result['metrics']['accuracy']:.3f}")

if __name__ == "__main__":
    main() 