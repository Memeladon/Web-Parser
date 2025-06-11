import sys
from src.classifiers.article_classifier import ArticleClassifier

def print_prediction(model_name, classifier, text):
    classifier.load_model(model_name)
    classes = classifier.get_classes(model_name)
    result = classifier.predict(text, model_name=model_name)
    print(f"\n=== {model_name.upper()} ===")
    print(f"Предсказанный класс: {result['prediction']}")
    print("Топ-5 вероятных классов:")
    # Получаем вероятности по классам
    probs = result['probabilities']
    # Если ключи - индексы, преобразуем к названиям классов
    if all(isinstance(k, (int, str)) and str(k).isdigit() for k in probs.keys()):
        # Сопоставляем индексы с названиями классов
        class_map = {str(i): label for i, label in enumerate(classes)}
        probs_named = {class_map.get(str(k), str(k)): v for k, v in probs.items()}
    else:
        probs_named = probs
    top5 = sorted(probs_named.items(), key=lambda x: x[1], reverse=True)[:5]
    for label, prob in top5:
        print(f"{label}: {prob:.3f}")

def main():
    classifier = ArticleClassifier()
    test_texts = [
        "QUESTION: What is the Force in Star Wars and how does it work? ANSWER: The Force is a mystical energy field in the Star Wars universe that gives Jedi and Sith their power.",
        "QUESTION: What is the ring of power in Tolkien's legendarium? ANSWER: The One Ring is a central plot element in The Lord of the Rings.",
        "QUESTION: How does matrix multiplication work in mathematics? ANSWER: Matrix multiplication is a binary operation that produces a matrix from two matrices.",
        "QUESTION: What is the main plot of Game of Thrones? ANSWER: Game of Thrones is a fantasy series involving the struggle for the Iron Throne.",
        "QUESTION: What is the role of Markov chains in probability theory? ANSWER: Markov chains are mathematical systems that undergo transitions from one state to another."
    ]
    for text in test_texts:
        print(f'\nТестовый текст для предсказания:\n{text}\n')
        for model_name in ['random_forest', 'naive_bayes', 'lr']:
            print_prediction(model_name, classifier, text)

if __name__ == '__main__':
    main() 