from article_classifier import predict_category
import argparse

def main():
    parser = argparse.ArgumentParser(description="Predict article category")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    parser.add_argument("--model", type=str, default="naive_bayes", 
                      choices=["naive_bayes", "random_forest"],
                      help="Model to use for prediction")
    args = parser.parse_args()

    # Get prediction
    result = predict_category(args.text, model_name=args.model)
    
    print(f"\nPrediction using {args.model}:")
    print(f"Predicted category: {result['prediction']}")
    print("\nTop 5 most likely categories:")
    
    # Sort probabilities and show top 5
    sorted_probs = sorted(result['probabilities'].items(), 
                         key=lambda x: x[1], 
                         reverse=True)[:5]
    
    for category, prob in sorted_probs:
        print(f"{category}: {prob:.3f}")

if __name__ == "__main__":
    main() 

# # Using Naive Bayes
# python -m src.classifiers.predict --text "your text here" --model naive_bayes

# Using Random Forest
# python -m src.classifiers.predict --text "your text here" --model random_forest