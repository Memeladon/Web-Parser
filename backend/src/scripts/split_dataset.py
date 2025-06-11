import logging
from sqlalchemy.orm import Session
from src.database.dependencies import session
from src.database.models import Article, DatasetSplit
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict, Counter
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tag_distribution(articles):
    """Get distribution of tags across articles."""
    tag_counts = defaultdict(int)
    for article in articles:
        if article.abstract:
            for tag in article.abstract.split():
                tag_counts[tag] += 1
    return tag_counts

def get_top_n_tags(articles, top_n=30, min_freq=3):
    tag_counts = Counter()
    for article in articles:
        if article.abstract:
            tag_counts.update(article.abstract.split())
    most_common = [tag for tag, count in tag_counts.most_common(top_n) if count >= min_freq]
    return set(most_common)

def split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42, top_n=30, min_freq=3):
    """Split the dataset into train, validation, and test sets while maintaining tag distribution (oversampling by tags)."""
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("Ratios must be between 0 and 1")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Ratios must sum to 1.0")

    db = session()
    try:
        # Get all cleaned articles
        articles = db.query(Article).filter(Article.cleaned == True).all()
        logger.info(f"Found {len(articles)} articles")

        # Get top-N frequent tags
        frequent_tags = get_top_n_tags(articles, top_n=top_n, min_freq=min_freq)
        logger.info(f"Top {top_n} frequent tags (min freq {min_freq}): {sorted(frequent_tags)}")

        # Group articles by tags
        tag_to_articles = defaultdict(list)
        articles_without_frequent_tags = []
        
        for article in articles:
            if not article.abstract:
                articles_without_frequent_tags.append(article)
                continue
                
            tags = set(article.abstract.split())
            common = tags & frequent_tags
            
            if not common:
                articles_without_frequent_tags.append(article)
            else:
                for tag in common:
                    tag_to_articles[tag].append(article)

        logger.info(f"Grouped articles into {len(tag_to_articles)} tag groups")
        logger.info(f"Found {len(articles_without_frequent_tags)} articles without frequent tags")

        # Initialize sets for splits
        train_articles = set()
        val_articles = set()
        test_articles = set()

        # First, ensure each frequent tag has at least one article in train
        for tag, group in tag_to_articles.items():
            if not group:
                continue
            # Randomly select one article for training
            train_article = random.choice(group)
            train_articles.add(train_article)
            # Remove it from the group
            group.remove(train_article)
            if not group:
                continue
            # Split remaining articles
            n = len(group)
            if n == 1:
                val_articles.add(group[0])
            elif n == 2:
                val_articles.add(group[0])
                test_articles.add(group[1])
            else:
                n_val = max(1, int(n * val_ratio / (val_ratio + test_ratio)))
                n_test = n - n_val
                val, test = train_test_split(group, train_size=n_val, random_state=random_state)
                val_articles.update(val)
                test_articles.update(test)

        # Then, split remaining articles with frequent tags
        remaining_articles = []
        for group in tag_to_articles.values():
            remaining_articles.extend(group)
        
        if remaining_articles:
            n = len(remaining_articles)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            n_test = n - n_train - n_val
            
            if n_test < 1:
                n_train -= 1
                n_test = 1
            if n_val < 1:
                n_train -= 1
                n_val = 1
                
            train, temp = train_test_split(remaining_articles, train_size=n_train, random_state=random_state)
            val, test = train_test_split(temp, train_size=n_val, random_state=random_state)
            
            train_articles.update(train)
            val_articles.update(val)
            test_articles.update(test)

        # Finally, split articles without frequent tags
        if articles_without_frequent_tags:
            n = len(articles_without_frequent_tags)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            n_test = n - n_train - n_val
            
            if n_test < 1:
                n_train -= 1
                n_test = 1
            if n_val < 1:
                n_train -= 1
                n_val = 1
                
            train, temp = train_test_split(articles_without_frequent_tags, train_size=n_train, random_state=random_state)
            val, test = train_test_split(temp, train_size=n_val, random_state=random_state)
            
            train_articles.update(train)
            val_articles.update(val)
            test_articles.update(test)

        # Update database
        for article in articles:
            article.dataset_split = DatasetSplit.UNASSIGNED
        for article in train_articles:
            article.dataset_split = DatasetSplit.TRAIN
        for article in val_articles:
            article.dataset_split = DatasetSplit.VALIDATION
        for article in test_articles:
            article.dataset_split = DatasetSplit.TEST
        db.commit()

        # Log results
        logger.info(f"\nDataset split complete:")
        logger.info(f"Training set: {len(train_articles)} articles")
        logger.info(f"Validation set: {len(val_articles)} articles")
        logger.info(f"Test set: {len(test_articles)} articles")

        # Log tag distribution
        def get_tag_distribution(articles):
            tag_counts = Counter()
            for article in articles:
                if article.abstract:
                    tag_counts.update(article.abstract.split())
            return tag_counts

        train_tags = get_tag_distribution(train_articles)
        val_tags = get_tag_distribution(val_articles)
        test_tags = get_tag_distribution(test_articles)

        logger.info("\nTag distribution in splits:")
        logger.info("\nTraining set:")
        for tag in sorted(frequent_tags):
            count = train_tags.get(tag, 0)
            logger.info(f"  {tag}: {count}")
        logger.info("\nValidation set:")
        for tag in sorted(frequent_tags):
            count = val_tags.get(tag, 0)
            logger.info(f"  {tag}: {count}")
        logger.info("\nTest set:")
        for tag in sorted(frequent_tags):
            count = test_tags.get(tag, 0)
            logger.info(f"  {tag}: {count}")

    except Exception as e:
        db.rollback()
        logger.error(f"Error splitting dataset: {e}")
        raise
    finally:
        db.close()

def reset_splits():
    """Reset all articles to unassigned split."""
    db: Session = session()
    try:
        articles = db.query(Article).all()
        for article in articles:
            article.dataset_split = DatasetSplit.UNASSIGNED
        db.commit()
        logger.info(f"Reset dataset split for {len(articles)} articles")
    except Exception as e:
        logger.error(f"Error resetting splits: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset into train/validation/test sets")
    parser.add_argument("--reset", action="store_true", help="Reset all splits to unassigned")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Proportion of data for training")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Proportion of data for validation")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Proportion of data for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--top-n", type=int, default=30, help="Top N frequent tags")
    parser.add_argument("--min-freq", type=int, default=3, help="Minimum frequency for a tag to be considered frequent")
    args = parser.parse_args()
    
    if args.reset:
        reset_splits()
    else:
        split_dataset(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.seed,
            top_n=args.top_n,
            min_freq=args.min_freq
        ) 