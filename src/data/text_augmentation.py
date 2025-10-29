"""
Data augmentation techniques for text to improve model generalization
"""

import random
import re
from typing import List, Optional
import nltk
try:
    from nltk.corpus import wordnet
except:
    pass

class TextAugmenter:
    """Text augmentation for better generalization"""

    def __init__(self, aug_prob: float = 0.1):
        """
        Args:
            aug_prob: Probability of applying augmentation to each word
        """
        self.aug_prob = aug_prob

    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """
        Replace n random words with their synonyms

        Args:
            text: Input text
            n: Number of words to replace

        Returns:
            Augmented text
        """
        words = text.split()

        # Skip if text too short
        if len(words) < 3:
            return text

        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalnum()]))
        random.shuffle(random_word_list)

        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)

            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1

            if num_replaced >= n:
                break

        return ' '.join(new_words)

    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert n synonyms into the sentence

        Args:
            text: Input text
            n: Number of words to insert

        Returns:
            Augmented text
        """
        words = text.split()

        # Skip if text too short
        if len(words) < 3:
            return text

        new_words = words.copy()

        for _ in range(n):
            self._add_word(new_words)

        return ' '.join(new_words)

    def random_swap(self, text: str, n: int = 2) -> str:
        """
        Randomly swap n pairs of words

        Args:
            text: Input text
            n: Number of swaps

        Returns:
            Augmented text
        """
        words = text.split()

        # Skip if text too short
        if len(words) < 3:
            return text

        new_words = words.copy()

        for _ in range(n):
            new_words = self._swap_word(new_words)

        return ' '.join(new_words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p

        Args:
            text: Input text
            p: Probability of deleting each word

        Returns:
            Augmented text
        """
        words = text.split()

        # Skip if text too short
        if len(words) < 3:
            return text

        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)

        # If all words deleted, return random word
        if len(new_words) == 0:
            return random.choice(words)

        return ' '.join(new_words)

    def augment(self, text: str, num_aug: int = 1, techniques: Optional[List[str]] = None) -> List[str]:
        """
        Apply multiple augmentation techniques

        Args:
            text: Input text
            num_aug: Number of augmented versions to generate
            techniques: List of techniques to use (default: all)
                       Options: ['synonym', 'insert', 'swap', 'delete']

        Returns:
            List of augmented texts
        """
        if techniques is None:
            techniques = ['synonym', 'insert', 'swap', 'delete']

        augmented_texts = []

        for _ in range(num_aug):
            # Randomly select a technique
            technique = random.choice(techniques)

            if technique == 'synonym':
                aug_text = self.synonym_replacement(text, n=2)
            elif technique == 'insert':
                aug_text = self.random_insertion(text, n=1)
            elif technique == 'swap':
                aug_text = self.random_swap(text, n=2)
            elif technique == 'delete':
                aug_text = self.random_deletion(text, p=0.1)
            else:
                aug_text = text

            # Avoid duplicates
            if aug_text != text and aug_text not in augmented_texts:
                augmented_texts.append(aug_text)

        return augmented_texts

    def _get_synonyms(self, word: str) -> set:
        """Get synonyms for a word using WordNet"""
        synonyms = set()

        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym)
        except:
            # WordNet not available
            pass

        return synonyms

    def _add_word(self, words: List[str]):
        """Add a random synonym to the word list"""
        random_word = random.choice([word for word in words if word.isalnum()])
        synonyms = self._get_synonyms(random_word)

        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            random_idx = random.randint(0, len(words) - 1)
            words.insert(random_idx, synonym)

    def _swap_word(self, words: List[str]) -> List[str]:
        """Swap two random words"""
        if len(words) < 2:
            return words

        new_words = words.copy()
        idx1, idx2 = random.sample(range(len(words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

        return new_words


class TwitterTextCleaner:
    """Advanced text cleaning for Twitter data"""

    @staticmethod
    def clean_text(text: str,
                   remove_urls: bool = True,
                   remove_mentions: bool = False,
                   remove_hashtags: bool = False,
                   remove_numbers: bool = False,
                   lowercase: bool = True,
                   remove_extra_spaces: bool = True) -> str:
        """
        Clean and normalize text

        Args:
            text: Input text
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_numbers: Remove numbers
            lowercase: Convert to lowercase
            remove_extra_spaces: Remove extra whitespace

        Returns:
            Cleaned text
        """
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions
        if remove_mentions:
            text = re.sub(r'@\w+', '', text)

        # Remove hashtags (keep the text part)
        if remove_hashtags:
            text = re.sub(r'#(\w+)', r'\1', text)

        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Remove extra spaces
        if remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text).strip()

        # Lowercase
        if lowercase:
            text = text.lower()

        return text

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text (repeated chars, etc)

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Reduce repeated characters (e.g., "loooove" -> "love")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text
