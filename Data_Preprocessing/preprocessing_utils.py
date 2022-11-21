from nltk.tokenize import TweetTokenizer
import re

def tokenize(text: str) -> list[str]:
    tk = TweetTokenizer()
    tokenized = tk.tokenize(text)
    return tokenized

def replace_username(post: str) -> str:
    post = re.sub("@[A-Za-z0-9_]+", '<username>', post)
    return post

def replace_urls(post: str) -> str:
    regex = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
    post = re.sub(regex, "<url>", post)
    return post

def remove_punctuation(post: str) -> str:
    post = re.sub('([*]+)|([/]+)|([\]+)|([\)]+)|([\(]+)|([:]+)|([#]+)|([\.]+)|([,]+)|([-]+)|([!]+)|([\?])|([;]+)|[\']|[\"]', '', post)
    return post