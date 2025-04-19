#!/usr/bin/env python3
"""
Story Inconsistency Checker

This program processes a story (input as text or extracted from a PDF) and checks each word for logical
inconsistencies using an LLM API (awanllm). It generates character/scene documents from the story, ranks them
based on the context (last three sentences), and for each word, calls the LLM with the following prompts:

1. Check inconsistency:
   word:
   {word}

   last three sentences:
   {lastThreeSentences}

   relevant document:
   {relevantDocument}

2. Replacement prompt (if inconsistency is detected):
   we found that the indicated "word" (below) is not correct in the current context. find the best replacement word(s)
   based on the "relevant document" (below) and the "last three sentences" (below).

   word:
   {word}

   relevant document:
   {relevantDocument}

   last three sentences:
   {lastThreeSentences}

3. Update relevant document prompt:
   new word:
   {newWord}

   relevant document:
   {relevantDocument}

The program includes error handling and retry logic for the LLM API calls.
"""

import requests
import json
import re
import logging
from typing import List, Tuple, Dict

# AWANLLM API configuration (similar to the attached main.py)
AWANLLM_API_KEY = "e6b88d40-29f5-4bfa-9677-171b55170890"
AWANLLM_URL = "https://api.awanllm.com/v1/chat/completions"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Utility functions for text processing ---

def split_into_sentences(text: str) -> List[str]:
    """
    Splits the text into sentences using simple punctuation delimiters.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def get_last_three_sentences(sentences: List[str], current_index: int) -> str:
    """
    Returns a string containing the last three sentences before the sentence at current_index.
    """
    start = max(0, current_index - 3)
    context = sentences[start:current_index]
    return " ".join(context)

# --- Ranking functions ---
# Here we use Jaccard similarity as a placeholder for all four ranking functions.
def jaccard_similarity(query: str, document: str) -> float:
    """
    Computes the Jaccard similarity between two strings.
    """
    query_words = set(query.lower().split())
    document_words = set(document.lower().split())
    if not query_words or not document_words:
        return 0.0
    intersection = query_words.intersection(document_words)
    union = query_words.union(document_words)
    return len(intersection) / len(union)

def bm25_similarity(query: str, document: str) -> float:
    return jaccard_similarity(query, document)

def cross_encoder_similarity(query: str, document: str) -> float:
    return jaccard_similarity(query, document)

def colbert_similarity(query: str, document: str) -> float:
    return jaccard_similarity(query, document)

def combined_similarity(query: str, document: str) -> float:
    """
    Combines the scores of the four ranking functions using a simple average.
    """
    scores = [
        jaccard_similarity(query, document),
        bm25_similarity(query, document),
        cross_encoder_similarity(query, document),
        colbert_similarity(query, document)
    ]
    return sum(scores) / len(scores)

def rank_documents(context: str, documents: List[str]) -> List[str]:
    """
    Ranks the provided documents based on their combined similarity to the context.
    Returns the top 5 documents.
    """
    ranked = sorted(documents, key=lambda doc: combined_similarity(context, doc), reverse=True)
    return ranked[:5]

# --- LLM API call functions ---

def call_llm(prompt: str, retries: int = 3) -> str:
    """
    Calls the AWANLLM API with the given prompt and returns the response text.
    Implements error handling and retries.
    """
    payload = json.dumps({
        "model": "Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant who gives thoughtful and correct answers."},
            {"role": "user", "content": prompt}
        ],
        "repetition_penalty": 1.1,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 1024,
        "stream": False
    })

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {AWANLLM_API_KEY}"
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(AWANLLM_URL, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            full_json = response.json()
            if "choices" in full_json and len(full_json["choices"]) > 0:
                content = full_json["choices"][0]["message"]["content"].strip()
                logging.info("LLM call successful on attempt %d.", attempt)
                return content
            else:
                logging.error("LLM API response missing 'choices'. Response: %s", full_json)
        except (requests.RequestException, json.JSONDecodeError) as e:
            logging.error("Error on LLM call attempt %d: %s", attempt, e)
    raise RuntimeError("LLM API call failed after multiple attempts.")

def check_inconsistency(word: str, last_three_sentences: str, document: str) -> str:
    """
    Calls the LLM API to check if the indicated word is logically inconsistent with the provided context and document.
    The prompt follows exactly the pseudocode wording.
    """
    prompt = f"""word:
{word}

last three sentences:
{last_three_sentences}

relevant document:
{document}"""
    response = call_llm(prompt)
    logging.info("Check inconsistency response for '%s': %s", word, response)
    return response.lower()

def get_replacement_word(word: str, last_three_sentences: str, document: str) -> str:
    """
    Calls the LLM API to get the best replacement word(s) when a word is found to be inconsistent.
    The prompt follows exactly the pseudocode wording.
    """
    prompt = f"""we found that the indicated "word" (below) is not correct in the current context. find the best replacement word(s) based on the "relevant document" (below) and the "last three sentences" (below).

word:
{word}

relevant document:
{document}

last three sentences:
{last_three_sentences}"""
    replacement = call_llm(prompt)
    logging.info("Replacement for '%s': %s", word, replacement)
    return replacement.strip()

def update_relevant_document(new_word: str, document: str) -> str:
    """
    Calls the LLM API to update the relevant document based on the new word.
    The prompt follows exactly the pseudocode wording.
    """
    prompt = f"""new word:
{new_word}

relevant document:
{document}"""
    updated_document = call_llm(prompt)
    logging.info("Updated document after new word '%s'.", new_word)
    return updated_document.strip()

# --- Document (world) creation ---

def generate_character_documents(story: str) -> List[str]:
    """
    Generates a list of character/scene documents from the story.
    For simplicity, the story is split by newline characters.
    """
    documents = [doc.strip() for doc in story.split('\n') if doc.strip()]
    if not documents:
        documents = [story]
    return documents

# --- Main processing function ---

def process_story(story: str) -> Tuple[str, List[Dict]]:
    """
    Processes the given story, checking each word for logical inconsistencies.
    Returns the edited story and a list of changes.
    """
    sentences = split_into_sentences(story)
    documents = generate_character_documents(story)
    
    validated_sentences = []
    changes = []  # List of dictionaries with sentence and word change details

    for sent_idx, sentence in enumerate(sentences):
        words = sentence.split()  # Simple tokenization; production code may need more robust handling.
        validated_words = []
        context = get_last_three_sentences(sentences, sent_idx)
        top_docs = rank_documents(context, documents)
        
        for word_idx, word in enumerate(words):
            response = None
            # Try each of the top 5 documents until a definitive answer is received.
            for doc in top_docs:
                response = check_inconsistency(word, context, doc)
                if response in ["yes", "no"]:
                    break  # If a definitive answer ("yes" or "no") is returned, stop trying further docs.
            # If no definitive answer or if "no", the word is assumed correct.
            if response is None or response in ["dnr", "no"]:
                validated_words.append(word)
            elif response == "yes":
                # Get replacement word(s)
                replacement = get_replacement_word(word, context, doc)
                validated_words.append(replacement)
                changes.append({
                    'sentence_idx': sent_idx,
                    'word_idx': word_idx,
                    'original': word,
                    'replacement': replacement
                })
                # Update the relevant document based on the new word
                updated_doc = update_relevant_document(replacement, doc)
                doc_index = documents.index(doc)
                documents[doc_index] = updated_doc
            else:
                validated_words.append(word)
        validated_sentences.append(" ".join(validated_words))
    
    edited_story = " ".join(validated_sentences)
    return edited_story, changes

# --- Main entry point ---

def main():
    """
    Main function to run the Story Inconsistency Checker.
    """
    print("Welcome to the Story Inconsistency Checker!")
    print("You can paste your story below (or provide text extracted from a PDF):")
    story = input("Enter your story:\n")
    
    try:
        edited_story, changes = process_story(story)
        print("\nEdited Story:")
        print(edited_story)
        if changes:
            print("\nList of Changes (with sentence and word positions):")
            for change in changes:
                print(f"Sentence {change['sentence_idx'] + 1}, Word {change['word_idx'] + 1}: "
                      f"'{change['original']}' -> '{change['replacement']}'")
        else:
            print("\nNo inconsistencies were found in the story.")
    except Exception as e:
        logging.exception("An error occurred while processing the story: %s", e)
        print("An error occurred during processing. Please try again.")

if __name__ == "__main__":
    main()
