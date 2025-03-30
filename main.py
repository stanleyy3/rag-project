import requests
import json

# Define the corpus of food-related documents
corpus_of_documents = [
    "Enjoy a hearty breakfast of scrambled eggs with toast and fresh fruit.",
    "Savor a classic BLT sandwich with crispy bacon, lettuce, and tomato on toasted bread.",
    "Indulge in a creamy bowl of tomato soup paired with a perfectly grilled cheese sandwich.",
    "Delight in a fresh garden salad topped with grilled chicken and a zesty vinaigrette.",
    "Relish a plate of pasta primavera with seasonal vegetables in a light garlic sauce.",
    "Experience the flavors of a spicy chicken curry served with steamed jasmine rice.",
    "Feast on a juicy burger loaded with all the fixings and a side of crispy fries.",
    "Try a comforting bowl of beef stew with tender vegetables and a rich, savory broth.",
    "Enjoy a vibrant sushi platter featuring a variety of fresh fish and seasonal vegetables.",
    "Taste a hearty quinoa and black bean salad tossed in a tangy lime dressing."
]

# Compute Jaccard similarity between two strings
def jaccard_similarity(query, document):
    query_words = query.lower().split(" ")
    document_words = document.lower().split(" ")
    intersection = set(query_words).intersection(set(document_words))
    union = set(query_words).union(set(document_words))
    return len(intersection) / len(union)

# Return the document from the corpus most similar to the query
def return_response(query, corpus):
    similarities = []
    for doc in corpus:
        similarity = jaccard_similarity(query, doc)
        similarities.append(similarity)
    return corpus[similarities.index(max(similarities))]

# Get the user's input and retrieve the most relevant document
user_input = input("What is a food that you like?\n")
relevant_document = return_response(user_input, corpus_of_documents)

# Create the prompt for the LLM by injecting the relevant document and user input
prompt = f"""You are a bot that makes recommendations for food. You answer in concise, but thoughtful sentences and do not include extra information.
This is the relevant document: {relevant_document}
The user input is: {user_input}
Compile a recommendation to the user based on the relevant document and the user input.
"""

# AWANLLM API configuration
AWANLLM_API_KEY = "e6b88d40-29f5-4bfa-9677-171b55170890"
url = "https://api.awanllm.com/v1/chat/completions"

payload = json.dumps({
    "model": "Meta-Llama-3.1-8B-Instruct",
    "messages": [
        {"role": "system", "content": "you are a helpful assistant who gives thoughtful and correct answers."},
        {"role": "user", "content": prompt}
    ],
    "repetition_penalty": 1.1,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 1024,
    "stream": False  # Disable streaming for easier debugging.
})

headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {AWANLLM_API_KEY}"
}

# Send the API request
response = requests.request("POST", url, headers=headers, data=payload)

# Debugging: Print status code and raw response content
print("HTTP Status Code:", response.status_code)
raw_content = response.content.decode('utf-8')
print("Raw Response Content:", raw_content)

# Process the full JSON response and extract the recommendation text
try:
    full_json = response.json()
    if "choices" in full_json and len(full_json["choices"]) > 0:
        recommendation = full_json["choices"][0]["message"]["content"]
    else:
        recommendation = str(full_json)
    print("Final Recommendation:")
    print(recommendation)
except json.JSONDecodeError as e:
    print("Error decoding JSON:", e)
