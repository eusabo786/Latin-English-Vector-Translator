import numpy as np
import random

# =========================
# 1. Word map (x, y)
# =========================
embeddings = {
    "amare": np.array([1.5, 1.6]),
    "love": np.array([1.6, 1.7]),

    "videre": np.array([2.0, 2.0]),
    "see": np.array([2.1, 2.1]),

    "odium": np.array([-1.5, -1.6]),
    "hate": np.array([-1.6, -1.7]),

    "pax": np.array([1.0, 1.0]),
    "peace": np.array([1.1, 1.2]),

    "inordinatio": np.array([-1.2, -1.3]),
    "disorder": np.array([-1.3, -1.4]),

    "ordo": np.array([1.2, 1.3]),
    "order": np.array([1.3, 1.4]),

    "bellum": np.array([-1.0, -1.0]),
    "war": np.array([-1.1, -1.2]),

    "puella": np.array([0.0, 2.0]),
    "girl": np.array([0.1, 2.1]),

    "currere": np.array([3.0, 0.0]),
    "run": np.array([3.1, 0.1]),

    "ingredior": np.array([-3.0, 0.0]),
    "walk": np.array([-3.1, -0.1]),
}

english_words = ["love", "see", "war", "girl", "run", "walk", "hate", "peace", "order", "disorder"]

# =========================
# 2. Similarity
# =========================
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return np.dot(a, b) / (norm_a * norm_b)

# =========================
# 3. Translator
# =========================
def translate(word, top_n=5):
    if word not in embeddings:
        return None

    word_vector = embeddings[word]
    results = []

    for other_word in english_words:
        if other_word not in embeddings:
            continue

        similarity = cosine_similarity(word_vector, embeddings[other_word])
        results.append((other_word, similarity))

    random.shuffle(results)
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_n]

# =========================
# 4. User input
# =========================
user_word = input("Enter a Latin word: ").lower()

result = translate(user_word)

if result is None:
    print("Word not found in dictionary.")
else:
    print(f"\nTop translations for '{user_word}':")
    for i, (word, score) in enumerate(result, start=1):
        print(f"{i}° {word} -> similarity: {score:.3f}")
