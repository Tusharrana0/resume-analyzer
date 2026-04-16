import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    words = [word for word in tokens if word.isalnum()]
    return " ".join(words)

def read_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

resume = read_file("sample_resume.txt")
job_desc = read_file("job_description.txt")

resume_clean = preprocess(resume)
job_clean = preprocess(job_desc)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume_clean, job_clean])

similarity = cosine_similarity(vectors[0:1], vectors[1:2])

print(f"Resume Match Score: {round(similarity[0][0] * 100, 2)}%")

if similarity[0][0] < 0.5:
    print("Suggestion: Add more relevant skills and keywords.")
else:
    print("Good match! Your resume is aligned.")
