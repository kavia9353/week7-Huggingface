import pandas as pd
from transformers import pipeline

# 1. Sentiment Analysis (Lightweight DistilBERT)
print("Loading Sentiment Analysis pipeline...")
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I really enjoyed learning about artificial intelligence.",
    "The new technology is very useful and easy to understand.",
    "I am unhappy with the poor performance of this software.",
    "Hugging Face makes natural language processing accessible.",
    "The product quality is excellent and I am very satisfied."
]

sentiment_results = sentiment_analyzer(texts)
sentiment_df = pd.DataFrame({
    "Text": texts,
    "Sentiment": [res["label"] for res in sentiment_results],
    "Score": [res["score"] for res in sentiment_results]
})
print("Sentiment Analysis completed successfully!\n")
print(sentiment_df)

# Save sentiment results
sentiment_df.to_csv("dataset/sentiment_results.csv", index=False)


# 2. Summarization (Lightweight T5-Small ~240MB instead of 1.22GB)
print("\nLoading Summarization pipeline...")
summarizer = pipeline(
    "summarization", 
    model="t5-small"
)

long_text = """
Hugging Face is an open-source platform that provides tools for machine learning and artificial intelligence applications. 
It is best known for its Transformers library, which simplifies the implementation of state-of-the-art natural language processing models. 
With Hugging Face pipelines, developers and researchers can quickly perform tasks such as text classification, summarization, translation, and question answering without needing to manage complex model architectures manually.
"""

summary = summarizer(long_text, max_length=50, min_length=25, do_sample=False)
summary_text = summary[0]["summary_text"]
print("Summarization completed successfully!\n")
print("Summary:", summary_text)

# Save summary result
summary_df = pd.DataFrame([{"Original_Text": long_text, "Summary": summary_text}])
summary_df.to_csv("dataset/summarized_text.csv", index=False)


# 3. Translation (English to French using T5-Small ~240MB)
print("\nLoading Translation pipeline...")
translator = pipeline(
    "translation_en_to_fr", 
    model="t5-small"
)

translation_result = translator("Hugging Face makes machine learning accessible to everyone.")
translated_text = translation_result[0]["translation_text"]
print("Translation completed successfully!\n")
print("Translated Text:", translated_text)

# Save translation result
translation_df = pd.DataFrame([{"English": "Hugging Face makes machine learning accessible to everyone.", "French": translated_text}])
translation_df.to_csv("dataset/translated_text.csv", index=False)


# 4. Question Answering (Lightweight DistilBERT QA ~260MB)
print("\nLoading Question Answering pipeline...")
qa_pipeline = pipeline(
    "question-answering", 
    model="distilbert/distilbert-base-cased-distilled-squad"
)

context = "Hugging Face was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf."
question = "Who founded Hugging Face?"

qa_result = qa_pipeline(question=question, context=context)
print("Question Answering completed successfully!\n")
print("Answer:", qa_result["answer"])

# Save QA result
qa_df = pd.DataFrame([{"Context": context, "Question": question, "Answer": qa_result["answer"], "Score": qa_result["score"]}])
qa_df.to_csv("dataset/qa_results.csv", index=False)

print("\nAll pipeline tasks executed and results saved successfully in the dataset/ directory!") 
summary_df = pd.DataFrame({
    "Original Text": [long_text],
    "Summary": [summary]
})

summary_df.to_csv(
    "dataset/summarized_text.csv",
    index=False
)

print("\nSummary saved successfully!")
# Text Summarization
summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

long_text = """
Artificial intelligence is becoming an important technology in many areas.
It is used in education, healthcare, business, transportation and communication.
Machine learning allows computers to learn from data and improve their performance.
Natural language processing helps computers understand and process human language.
Hugging Face provides many pre-trained transformer models that can be used for
different NLP tasks such as sentiment analysis, translation, summarization and
question answering. These tools make it easier for students and developers to
build useful artificial intelligence applications.
"""

summary = summarizer(
    "Summarize this text: " + long_text,
    max_new_tokens=60
)[0]["generated_text"]

print("\nSummary:")
print(summary)
# Translation
translator = pipeline(
    "translation_en_to_fr",
    model="Helsinki-NLP/opus-mt-en-fr"
)

english_text = "Artificial intelligence is changing the way people learn and work."

translation = translator(english_text)[0]["translation_text"]

print("\nEnglish:")
print(english_text)

print("\nFrench Translation:")
print(translation)

pd.DataFrame({
    "Original Text": [english_text],
    "Translation": [translation],
    "Language": ["French"]
}).to_csv(
    "dataset/translated_text.csv",
    index=False
)

print("\nTranslation saved successfully!")
# ---------------- QUESTION ANSWERING ----------------

qa = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

context = """
Artificial intelligence is a technology that enables computers to perform
tasks that normally require human intelligence. Machine learning is a branch
of artificial intelligence that allows computers to learn from data.
Natural language processing helps computers understand human language.
Hugging Face provides pre-trained transformer models for many NLP tasks.
"""

questions = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What does natural language processing help computers understand?",
    "What does Hugging Face provide?",
    "What is machine learning a branch of?"
]

qa_results = []

for question in questions:
    answer = qa(
        question=question,
        context=context
    )

    qa_results.append({
        "Question": question,
        "Answer": answer["answer"],
        "Score": answer["score"]
    })

print("\nQuestion Answering Results:")

for result in qa_results:
    print("\nQuestion:", result["Question"])
    print("Answer:", result["Answer"])

pd.DataFrame(qa_results).to_csv(
    "dataset/qa_results.csv",
    index=False
)

print("\nQA results saved successfully!")
from transformers import pipeline
import pandas as pd

print("Hugging Face Pipelines Task")
df = pd.read_csv("dataset/input_text.csv")

sentiment = pipeline("sentiment-analysis")

results = []

for text in df["text"]:
    result = sentiment(text)[0]
    results.append({
        "Text": text,
        "Sentiment": result["label"],
        "Score": result["score"]
    })

sentiment_df = pd.DataFrame(results)

sentiment_df
# Pipeline Comparison

comparison = pd.DataFrame({
    "Original Text": [
        "Artificial intelligence is changing the way people learn and work."
    ],
    "Summary": [summary],
    "Translation": [translation],
    "Sentiment": [results[0]["Sentiment"]],
    "Answer": [qa_results[0]["Answer"]]
})

comparison.to_csv(
    "dataset/pipeline_comparison.csv",
    index=False
)

print("Pipeline comparison saved successfully!")