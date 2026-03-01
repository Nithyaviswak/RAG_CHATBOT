import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector = embeddings.embed_query("hello world")
    print("✅ Success! Gemini Embeddings are working.")
except Exception as e:
    print(f"❌ Failed! The error is: {e}")