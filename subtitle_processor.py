import os
import re
import numpy as np
import librosa
import speech_recognition as sr
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

class SubtitleProcessor:
    def __init__(self):
        self.api_key = "API-KEY"
        genai.configure(api_key=self.api_key)  
        self.chroma_client = chromadb.PersistentClient(path="./subtitle_db")
        
        if "subtitles" in self.chroma_client.list_collections():
            self.chroma_client.delete_collection("subtitles")
        self.collection = self.chroma_client.create_collection("subtitles")
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorizer = TfidfVectorizer()
        self.load_subtitle_data()
    
    def clean_text(self, text):
        text = re.sub(r'\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower().strip()
    
    def chunk_document(self, text):
        words = text.split()
        chunk_size = min(500, len(words))
        overlap = max(50, int(0.1 * chunk_size))
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
        return chunks
    
    def load_subtitle_data(self, directory="subtitles"):  
        all_texts = []
        for filename in os.listdir(directory):
            if filename.endswith(".srt") or filename.endswith(".txt"):
                with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                    content = file.read()
                    cleaned_text = self.clean_text(content)
                    chunks = self.chunk_document(cleaned_text)
                    embeddings = self.model.encode(chunks)
                    all_texts.extend(chunks)
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        self.collection.add(
                            embeddings=[embedding.tolist()],
                            documents=[chunk],
                            metadatas=[{"name": filename, "chunk_id": i}],
                            ids=[f"{filename}_{i}"]
                        )
        self.vectorizer.fit(all_texts)  
    
    def process_audio_query(self, audio_path):
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source, duration=120)  
        try:
            return r.recognize_google(audio)
        except Exception:
            return None
    
    def search(self, query, method="semantic"):
        if isinstance(query, str) and query.endswith(".wav"):
            query = self.process_audio_query(query) or ""
        
        if method == "semantic":
            query_embedding = self.model.encode([query])[0]
            results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=5)
            
            retrieved_results = []
            for i, doc in enumerate(results.get('documents', [[]])[0]):  # Handle missing keys
                retrieved_results.append({
                    "content": doc,
                    "score": results.get("distances", [[0]])[0][i],
                    "metadata": results.get("metadatas", [[{}]])[0][i]
                })
        else:
            tfidf_matrix = self.vectorizer.transform([query])
            doc_matrix = self.vectorizer.transform([doc for doc in self.collection.get().get('documents', [])])
            scores = np.dot(doc_matrix, tfidf_matrix.T).toarray().flatten()
            sorted_indices = np.argsort(scores)[::-1][:5]
            retrieved_results = [{
                'content': self.collection.get().get('documents', [])[i],
                'score': scores[i],
                'metadata': self.collection.get().get('metadatas', [])[i]
            } for i in sorted_indices]
        
        return retrieved_results  
    
    def rag_generate(self, query, retrieved_results):
        if not isinstance(retrieved_results, list):
            print("Error: retrieved_results is not a list!", retrieved_results)
            return "Error: Invalid results format."
        
        system_prompt = "You are an AI that enhances subtitle searches. Given a query and retrieved subtitle chunks, generate a response using context."
        
        try:
            context = "\n".join([f"{r.get('content', 'No Content')} (Score: {r.get('score', 0):.3f})" for r in retrieved_results])
        except Exception as e:
            print("Error formatting context:", e)
            return "Error formatting retrieved data."
        
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(f"{system_prompt}\nQuery: {query}\nContext:\n{context}")
            return response.text
        except Exception as e:
            print("Error generating response:", e)
            return "Error generating response."
