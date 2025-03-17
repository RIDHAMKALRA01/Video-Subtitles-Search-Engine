
import streamlit as st
from subtitle_processor import SubtitleProcessor
import os


@st.cache_resource
def init_processor():
    return SubtitleProcessor()

processor = init_processor()

def main():
    st.title("Video Subtitle Search Engine")
    st.write("Search through video subtitles using text or audio queries.")

    # Sidebar for data status
    with st.sidebar:
        st.header("Data Status")
        st.info("Sample subtitle data loaded automatically.")
        if st.button("Reload Sample Data"):
            with st.spinner("Reloading sample data..."):
                processor.load_sample_data()
                st.success("Data reloaded successfully!")

    # Main search interface
    search_type = st.radio("Select search type:", ("Text Query", "Audio Query"))
    
    if search_type == "Text Query":
        query = st.text_input("Enter your search query:", "Someone saves the world")
        if st.button("Search") and query:
            with st.spinner("Searching and generating response..."):
                results = processor.search(query)
                rag_response = processor.rag_generate(query, results)
                display_results(results, rag_response)
    
    else:
        uploaded_file = st.file_uploader("Upload audio file (WAV format)", type=['wav'])
        if uploaded_file and st.button("Search"):
            with st.spinner("Processing audio and generating response..."):
                temp_path = "temp_audio.wav"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                results = processor.search(temp_path)
                rag_response = processor.rag_generate(temp_path, results) if results else "No results to process."
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                display_results(results, rag_response)

def display_results(results, rag_response):
    if not results:
        st.warning("No results found or error processing query")
    else:
        st.subheader("Search Results")
        for result in results:
            with st.expander(f"Match - Score: {result['score']:.3f}"):
                st.write(f"File: {result['metadata']['name']}")
                st.write(f"Chunk ID: {result['metadata']['chunk_id']}")
                st.write("Content:")
                st.text(result['content'])

    st.subheader("Google Gemini Enhanced Response")
    st.write(rag_response)

if __name__ == "__main__":
    main()