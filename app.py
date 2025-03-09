import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_resources():
    """Initialize embeddings, Pinecone client, and BM25 encoder"""
    # Load environment variables
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    
    # Initialize Hugging Face Embeddings (for dense vectors)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    
    # Connect to existing index
    index_name = "hybrid-search-langchain-pinecone"
    index = pc.Index(index_name)
    
    # Load BM25 encoder from saved values
    bm25_encoder = BM25Encoder().load("bm25_values.json")
    
    return embeddings, index, bm25_encoder

def hybrid_search(query, index, embeddings, bm25_encoder, top_k=5, alpha=0.7):
    """
    Perform hybrid search using both dense and sparse embeddings
    
    Args:
        query: The search query string
        index: Pinecone index
        embeddings: HuggingFace embeddings model
        bm25_encoder: BM25 encoder for sparse vectors
        top_k: Number of results to return
        alpha: Weight for dense vectors (1-alpha = weight for sparse vectors)
        
    Returns:
        List of search results
    """
    # Get dense vector for query
    dense_emb = embeddings.embed_query(query)
    
    # Get sparse vector for query
    sparse_emb = bm25_encoder.encode_queries([query])[0]
    
    # Perform hybrid search
    results = index.query(
        vector=dense_emb,
        sparse_vector=sparse_emb,
        top_k=top_k,
        include_metadata=True,
        alpha=alpha
    )
    
    return results.matches

def insert_sample_data(index, embeddings, bm25_encoder, sentences):
    """Insert sample data into Pinecone index"""
    # Get dense and sparse vectors for sentences
    dense_vectors = [embeddings.embed_query(s) for s in sentences]
    sparse_vectors = bm25_encoder.encode_documents(sentences)
    
    # Create upsert batch
    batch = []
    for i, (sentence, dense_vec, sparse_vec) in enumerate(zip(sentences, dense_vectors, sparse_vectors)):
        batch.append({
            "id": f"doc_{i}",
            "values": dense_vec,
            "sparse_values": sparse_vec,
            "metadata": {"text": sentence}
        })
    
    # Upsert to Pinecone
    index.upsert(batch)
    st.success(f"Successfully inserted {len(batch)} documents")

# Main Streamlit app
def main():
    st.title("Hybrid Search with Pinecone and BM25")
    
    # Initialize resources
    with st.spinner("Loading resources..."):
        embeddings, index, bm25_encoder = initialize_resources()
    
    # App sections
    tab1, tab2 = st.tabs(["Search", "Upload Data"])
    
    # Search tab
    with tab1:
        st.header("Search Documents")
        query = st.text_input("Enter your search query:")
        alpha = st.slider("Dense-Sparse Weight (alpha)", 0.0, 1.0, 0.7, 
                         help="1.0 = pure dense search, 0.0 = pure sparse search")
        top_k = st.number_input("Number of results", min_value=1, max_value=20, value=5)
        
        if st.button("Search") and query:
            with st.spinner("Searching..."):
                results = hybrid_search(query, index, embeddings, bm25_encoder, top_k, alpha)
                
            if results:
                st.subheader("Search Results")
                for i, result in enumerate(results):
                    with st.expander(f"Result {i+1} (Score: {result.score:.4f})"):
                        st.markdown(result.metadata.get("text", "No text available"))
                        st.json({
                            "id": result.id,
                            "score": result.score,
                            "metadata": result.metadata
                        })
            else:
                st.info("No results found.")
    
    # Upload data tab
    with tab2:
        st.header("Insert Sample Data")
        st.markdown("Use this section to insert your sample sentences into Pinecone.")
        
        # Sample data from your notebook
        default_data = "The cat jumped over the fence.\nShe found a hidden note under the table.\nThe sky turned orange at sunset."
        
        sample_data = st.text_area("Enter sample sentences (one per line):", 
                                  value=default_data, height=200)
        
        if st.button("Insert Data"):
            sentences = [s.strip() for s in sample_data.split("\n") if s.strip()]
            if sentences:
                with st.spinner("Inserting data..."):
                    insert_sample_data(index, embeddings, bm25_encoder, sentences)
            else:
                st.error("Please enter at least one sentence.")

if __name__ == "__main__":
    main()