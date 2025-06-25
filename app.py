import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import subprocess
import os

# --- Configuration ---
# Ensure the chroma_db directory exists
CHROMA_DB_DIR = "./chroma_db"
if not os.path.exists(CHROMA_DB_DIR):
    os.makedirs(CHROMA_DB_DIR)

# --- Data Loading and Preparation ---
@st.cache_resource(show_spinner="Loading and preparing data...")
def load_and_prepare_data(csv_path='formulations_cleaned.csv'):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: {csv_path} not found. Please make sure the CSV file is in the same directory as the app.")
        st.stop()

    def format_ingredients(group):
        # Ensure 'part' column exists and is handled (assuming it's 'Part' after initial cleaning)
        # Check if 'part' column exists, otherwise default to empty string
        part_col_name = 'part' if 'part' in group.columns else 'Part' # Adjust based on your exact cleaned CSV column name
        
        formatted_list = []
        for _, row in group.iterrows():
            part_info = f", Part {row[part_col_name]}" if part_col_name in row and pd.notna(row[part_col_name]) else ""
            formatted_list.append(
                f"- {row['ingredient']} ({row['inci']}){part_info}: {row['percent']}%"
            )
        return "\n".join(formatted_list)

    # Apply the formatting and rename for consistent access
    formulations = (
        df.groupby('product_name')
        .apply(format_ingredients)
        .reset_index()
        .rename(columns={0: 'formulation_text'})
    )
    return formulations

# --- Embedding Model and Vector Database Initialization ---
@st.cache_resource(show_spinner="Initializing embedding model and database...")
def init_embedding_and_db(formulations_data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.Client(Settings(persist_directory=CHROMA_DB_DIR))
    collection = chroma_client.get_or_create_collection("cosmetic_formulas")

    # Only add documents if the collection is empty
    if collection.count() == 0:
        with st.spinner("Populating vector database... (This might take a moment)"):
            documents = formulations_data['formulation_text'].tolist()
            metadatas = [{"product_name": name} for name in formulations_data['product_name'].tolist()]
            ids = [str(i) for i in range(len(documents))]

            # ChromaDB add can take lists directly for efficiency
            collection.add(
                documents=documents,
                embeddings=model.encode(documents).tolist(),
                ids=ids,
                metadatas=metadatas
            )
        st.success(f"Vector database populated with {collection.count()} formulations!")
    else:
        st.info(f"Vector database already contains {collection.count()} formulations. Skipping re-population.")
            
    return model, collection

# --- Retrieval Function ---
def retrieve_similar_formulas(embedding_model, chroma_collection, query, top_k=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents'] # Ensure documents are included in the results
    )
    # results['documents'] is a list of lists, take the first inner list which contains the documents
    return results['documents'][0] if results['documents'] else []

# --- Generation with Ollama ---
def generate_formula_with_ollama(query, retrieved_formulas):
    if not retrieved_formulas:
        # Fallback if no formulas were retrieved
        base_prompt = (
            f"Generate a new cosmetic formula for: {query}\n"
            "Please list each ingredient with its INCI name, percentage (total 100%), "
            "and specify which part (e.g., Part A, Part B) it belongs to."
        )
    else:
        # Construct prompt with retrieved examples
        example_formulations_text = "\n\n".join(retrieved_formulas)
        base_prompt = (
            f"Here are some example cosmetic formulations:\n\n"
            f"{example_formulations_text}\n\n"
            f"Based on these examples, generate a new cosmetic formula for: {query}\n"
            "Please list each ingredient with its INCI name, percentage (total 100%), "
            "and specify which part (e.g., Part A, Part B) it belongs to."
        )

    st.info("Sending request to Ollama...")
    try:
        # Attempt to run the ollama command
        result = subprocess.run(
            ["ollama", "run", "llama3", base_prompt],
            capture_output=True, text=True, check=True # check=True raises CalledProcessError for non-zero exit codes
        )
        return result.stdout
    except FileNotFoundError:
        st.error("Ollama command not found. Make sure Ollama is installed and in your system's PATH.")
        st.error("Download Ollama from https://ollama.com/ and pull the 'llama3' model.")
        return "Error: Ollama not found or not configured."
    except subprocess.CalledProcessError as e:
        st.error(f"Error running Ollama: {e.stderr}")
        st.error("Please ensure the 'llama3' model is pulled (e.g., `ollama pull llama3`) and Ollama server is running.")
        return f"Error: Ollama process failed. {e.stderr}"
    except Exception as e:
        st.error(f"An unexpected error occurred while calling Ollama: {e}")
        return "Error: An unexpected error occurred during generation."


# --- Streamlit UI ---
st.set_page_config(page_title="Cosmetic Formula Generator", layout="wide")

st.title("🧪 Cosmetic Formula Generator (RAG)")
st.markdown("""
Enter a description of the cosmetic product you want to formulate. 
The system will retrieve similar existing formulas and use them to generate a new one, 
including ingredients, INCI names, percentages, and their respective parts.
""")

# Load data and initialize models only once
formulations_df = load_and_prepare_data()
embedding_model, chroma_collection = init_embedding_and_db(formulations_df)

# User input
user_query = st.text_area(
    "Describe the cosmetic formula you want to generate:",
    "A gentle moisturizing face cream for sensitive skin, suitable for daily use."
)

if st.button("✨ Generate Formula ✨"):
    if not user_query.strip():
        st.warning("Please enter a description for the cosmetic formula.")
    else:
        with st.spinner("Retrieving relevant formulas and generating new one..."):
            retrieved_documents = retrieve_similar_formulas(embedding_model, chroma_collection, user_query, top_k=5)
            
            if not retrieved_documents:
                st.warning("Could not retrieve any similar formulas. Generating based on general knowledge.")
                
            generated_formula = generate_formula_with_ollama(user_query, retrieved_documents)
            
            st.subheader("Generated Cosmetic Formula:")
            st.code(generated_formula, language='text') # Use st.code for better formatting of multi-line text

st.markdown("---")
st.caption("Powered by Streamlit, Sentence Transformers, ChromaDB, and Ollama.")
st.caption("Ensure Ollama is running and 'llama3' model is pulled for full functionality.")
