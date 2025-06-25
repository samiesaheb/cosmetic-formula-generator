# Cosmetic Formula Generator (RAG)

## Overview

This project is an open-source Retrieval-Augmented Generation (RAG) system designed to generate new cosmetic formulations based on a dataset of existing cosmetic product formulas. It leverages:

- **ChromaDB** for vector storage and semantic retrieval,
- **Sentence Transformers** for embedding cosmetic formulations,
- **Ollama** to run open-source large language models (LLMs) locally,
- **Streamlit** for an interactive web interface.

Users can input a product description (e.g., "a gentle moisturizing face cream for sensitive skin"), and the system retrieves relevant existing formulas from the dataset and generates a novel, detailed cosmetic formula including ingredient names, INCI names, percentages, and formulation parts (e.g., Part A, Part B).

---

## Features

- **Semantic Search:** Retrieves the most relevant cosmetic formulas from your dataset based on user queries.
- **LLM Generation:** Generates new cosmetic formulas informed by retrieved examples.
- **Ingredient Details:** Outputs ingredient names, INCI names, percentages, and part assignments.
- **Local and Open Source:** Runs fully locally using open-source tools and models—no paid APIs required.
- **Interactive UI:** Streamlit app for easy querying and formula generation.

---

## What It Does Well

- **Contextual Generation:** Combines retrieval and generation to produce formulas grounded in real examples.
- **Ingredient Part Inclusion:** Clearly specifies which part each ingredient belongs to, aiding formulation clarity.
- **Open-Source Stack:** Uses freely available tools and models, making it accessible and customizable.
- **User-Friendly Interface:** Simple web app interface for non-technical users to generate cosmetic formulas.

---

## Known Limitations and What It Does Wrong

- **Percentage Sum Accuracy:** Generated ingredient percentages may not always sum exactly to 100%. Post-processing normalization or manual adjustment is recommended.
- **Model Dependency:** The quality of generated formulas depends on the underlying LLM and dataset quality.
- **Ingredient Safety & Compliance:** The system does not verify ingredient safety, regulatory compliance, or formulation feasibility—human expert review is essential.
- **Performance:** Embedding and retrieval can be slow on large datasets without optimization.
- **Ollama Dependency:** Requires Ollama installed locally with compatible LLMs; setup may be complex for some users.

---

## Installation and Setup

1. **Clone the repository:**

git clone https://github.com/yourusername/cosmetic-formula-generator.git
cd cosmetic-formula-generator

2. **Create and activate a Python virtual environment (recommended):**

3. python3 -m venv venv
source venv/bin/activate # macOS/Linux
venv\Scripts\activate # Windows

3. **Install dependencies:**

pip install -r requirements.txt

4. **Download and prepare your cleaned dataset:**

Place your `formulations_cleaned.csv` file in the project root directory.

5. **Install and set up Ollama:**

- Download from [https://ollama.com/](https://ollama.com/)
- Pull the LLM model (e.g., `llama3`):

  ```
  ollama pull llama3
  ```

6. **Run the Streamlit app:**

streamlit run app.py

7. **Open your browser at** `http://localhost:8501` and start generating cosmetic formulas!

---

## Usage

- Enter a detailed description of the cosmetic product you want to formulate.
- Click **Generate Formula**.
- The app retrieves related formulas and generates a new formula with ingredient parts and percentages.
- Review and adjust the generated formula as needed.

---

## Future Improvements

- Add automatic normalization of ingredient percentages to sum to 100%.
- Integrate safety and regulatory checks for ingredients.
- Support exporting generated formulas in structured formats (JSON, CSV).
- Improve prompt engineering for more consistent output.
- Add user feedback loop to iteratively improve formula quality.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [ChromaDB](https://github.com/chroma-core/chroma)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/)



