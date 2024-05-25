## Big Data Question Answering with Langchain and Streamlit

This repository contains a Python application for question-answering based on big data lecture notes from the Institute of Engineering (IOE). The system utilizes the Langchain framework for natural language processing (NLP) tasks and Streamlit for building an interactive user interface (UI).

### Features:
- **Document Processing:** The application loads lecture notes on big data from the IOE and processes them into smaller chunks suitable for analysis.
- **Embedding Generation:** Text chunks are converted into numerical embeddings using the Langchain framework, specifically the Ollama model for text embeddings.
- **Chroma Database:** Embeddings are stored in the Chroma vector store, facilitating efficient similarity search and retrieval.
- **User Interaction:** Users can input questions related to big data through an intuitive Streamlit interface.
- **Question-Answering:** The system uses the context from relevant document embeddings to generate accurate responses to user queries.
- **Interactive UI:** Streamlit provides a user-friendly interface for querying the system and visualizing responses in real-time.

### Project Flow:
- **Start**
- **Load Documents**
- **Split Documents into Chunks**
- **Calculate Chunk IDs**
- **Add to Chroma Vector Store**
- **User Inputs a Query in the Streamlit app**
- **Embed Query**
- **Perform Similarity Search in Chroma**
- **Retrieve Relevant Chunks**
- **Formulate Prompt with Retrieved Chunks**
- **Generate Response using Language Model**
- **Show Response and Query in Streamlit app**
- **End**

### Usage:
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the application using `streamlit run app.py`.
4. Input your questions in the provided text field and submit them to get instant answers based on the processed lecture notes.

### Data Source:
The lecture notes on big data used in this project are sourced from the Institute of Engineering (IOE) curriculum materials.

### Acknowledgements:
- Langchain: [Link to Langchain GitHub repository](https://github.com/your-langchain-repo)
- Streamlit: [Link to Streamlit GitHub repository](https://github.com/streamlit/streamlit)

### Contributions:
Contributions to this project are welcome! If you encounter any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.
