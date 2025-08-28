# LegalMEL: Spanish Legal Chatbot

## Overview

LegalMEL is a project aimed at developing a Spanish legal chatbot that leverages the MEL (Modelo de Embeddings Legales) model from Hugging Face Transformers. This chatbot is designed to assist users by providing relevant legal information and answering questions based on legal texts.

## Key Features

- **MEL Model Integration**: The chatbot utilizes the MEL model, which is specifically trained for legal Spanish texts, to generate embeddings that capture the semantic meaning of legal documents.
- **Text Indexing with FAISS**: The project employs FAISS (Facebook AI Similarity Search) to index the embeddings of legal texts, allowing for efficient retrieval of relevant sections in response to user queries.
- **Dynamic Response Generation**: The chatbot can generate context-aware responses using a language model, ensuring that the answers are not only relevant but also coherent and informative.

## Project Structure

- `chat_rag.py`: Contains the main implementation of the chatbot, including functions for embedding legal texts, indexing them, and generating responses based on user queries.
- `LICENSE`: Details the Creative Commons Attribution-NonCommercial 4.0 International Public License under which the project is shared.
- `.gitignore`: Specifies files and directories to be ignored by Git, such as the `env/` directory for virtual environment files.
- `requirements.txt`: Lists the necessary Python dependencies for the project, including `torch`, `transformers`, `faiss-cpu`, `numpy`, and `PyPDF2`.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd LegalMEL
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To use the chatbot, run the `chat_rag.py` script. You can modify the `question` variable in the script to ask different legal questions. The chatbot will retrieve relevant legal information and generate a response based on the context provided by the indexed legal texts.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License. Please refer to the LICENSE file for more details. MEL was released under an [specific NonCommercial License](https://huggingface.co/IIC/MEL/blob/main/LICENSE).

## Contributing

Contributions to the LegalMEL project are welcome! If you have suggestions or improvements, feel free to submit a pull request or open an issue.

## Acknowledgments

Special thanks to the developers of the MEL model and the Hugging Face Transformers library for providing the tools necessary to build this legal chatbot.
