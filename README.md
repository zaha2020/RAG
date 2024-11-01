# Retrieval-Augmented Generation (RAG) System for PDF Documents

## Decription:

This mini-project implements a Retrieval-Augmented Generation (RAG) system for document-based Q&A using large language models, specifically designed for Persian PDF files. The system utilizes jtatman/orca-tau-4k-persian-alpaca-f32 to perform semantic search and generate contextually accurate answers. The PDFProcessor class extracts and processes text from Persian PDFs, enabling efficient retrieval of relevant information. When a query is made, the system retrieves the most relevant chunks from the PDF and feeds them into jtatman/orca-tau-4k-persian-alpaca-f32 to generate responses. This setup is ideal for robust Q&A in Persian and multilingual contexts.
