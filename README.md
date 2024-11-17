# **Judicial RAG System for Supreme Court of Pakistan**
This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system designed for judicial use cases, specifically for the Supreme Court of Pakistan (SCP). The project focuses on leveraging Large Language Models (LLMs) and advanced retrieval techniques to analyze and generate detailed, relevant, and accurate responses to legal queries based on the SCP Judgements.

## **Project Overview**
The primary goal of this project is to develop a robust, efficient, and reliable RAG pipeline capable of handling legal texts, retrieving relevant sections, and generating comprehensive summaries and answers to user queries. The system integrates advanced techniques for metadata handling, retrieval, and generation enhancement through context handling and summarization to provide high-quality outputs tailored to the judicial domain. This project is a collaborative effort of Abdul Haseeb, Annayah Usman, and Sawera Hanif.

## **Key Features**
* LLMs Used:
  * LLaMa 3.1. 8B Instruct
  * Mistral 7B Instruct
* Embedding Model Used:
  * Stella 1.5B (Dunzhang/Stella_en_1.5B_v5)
* Retrieval Techniques:
  * Similarity Search
  * BM25
  * Ensemble Retriever (Best Performer)
* Chunking Strategy & Metadata Handling:
  * Metadata embedding in the splitted chunks for improved precision and relevance during retrieval.
* Summarization Strategies:
  * Without Query Bias: Generate summaries without including the user query to avoid biases in the generated answers.
* Long Context Handling:
  * Implemented Long Context Reordering to improve context flow before summarization and query answering.

## **Workflow**
* Corpus Generation & Preprocessing
  * Step 1: Web scrape SCP judgments and associated metadata (titles, case numbers, author judges, etc.). For further details refer to the DocSrapper.ipynb in the repo.
  * Step 2: Convert judgment PDFs to text (incld. OCR) and link them with metadata. For further details refer to the PDF2TXT&CSV.ipynb in the repo.
  * Step 3: Preprocess text by removing redundant elements, normalizing whitespace, and cleaning content. For further details refer to the TextPreprocessing.ipynb in the repo.
* Chunking Strategy & Metadata Handling
  * Split documents into overlapping chunks of 1,800 characters for optimal embedding and retrieval through RecursiveCharacterTextSplitter from LangChain.
  * Embed each chunk with its associated metadata string.
* Embedding & Retrieval
  * Use dunzhang/stella_en_1.5B_v5 embedding model for high-dimensional (1024D) representations.
  * Store document embeddings in Pinecone for vector-based retrieval.
* Retrieval-Augmented Generation (RAG) Experimentation
  * Experiment with different query types (generic, specific, completely generic) and retrieval strategies (ensemble, similarity search, BM25).
  * Experiment with generation enhancement strategies like adding summary in the context and reordering of the context.
  * Evaluate whether summary should have query bias or not.
  * Evaluate whether long context reordering improves the query response.
  * Compare the LLMs with different query types on the best generation and retrieval strategy.
* Evaluation & Results
  * For the best generation and retrieval strategy along with superior LLM, compare RAG outputs for multiple queries with different query types for different documents.
Note: For further details on parts 2-5 refer to SCPRAG.ipynb in the repo.

## **Repository Contents**
| Folder/File                                          | Description                                                                                                                            |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| /Corpus/ | Contains scrapped judgements in PDF & Text format along with its metadata as a CSV file. Also, contains merged content and metadata CSV file as well as final cleaned corpus CSV. |
| /Corpus Generation & Preprocessing/DocScrapper.ipynb | Scrapes SCP judgments and associated metadata from the official website.     |
| /Corpus Generation & Preprocessing/PDF2TXT&CSV.ipynb | Converts judgment PDFs to text, performs OCR, and links metadata.     |
| /Corpus Generation & Preprocessing/TextPreprocessing.ipynb | Preprocesses text to clean, normalize, and structure legal documents.     |
| /SCPRAG.ipynb | Implements the RAG pipeline, chunking, embedding, and evaluation.     |
| /Report.PDF    | Detailed report explaining the workflow, experiments, and results.    |

## **Future Work**
* Metadata Embedding Strategy: Embed metadata separately for improved matching precision.
* Self-Querying Mechanism: Automate metadata tagging for enhanced search flexibility.
* Higher-Dimension Embeddings: Experiment with models offering richer representations for complex legal texts.
Note: More on this is present in the Report.PDF.

## **Acknowledgments**
* Tools: LangChain, HuggingFace, Pinecone, Selenium, PyMuPDF, PDF2image, PyTesseract, Beautifulsoup4, Pandas, Numpy, RegEx
* Models: LLaMa 3.1. 8B Instruct, Mistral 7B Instruct, Stella 1.5B V5, LegalBERT
* Data Source: Supreme Court of Pakistan Website.
* Instructor: Dr Sajjad Haider (Professor IBA Karachi)
