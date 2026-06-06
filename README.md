# ritam

A FastAPI application that provides a RAG (Retrieval-Augmented Generation) service using a Google Gemini LLM.

## Overview

This project is a Python-based application that leverages the power of Large Language Models (LLMs) to provide a
Retrieval-Augmented Generation (RAG) service. It uses a FastAPI backend to expose an API for querying the RAG service,
which can answer questions based on a provided dataset. The core of the RAG service is powered by Google's Gemini LLM.

## Blog post (walkthrough)

I wrote up the architecture, trade-offs, and what I learned here:

- https://vbhargava.org/writing/ritam-rag-v1/

## Features

- **FastAPI Backend**: A modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on
  standard Python type hints.
- **RAG Service**: The core of the application, which can answer questions based on a provided dataset.
- **Google Gemini LLM**: The application uses Google's Gemini LLM for both generating responses and creating embeddings.
- **File-based Vector Store**: The application supports a file-based vector store for storing and retrieving document
  embeddings.
- **Infrastructure as Code**: The infrastructure for the application is defined using Terraform, which allows you to
  deploy the application as a Docker container on Google Cloud Run.
- **Containerized Application**: The application is containerized using Docker, which makes it easy to deploy and run in
  any environment.
- **uv for Package Management**: The project uses `uv` for package management.

## Architecture

The application is composed of the following components:

- **FastAPI Backend**: The entry point of the application, which exposes the API for querying the RAG service.
- **RAG Service**: The core of the application, which is responsible for answering questions based on a provided
  dataset.
- **Google Gemini LLM**: The LLM used for generating responses and creating embeddings.
- **File-based Vector Store**: The vector store used for storing and retrieving document embeddings.
- **Terraform**: The tool used for defining the infrastructure of the application.
- **Docker**: The tool used for containerizing the application.

## Getting Started

### Prerequisites

- Python 3.14+
- `uv` for package management
- A Google API key for the Gemini LLM

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bhvishal9/ritam.git
   ```
2. Install the dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

### Configuration

The application requires a Google API key for the Gemini LLM. You can set the API key as an environment variable:

```bash
export LLM_API_KEY="your-api-key"
```

### Running the Application

To run the FastAPI server, you can use the following command:

```bash
uvicorn src.ritam.api.main:app --reload
```

## Usage

### CLI Commands

The project includes a CLI for indexing documents and querying the RAG service directly (without the API server).

#### Index

Index a directory of documents into a named dataset:

```bash
uv run python -m ritam.naive_rag index --dataset <dataset-name>
```

Options:

| Option | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Name of the dataset to create |
| `--source-dir` | `assets/docs` | Directory containing documents to index |
| `--chunk-size` | `10000` | Chunk size in characters |
| `--chunk-separator` | `\n\n` | String used to split chunks |

Example:

```bash
uv run python -m ritam.naive_rag index --dataset my-ducks --source-dir ./assets/ducks --chunk-size 5000
```

#### Query

Interactively query a previously indexed dataset:

```bash
uv run python -m ritam.naive_rag query --dataset <dataset-name>
```

You will be prompted to enter a question. The command retrieves the top 3 relevant chunks and returns a generated answer along with the sources used.

Example:

```bash
uv run python -m ritam.naive_rag query --dataset my-ducks
```

### API Endpoints

The following API endpoints are available:

- `GET /health`: Health check endpoint.
- `POST /echo`: Echo endpoint.
- `POST /query`: Query the RAG service.

### RAG Service

To query the RAG service, you can send a POST request to the `/query` endpoint with the following JSON payload:

```json
{
  "query": "What is the capital of France?",
  "dataset": "my-dataset",
  "top_k": 5
}
```

## Deployment

The application can be deployed as a Docker container on Google Cloud Run. The infrastructure is defined using
Terraform.

To deploy the application, you can use the following commands:

```bash
terraform init
terraform apply
```

## Development

### Running Tests

To run the tests, you can use the following command:

```bash
pytest
```

### Code Formatting and Linting

The project uses `ruff` for code formatting and linting. To format and lint the code, you can use the following
commands:

```bash
ruff format .
ruff check .
```