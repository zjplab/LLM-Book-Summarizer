# PDF Chapter Summarizer

## Overview

This is a Streamlit-based PDF Chapter Summarizer application that uses advanced AI models to process PDF documents and generate chapter-by-chapter summaries. The application leverages LlamaIndex for document processing and retrieval, supporting both OpenAI and Anthropic language models for summary generation.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **UI Components**: 
  - Main interface for PDF upload and processing
  - Sidebar for configuration (model selection, API keys)
  - Session state management for persistent data across user interactions
- **Layout**: Wide layout with responsive design using Streamlit's built-in components

### Backend Architecture
- **Document Processing**: LlamaIndex framework for PDF parsing and indexing
- **LLM Integration**: Abstracted LLM configuration supporting multiple providers
- **Chapter Detection**: Pattern-based text parsing to identify chapter boundaries
- **Summary Generation**: DocumentSummaryIndex with TreeSummarize response synthesizer

### Language Model Support
- **OpenAI**: GPT-4o model (latest as of May 13, 2024)
- **Anthropic**: Claude Sonnet 4 (claude-sonnet-4-20250514) as the preferred latest model
- **Custom AI Vendor**: Support for custom API endpoints including OpenRouter, DeepSeek, and other OpenAI-compatible services
- **Configuration**: Adjustable temperature (0.0-1.0), 4000 max tokens, custom model names and API hosts

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Primary Streamlit interface and application orchestration
- **Features**: 
  - PDF file upload handling
  - Model provider selection
  - Session state management for processing results
  - Integration with all utility modules

### 2. LLM Configuration (`utils/llm_config.py`)
- **Purpose**: Centralized language model setup and configuration
- **Features**:
  - Multi-provider support (OpenAI, Anthropic)
  - API key validation
  - Model-specific parameter configuration
  - Temperature and token limit optimization for summarization tasks

### 3. PDF Processing (`utils/pdf_processor.py`)
- **Purpose**: PDF document parsing and chapter extraction
- **Features**:
  - LlamaIndex PDFReader integration
  - Chapter boundary detection using regex patterns
  - Content cleaning and preprocessing
  - Document metadata enrichment

### 4. Export Utilities (`utils/export_utils.py`)
- **Purpose**: Summary export functionality in multiple formats
- **Features**:
  - Text format export with table of contents
  - Markdown format export support
  - Timestamp and metadata inclusion
  - Structured output formatting

## Data Flow

1. **PDF Upload**: User uploads PDF file through Streamlit interface
2. **Document Processing**: PDF is parsed using LlamaIndex PDFReader
3. **Chapter Extraction**: Text is split into chapters using pattern matching
4. **Document Indexing**: Chapters are indexed using DocumentSummaryIndex
5. **Summary Generation**: AI model generates summaries for each chapter
6. **Result Storage**: Summaries stored in session state for persistence
7. **Export**: Users can download summaries in text or markdown format

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **LlamaIndex**: Document processing and retrieval framework
- **OpenAI**: GPT model integration
- **Anthropic**: Claude model integration

### Processing Libraries
- **PDFReader**: PDF document parsing (part of LlamaIndex)
- **tempfile**: Temporary file handling for uploads
- **BytesIO**: In-memory file operations

## Deployment Strategy

### Local Development
- Streamlit application designed for local execution
- Environment variables or user input for API keys
- Session-based state management for user data persistence

### Considerations
- API keys handled securely through password input fields
- Temporary file cleanup for uploaded PDFs
- Memory management for large documents through chunked processing

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- June 30, 2025. Initial setup