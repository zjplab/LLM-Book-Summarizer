import streamlit as st
import os
import tempfile
from io import BytesIO
from llama_index.core import Settings
from llama_index.core.indices import DocumentSummaryIndex
from llama_index.core.retrievers import SummaryIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.readers.file import PDFReader
from utils.llm_config import setup_llm
from utils.pdf_processor import process_pdf_with_chapters
from utils.export_utils import export_summaries

# Page configuration
st.set_page_config(
    page_title="PDF Chapter Summarizer",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'document_index' not in st.session_state:
    st.session_state.document_index = None
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'chapters' not in st.session_state:
    st.session_state.chapters = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

st.title("📚 PDF Chapter Summarizer")
st.markdown("Upload a PDF book and get chapter-by-chapter summaries using advanced AI models")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_provider = st.selectbox(
        "AI Model Provider",
        ["OpenAI", "Anthropic"],
        help="Choose your preferred AI model provider"
    )
    
    # API Key input
    if model_provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        model_name = st.selectbox(
            "Model",
            ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            help="Select the OpenAI model to use"
        )
    else:  # Anthropic
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            help="Enter your Anthropic API key"
        )
        model_name = st.selectbox(
            "Model",
            ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022"],
            help="Select the Anthropic model to use"
        )
    
    st.divider()
    
    # Summarization prompt customization
    st.subheader("📝 Summarization Settings")
    
    default_prompt = """Please provide a comprehensive summary of this chapter that includes:
1. Main topics and key concepts
2. Important details and examples
3. Key takeaways and insights
4. How this chapter relates to the overall theme

Keep the summary detailed but concise, focusing on the most important information."""
    
    summarization_prompt = st.text_area(
        "Summarization Prompt",
        value=default_prompt,
        height=150,
        help="Customize how you want each chapter to be summarized"
    )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        chunk_size = st.slider(
            "Chunk Size",
            min_value=512,
            max_value=4096,
            value=1024,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=200,
            value=50,
            help="Overlap between consecutive chunks"
        )

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📄 Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF book or document for chapter-by-chapter summarization"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"📊 File size: {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        # Process button
        if st.button("🚀 Process PDF", type="primary", disabled=not api_key):
            if not api_key:
                st.error("❌ Please provide an API key in the sidebar")
            else:
                try:
                    # Setup LLM
                    with st.spinner("🔧 Setting up AI model..."):
                        llm = setup_llm(model_provider, api_key, model_name)
                        Settings.llm = llm
                        Settings.chunk_size = chunk_size
                        Settings.chunk_overlap = chunk_overlap
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process PDF
                    with st.spinner("📖 Reading and parsing PDF..."):
                        documents = process_pdf_with_chapters(tmp_file_path)
                        st.session_state.chapters = [doc.metadata.get('chapter_title', f'Chapter {i+1}') 
                                                   for i, doc in enumerate(documents)]
                    
                    # Create DocumentSummaryIndex
                    with st.spinner("🏗️ Building document index..."):
                        st.session_state.document_index = DocumentSummaryIndex.from_documents(
                            documents,
                            show_progress=True
                        )
                    
                    # Generate summaries for each chapter
                    st.session_state.summaries = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, (doc, chapter_title) in enumerate(zip(documents, st.session_state.chapters)):
                        status_text.text(f"📝 Summarizing: {chapter_title}")
                        progress_bar.progress((i + 1) / len(documents))
                        
                        # Create retriever for this specific document
                        retriever = SummaryIndexRetriever(
                            index=st.session_state.document_index,
                            choice_select_prompt=summarization_prompt
                        )
                        
                        # Create TreeSummarize response synthesizer
                        response_synthesizer = TreeSummarize(
                            llm=llm,
                            summary_template=summarization_prompt
                        )
                        
                        # Create query engine
                        query_engine = RetrieverQueryEngine(
                            retriever=retriever,
                            response_synthesizer=response_synthesizer
                        )
                        
                        # Generate summary
                        summary_response = query_engine.query(
                            f"Summarize this chapter: {chapter_title}"
                        )
                        
                        st.session_state.summaries[chapter_title] = str(summary_response)
                    
                    # Cleanup
                    os.unlink(tmp_file_path)
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    st.session_state.processing_complete = True
                    
                    st.success("🎉 PDF processing completed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    if 'tmp_file_path' in locals():
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass

with col2:
    st.header("📋 Chapter Summaries")
    
    if st.session_state.processing_complete and st.session_state.summaries:
        # Export options
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("📥 Export as Text"):
                text_content = export_summaries(st.session_state.summaries, format="text")
                st.download_button(
                    label="Download Text File",
                    data=text_content,
                    file_name="chapter_summaries.txt",
                    mime="text/plain"
                )
        
        with col2b:
            if st.button("📥 Export as Markdown"):
                markdown_content = export_summaries(st.session_state.summaries, format="markdown")
                st.download_button(
                    label="Download Markdown File",
                    data=markdown_content,
                    file_name="chapter_summaries.md",
                    mime="text/markdown"
                )
        
        st.divider()
        
        # Display summaries
        for i, (chapter_title, summary) in enumerate(st.session_state.summaries.items()):
            with st.expander(f"📖 {chapter_title}", expanded=(i == 0)):
                st.markdown(summary)
                
                # Individual chapter export
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.download_button(
                        label="📄 Download as Text",
                        data=f"# {chapter_title}\n\n{summary}",
                        file_name=f"{chapter_title.replace(' ', '_').lower()}.txt",
                        mime="text/plain",
                        key=f"text_{i}"
                    )
                with col_exp2:
                    st.download_button(
                        label="📝 Download as Markdown",
                        data=f"# {chapter_title}\n\n{summary}",
                        file_name=f"{chapter_title.replace(' ', '_').lower()}.md",
                        mime="text/markdown",
                        key=f"md_{i}"
                    )
    else:
        st.info("👆 Upload a PDF file and click 'Process PDF' to generate chapter summaries")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        📚 PDF Chapter Summarizer powered by LlamaIndex & Advanced AI Models
    </div>
    """,
    unsafe_allow_html=True
)
