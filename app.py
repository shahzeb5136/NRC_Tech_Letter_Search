import streamlit as st
import requests
import json
import os
from llama_cloud_services import LlamaCloudIndex

# --- Page Configuration ---
st.set_page_config(
    page_title="NRC Document Query System",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dark Purple Theme ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main Background - Corporate Deep Teal/Dark */
    .stApp {
        background: linear-gradient(135deg, #002b2d 0%, #004044 50%, #005055 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001f21 0%, #003336 100%);
        border-right: 1px solid rgba(0, 163, 173, 0.3);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #f0f0f0;
    }
    
    /* Headers - Using ENEC Teal and White */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    h1 {
        /* Gradient using the logo's Cyan and Green */
        background: linear-gradient(90deg, #00A3AD, #8CC63F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        letter-spacing: -0.5px;
    }
    
    /* Text Input Styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(0, 31, 33, 0.8) !important;
        border: 2px solid rgba(0, 163, 173, 0.4) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        padding: 15px !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #8CC63F !important;
        box-shadow: 0 0 15px rgba(140, 198, 63, 0.2) !important;
    }
    
    /* Button Styling - ENEC Teal to Green Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #006269 0%, #00A3AD 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(0, 98, 105, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00A3AD 0%, #8CC63F 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(140, 198, 63, 0.4) !important;
    }
    
    /* Card Styling for Results */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 163, 173, 0.2);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        border-color: #8CC63F;
        background: rgba(255, 255, 255, 0.08);
    }
    
    .document-name {
        color: #00A3AD;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .section-number {
        color: #8CC63F;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Spinner/Loading */
    .stSpinner > div {
        border-color: #00A3AD !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00A3AD;
        border-radius: 10px;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: rgba(0, 43, 45, 0.6);
        border: 1px solid rgba(0, 163, 173, 0.3);
        border-radius: 10px;
        padding: 15px;
    }
    
    [data-testid="stMetricValue"] {
        color: #00A3AD !important;
    }
</style>
""", unsafe_allow_html=True)


# --- API Functions (from 01_query.py) ---
def get_core42_response(role: str, content: str, system_instruction: str = None) -> str:
    """Sends a message to the Core42 API."""
    API_KEY = "81052a984bee43ee865f296e5a88e5f1"
    API_URL = "https://api.core42.ai/v1/chat/completions"
    
    if not API_KEY:
        return "Error: API Key is missing."

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    
    messages.append({"role": role, "content": content})
    data = {"model": "gpt-4o", "stream": False, "messages": messages, "temperature": 0.0}

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"


def run_custom_rag(query: str):
    """Run RAG query using LlamaCloud."""
    index = LlamaCloudIndex(
        name="nrc",
        project_name="Default",
        organization_id="c9e1176c-fd12-4819-bfd7-fc23f80ab9b5",
        api_key="llx-7GWrpifAsJ6UJYIPoVa9jl2tSRwmeIOpbi4Hcj5ynctYzJ4B",
    )

    nodes = index.as_retriever(similarity_top_k=5).retrieve(query)
    
    context_text = ""
    for i, node_with_score in enumerate(nodes):
        chunk_content = node_with_score.node.get_content()
        metadata = node_with_score.node.metadata 
        
        context_text += f"\n--- Source {i+1} ---\n"
        context_text += f"METADATA: {metadata}\n"
        context_text += f"CONTENT: {chunk_content}\n"

    system_prompt = (
        "You are a technical document assistant for nuclear energy regulations. "
        "Your task is to identify the specific document name and section number "
        "that contains the information relevant to the user's question. "
        "\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the provided context chunks.\n"
        "2. Extract the 'Document Name' and 'Section Number' for relevant sources.\n"
        "3. Return ONLY a valid JSON object. Do not include any conversational text.\n"
        "4. If the answer is not in the context, return: {\"error\": \"Information not found in available documents.\"}\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "{\n"
        "  \"references\": [\n"
        "    {\"document_name\": \"string\", \"section_number\": \"string\", \"relevance_summary\": \"string\"}\n"
        "  ]\n"
        "}"
    )
    
    final_user_content = f"CONTEXT:\n{context_text}\n\nUSER QUESTION: {query}"

    return get_core42_response(
        role="user", 
        content=final_user_content, 
        system_instruction=system_prompt
    )


def get_pdf_file_info(file_path: str):
    """Get PDF file information for display and download."""
    try:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            return {
                "path": file_path,
                "name": file_name,
                "size": file_size,
                "size_formatted": f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.2f} MB"
            }
        return None
    except Exception as e:
        return None


def find_pdf_file(document_name: str, data_folder: str) -> str:
    """Find PDF file path based on document name."""
    # Clean the document name
    doc_name = document_name.strip()
    
    # If it already ends with .pdf, use as is
    if not doc_name.lower().endswith('.pdf'):
        doc_name = doc_name + '.pdf'
    
    # Search in data folder recursively
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.lower() == doc_name.lower():
                return os.path.join(root, file)
            # Also try partial match
            if doc_name.lower().replace('.pdf', '') in file.lower():
                return os.path.join(root, file)
    
    return None


# --- Main App ---
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚛️ NRC Query System")
        st.markdown("---")
        st.markdown("""
        <div style='color: #b0b0c0; font-size: 0.9rem; line-height: 1.6;'>
            <p><strong style='color: #bb86fc;'>About:</strong></p>
            <p>This dashboard uses AI-powered semantic search to find relevant sections in NRC Technical Letters.</p>
            <br>
            <p><strong style='color: #bb86fc;'>How to use:</strong></p>
            <ol>
                <li>Enter your technical question</li>
                <li>Click "Search Documents"</li>
                <li>Review the relevant sections</li>
                <li>Download the PDF document</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style='color: #666; font-size: 0.8rem;'>
            Powered by LlamaCloud & Core42 AI
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("# ⚛️ NRC Technical Document Query System")
    st.markdown("<p style='color: #b0b0c0; font-size: 1.1rem; margin-bottom: 30px;'>Search through nuclear regulatory documents using natural language queries</p>", unsafe_allow_html=True)

    # NRC Information Guidance Section
    with st.expander("ℹ️ About the US NRC & Technical Letters", expanded=False):
        st.markdown("""
            <div style='background: linear-gradient(145deg, rgba(52, 152, 219, 0.1), rgba(41, 128, 185, 0.05)); 
                        border-left: 4px solid #3498db; 
                        padding: 16px 20px; 
                        border-radius: 8px;
                        margin: 10px 0;'>
                <h4 style='color: #81d4fa; margin-top: 0; margin-bottom: 12px;'>Regulatory Oversight & Guidance</h4>
                <p style='color: #b0b0c0; font-size: 0.95rem; margin-bottom: 12px;'>
                    The <b>U.S. Nuclear Regulatory Commission (NRC)</b> is an independent agency responsible for ensuring the safe use of radioactive materials for beneficial civilian purposes while protecting people and the environment.
                </p>
                <div style='color: #a0a0b0; font-size: 0.9rem;'>
                    <p style='margin-bottom: 8px;'><strong style='color: #f39c12;'>📜 Key Technical Communication Types:</strong></p>
                    <ul style='margin-left: 20px; margin-bottom: 12px;'>
                        <li><b style='color: #ffffff;'>Generic Letters (GL):</b> Address specific technical or prehistoric issues that have a broad impact on the industry, often requesting specific actions or responses from licensees.</li>
                        <li><b style='color: #ffffff;'>Information Notices (IN):</b> Used to bring significant recently identified safety or environmental information to the attention of the industry.</li>
                        <li><b style='color: #ffffff;'>Bulletins (BL):</b> Urgent notices requiring specific action and a formal written response regarding critical safety issues.</li>
                        <li><b style='color: #ffffff;'>Regulatory Issue Summaries (RIS):</b> Communicate administrative technical information or provide clarification on existing regulations.</li>
                    </ul>
                    <p style='margin-bottom: 8px;'><strong style='color: #2ecc71;'>🔍 What to Look For:</strong></p>
                    <p style='margin-left: 5px;'>
                        When searching, use specific identifiers like <i>"IN 2023-01"</i> or <i>"GL 96-06"</i> to find technical evaluations regarding material degradation, cooling systems, or safety protocols.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Ask Specific Questions Guidance Section
    with st.expander("💡 Tips for Better Search Results", expanded=False):
        st.markdown("""
        <div style='background: linear-gradient(145deg, rgba(155, 89, 182, 0.1), rgba(138, 43, 226, 0.05)); 
                    border-left: 4px solid #9b59b6; 
                    padding: 16px 20px; 
                    border-radius: 8px;
                    margin: 10px 0;'>
            <h4 style='color: #bb86fc; margin-top: 0; margin-bottom: 12px;'>🎯 Ask Specific Questions for Best Results</h4>
            <p style='color: #b0b0c0; font-size: 0.95rem; margin-bottom: 12px;'>
                The more specific your question, the more accurate and relevant the results will be.
            </p>
            <div style='color: #a0a0b0; font-size: 0.9rem;'>
                <p style='margin-bottom: 8px;'><strong style='color: #f39c12;'>✅ Good Examples:</strong></p>
                <ul style='margin-left: 20px; margin-bottom: 12px;'>
                    <li>"What are the corrosion rate requirements for carbon steel in PWR containments?"</li>
                    <li>"Where can I find the Pourbaix diagram guidelines for nickel alloys?"</li>
                    <li>"What does NRC say about stress corrosion cracking in austenitic stainless steel?"</li>
                </ul>
                <p style='margin-bottom: 8px;'><strong style='color: #e74c3c;'>❌ Avoid Vague Questions:</strong></p>
                <ul style='margin-left: 20px;'>
                    <li>"Tell me about corrosion" (too broad)</li>
                    <li>"Steel information" (not specific enough)</li>
                    <li>"Everything about nuclear" (too general)</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Query input section
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "🔍 Enter your question:",
            placeholder="e.g., Where can I find info on predominance area diagram for a metal?",
            key="query_input"
        )
    
    with col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        search_button = st.button("🚀 Search Documents", use_container_width=True)
    
    st.markdown("---")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'selected_doc' not in st.session_state:
        st.session_state.selected_doc = None
    
    # Process query
    if search_button and user_question:
        with st.spinner("🔄 Searching through NRC documents..."):
            try:
                response = run_custom_rag(user_question)
                
                # Parse JSON response
                # Try to extract JSON from the response
                try:
                    # Clean the response if wrapped in markdown code blocks
                    clean_response = response.strip()
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    if clean_response.startswith("```"):
                        clean_response = clean_response[3:]
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]
                    
                    st.session_state.results = json.loads(clean_response.strip())
                except json.JSONDecodeError:
                    st.error("⚠️ Failed to parse response. Raw response:")
                    st.code(response)
                    st.session_state.results = None
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.results = None
    
    # Display results
    if st.session_state.results:
        results = st.session_state.results
        
        if "error" in results:
            st.warning(f"⚠️ {results['error']}")
        elif "references" in results and results["references"]:
            st.markdown(f"### 📋 Found {len(results['references'])} Relevant Reference(s)")
            
            # Create columns for results
            for idx, ref in enumerate(results["references"]):
                st.markdown(f"""
                <div class="result-card">
                    <div class="document-name">📄 {ref.get('document_name', 'Unknown Document')}</div>
                    <div class="section-number">📍 Section: {ref.get('section_number', 'N/A')}</div>
                    <div class="relevance-summary">{ref.get('relevance_summary', 'No summary available.')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add download button for document
                doc_name = ref.get('document_name', '')
                if doc_name:
                    # Find PDF file
                    data_folder = os.path.join(os.path.dirname(__file__), "Data")
                    pdf_path = find_pdf_file(doc_name, data_folder)
                    
                    if pdf_path and os.path.exists(pdf_path):
                        pdf_info = get_pdf_file_info(pdf_path)
                        if pdf_info:
                            col_info, col_download = st.columns([3, 1])
                            with col_info:
                                st.markdown(f"""
                                <div style='background: rgba(155, 89, 182, 0.1); 
                                            padding: 10px 15px; 
                                            border-radius: 8px; 
                                            border-left: 3px solid #9b59b6;
                                            margin: 5px 0;'>
                                    <span style='color: #e0d4ff;'>📁 File: <strong>{pdf_info['name']}</strong></span>
                                    <span style='color: #888; margin-left: 15px;'>Size: {pdf_info['size_formatted']}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_download:
                                with open(pdf_path, "rb") as file:
                                    st.download_button(
                                        label="📥 Download PDF",
                                        data=file,
                                        file_name=pdf_info['name'],
                                        mime="application/pdf",
                                        key=f"download_btn_{idx}",
                                        use_container_width=True
                                    )
                    else:
                        st.markdown(f"""
                        <div style='background: rgba(231, 76, 60, 0.1); 
                                    padding: 10px 15px; 
                                    border-radius: 8px; 
                                    border-left: 3px solid #e74c3c;
                                    margin: 5px 0;'>
                            <span style='color: #e74c3c;'>⚠️ PDF file not found for: {doc_name}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("ℹ️ No references found. Try rephrasing your question.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.85rem; padding: 20px;'>
        <p>NRC Technical Document Query System | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
