import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import os
import base64
from llama_cloud_services import LlamaCloudIndex

# --- Page Configuration ---
st.set_page_config(
    page_title="NRC Document Query System",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar collapsed by default
)

# --- Custom CSS for Dark Purple Theme (No Sidebar) ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main Background - Corporate Deep Teal/Dark */
    .stApp {
        background: linear-gradient(135deg, #002b2d 0%, #004044 50%, #005055 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
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
    
    /* FAQ Card Styling */
    .faq-card {
        background: rgba(0, 43, 45, 0.6);
        border: 1px solid rgba(0, 163, 173, 0.3);
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
    }
    
    .faq-question {
        color: #00A3AD;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 8px;
    }
    
    .faq-answer {
        color: #b0b0c0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Info Banner */
    .info-banner {
        background: linear-gradient(135deg, rgba(0, 163, 173, 0.15), rgba(140, 198, 63, 0.1));
        border: 1px solid rgba(0, 163, 173, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* PDF Viewer Container */
    .pdf-viewer-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(0, 163, 173, 0.3);
        border-radius: 12px;
        padding: 10px;
        margin: 15px 0;
    }
    
    /* Detailed Text Styling */
    .detailed-text {
        background: rgba(0, 31, 33, 0.6);
        border-left: 3px solid #00A3AD;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    .excerpt-text {
        background: rgba(140, 198, 63, 0.1);
        border-left: 3px solid #8CC63F;
        padding: 12px 15px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-style: italic;
        color: #d0d0d0;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 43, 45, 0.6);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #b0b0c0;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00A3AD !important;
    }
</style>
""", unsafe_allow_html=True)


# --- API Functions ---
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
    data = {"model": "gpt-4o", "stream": False, "messages": messages, "temperature": 0.1}

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"


def run_custom_rag(query: str):
    """Run RAG query using LlamaCloud with enhanced detail extraction."""
    index = LlamaCloudIndex(
        name="nrc",
        project_name="Default",
        organization_id="c9e1176c-fd12-4819-bfd7-fc23f80ab9b5",
        api_key="llx-7GWrpifAsJ6UJYIPoVa9jl2tSRwmeIOpbi4Hcj5ynctYzJ4B",
    )

    nodes = index.as_retriever(similarity_top_k=5).retrieve(query)
    
    context_text = ""
    raw_chunks = []  # Store raw chunks for display
    for i, node_with_score in enumerate(nodes):
        chunk_content = node_with_score.node.get_content()
        metadata = node_with_score.node.metadata 
        
        context_text += f"\n--- Source {i+1} ---\n"
        context_text += f"METADATA: {metadata}\n"
        context_text += f"CONTENT: {chunk_content}\n"
        
        # Store for display
        raw_chunks.append({
            "source_num": i + 1,
            "content": chunk_content,
            "metadata": metadata,
            "score": node_with_score.score if hasattr(node_with_score, 'score') else None
        })

    # Enhanced system prompt requesting more details
    system_prompt = (
        "You are a technical document assistant for nuclear energy regulations. "
        "Your task is to identify the specific document name and section number "
        "that contains the information relevant to the user's question, and provide "
        "comprehensive details about the content found. "
        "\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the provided context chunks thoroughly.\n"
        "2. Extract the 'Document Name' and 'Section Number' for relevant sources.\n"
        "3. Provide a DETAILED relevance_summary (3-5 sentences) explaining:\n"
        "   - What specific information was found\n"
        "   - Why this section is relevant to the user's question\n"
        "   - What technical concepts or data points are covered\n"
        "4. Extract 2-3 key_excerpts: direct quotes from the text that are most relevant.\n"
        "5. Provide technical_context: brief explanation of the technical significance.\n"
        "6. Return ONLY a valid JSON object. Do not include any conversational text.\n"
        "7. If the answer is not in the context, return: {\"error\": \"Information not found in available documents.\"}\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "{\n"
        "  \"references\": [\n"
        "    {\n"
        "      \"document_name\": \"string\",\n"
        "      \"section_number\": \"string\",\n"
        "      \"relevance_summary\": \"Detailed 3-5 sentence explanation...\",\n"
        "      \"key_excerpts\": [\"Direct quote 1...\", \"Direct quote 2...\"],\n"
        "      \"technical_context\": \"Brief technical significance explanation\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )
    
    final_user_content = f"CONTEXT:\n{context_text}\n\nUSER QUESTION: {query}"

    response = get_core42_response(
        role="user", 
        content=final_user_content, 
        system_instruction=system_prompt
    )
    
    return response, raw_chunks


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


def open_pdf_in_system_viewer(pdf_path: str):
    """Open the PDF file using the default system viewer (Windows)."""
    try:
        if os.name == 'nt':  # Check if running on Windows
            os.startfile(pdf_path)
            return True, "File opened successfully."
        else:
            # Fallback for other OS if needed in future (e.g., Mac/Linux)
            # import subprocess
            # subprocess.call(('open', pdf_path))
            return False, "Auto-open is currently supported only on Windows."
    except Exception as e:
        return False, f"Error opening file: {str(e)}"


# --- Main App ---
def main():
    # Display logo at the top left
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    
    # Main content
    st.markdown("# NRC Technical Document Query System")
    st.markdown("<p style='color: #b0b0c0; font-size: 1.1rem; margin-bottom: 30px;'>Search through nuclear regulatory documents using natural language queries</p>", unsafe_allow_html=True)

    # Info Banner
    st.markdown("""
    <div class="info-banner">
        <div style='display: flex; justify-content: space-between; flex-wrap: wrap; gap: 20px;'>
            <div style='flex: 1; min-width: 200px;'>
                <h4 style='color: #00A3AD; margin: 0 0 10px 0;'>⚛️ About This Tool</h4>
                <p style='color: #b0b0c0; font-size: 0.9rem; margin: 0;'>
                    AI-powered semantic search through NRC Technical Letters and regulatory documents.
                </p>
            </div>
            <div style='flex: 1; min-width: 200px;'>
                <h4 style='color: #8CC63F; margin: 0 0 10px 0;'>🔍 How to Use</h4>
                <p style='color: #b0b0c0; font-size: 0.9rem; margin: 0;'>
                    1. Enter your question → 2. Click Search → 3. Review results → 4. View or download PDF
                </p>
            </div>
            <div style='flex: 1; min-width: 200px;'>
                <h4 style='color: #f39c12; margin: 0 0 10px 0;'>⚡ Powered By</h4>
                <p style='color: #b0b0c0; font-size: 0.9rem; margin: 0;'>
                    LlamaCloud RAG + AI (GPT-4o)
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
                        <li><b style='color: #ffffff;'>Generic Letters (GL):</b> Address specific technical issues with broad industry impact.</li>
                        <li><b style='color: #ffffff;'>Information Notices (IN):</b> Bring significant safety or environmental information to the industry's attention.</li>
                        <li><b style='color: #ffffff;'>Bulletins (BL):</b> Urgent notices requiring specific action regarding critical safety issues.</li>
                        <li><b style='color: #ffffff;'>Regulatory Issue Summaries (RIS):</b> Communicate administrative technical information or clarifications.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Tips Section
    with st.expander("💡 Tips for Better Search Results", expanded=False):
        st.markdown("""
        <div style='background: linear-gradient(145deg, rgba(155, 89, 182, 0.1), rgba(138, 43, 226, 0.05)); 
                    border-left: 4px solid #9b59b6; 
                    padding: 16px 20px; 
                    border-radius: 8px;
                    margin: 10px 0;'>
            <h4 style='color: #bb86fc; margin-top: 0; margin-bottom: 12px;'>🎯 Ask Specific Questions for Best Results</h4>
            <div style='color: #a0a0b0; font-size: 0.9rem;'>
                <p style='margin-bottom: 8px;'><strong style='color: #2ecc71;'>✅ Good Examples:</strong></p>
                <ul style='margin-left: 20px; margin-bottom: 12px;'>
                    <li>"In Thermodynamic Reference Electrodes what is a well defined chemical RedOx couple."</li>
                    <li>"What does the NRC say about the Production of Chloride-Based Salt Fuel "</li>
                    <li>"What does NRC say about stress corrosion cracking in austenitic stainless steel?"</li>
                </ul>
                <p style='margin-bottom: 8px;'><strong style='color: #e74c3c;'>❌ Avoid Vague Questions:</strong></p>
                <ul style='margin-left: 20px;'>
                    <li>"Tell me about corrosion" (too broad)</li>
                    <li>"Steel information" (not specific enough)</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # FAQs Section
    with st.expander("❓ Frequently Asked Questions (FAQs)", expanded=False):
        st.markdown("""
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;'>
            <div class="faq-card">
                <div class="faq-question">🔎 What types of documents are searchable?</div>
                <div class="faq-answer">This tool searches through NRC Technical Letters including Generic Letters (GL), Information Notices (IN), Bulletins (BL), and Regulatory Issue Summaries (RIS). All documents are indexed and searchable via semantic search.</div>
            </div>
            <div class="faq-card">
                <div class="faq-question">🤖 How does the semantic search work?</div>
                <div class="faq-answer">The system uses LlamaCloud's RAG (Retrieval-Augmented Generation) to understand the meaning of your question, not just keywords. It finds the most relevant document sections and uses GPT-4o to provide detailed answers.</div>
            </div>
            <div class="faq-card">
                <div class="faq-question">📝 How should I phrase my questions?</div>
                <div class="faq-answer">Be specific! Include technical terms, material types, or NRC document identifiers (e.g., "IN 2023-01"). Questions like "What does NRC say about X in Y context?" work best.</div>
            </div>
            <div class="faq-card">
                <div class="faq-question">📄 How can I view the source PDF documents?</div>
                <div class="faq-answer">After searching, each result includes options to view the PDF in the app, open it in your system viewer (best for large files), or download it.</div>
            </div>
            <div class="faq-card">
                <div class="faq-question">🔒 Are my queries stored or logged?</div>
                <div class="faq-answer">Your queries are processed in real-time and are not permanently stored. The system only keeps temporary session data for displaying results.</div>
            </div>
            <div class="faq-card">
                <div class="faq-question">📊 What information is shown in results?</div>
                <div class="faq-answer">Results include: document name, section number, detailed relevance summary, key excerpts from the document, technical context, and the ability to view/download the source PDF.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
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
    if 'raw_chunks' not in st.session_state:
        st.session_state.raw_chunks = None
    if 'selected_pdf' not in st.session_state:
        st.session_state.selected_pdf = None
    
    # Process query
    if search_button and user_question:
        with st.spinner("🔄 Searching through NRC documents..."):
            try:
                response, raw_chunks = run_custom_rag(user_question)
                st.session_state.raw_chunks = raw_chunks
                
                # Parse JSON response
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
            
            # Create results for each reference
            for idx, ref in enumerate(results["references"]):
                st.markdown(f"""
                <div class="result-card">
                    <div class="document-name">📄 {ref.get('document_name', 'Unknown Document')}</div>
                    <div class="section-number">📍 Section: {ref.get('section_number', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display detailed relevance summary
                relevance_summary = ref.get('relevance_summary', 'No summary available.')
                st.markdown(f"""
                <div class="detailed-text">
                    <strong style='color: #00A3AD;'>📝 Summary:</strong><br/>
                    <span style='color: #d0d0d0;'>{relevance_summary}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Display key excerpts if available
                key_excerpts = ref.get('key_excerpts', [])
                if key_excerpts:
                    st.markdown("<p style='color: #8CC63F; font-weight: 600; margin-top: 15px;'>📌 Key Excerpts from Document:</p>", unsafe_allow_html=True)
                    for excerpt in key_excerpts:
                        st.markdown(f"""
                        <div class="excerpt-text">
                            "{excerpt}"
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display technical context if available
                technical_context = ref.get('technical_context', '')
                if technical_context:
                    st.markdown(f"""
                    <div style='background: rgba(243, 156, 18, 0.1); border-left: 3px solid #f39c12; padding: 12px 15px; border-radius: 0 8px 8px 0; margin: 10px 0;'>
                        <strong style='color: #f39c12;'>🔬 Technical Context (ALWAYS VALIDATE with the document):</strong><br/>
                        <span style='color: #d0d0d0;'>{technical_context}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # PDF Section with tabs for View/Download
                doc_name = ref.get('document_name', '')
                if doc_name:
                    data_folder = os.path.join(os.path.dirname(__file__), "Data")
                    pdf_path = find_pdf_file(doc_name, data_folder)
                    
                    if pdf_path and os.path.exists(pdf_path):
                        pdf_info = get_pdf_file_info(pdf_path)
                        if pdf_info:
                            st.markdown(f"""
                            <div style='background: rgba(155, 89, 182, 0.1); 
                                        padding: 10px 15px; 
                                        border-radius: 8px; 
                                        border-left: 3px solid #9b59b6;
                                        margin: 10px 0;'>
                                <span style='color: #e0d4ff;'>📁 File: <strong>{pdf_info['name']}</strong></span>
                                <span style='color: #888; margin-left: 15px;'>Size: {pdf_info['size_formatted']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Define tabs
                            tab1, tab2 = st.tabs(["↗️ Open in System Viewer", "📥 Download PDF"])

                            # TAB 1: System Viewer
                            with tab1:
                                st.markdown("<p style='color: #b0b0c0; font-size: 0.9rem;'>Click below to open the PDF directly in your computer's default PDF viewer (e.g., Adobe Acrobat, Edge).</p>", unsafe_allow_html=True)
                                if st.button(f"↗️ Open {pdf_info['name']}", key=f"open_sys_{idx}"):
                                    success, msg = open_pdf_in_system_viewer(pdf_path)
                                    if success:
                                        st.success(f"✅ {msg}")
                                    else:
                                        st.error(f"❌ {msg}")

                            # TAB 3: Download Button
                            with tab2:
                                with open(pdf_path, "rb") as file:
                                    st.download_button(
                                        label="📥 Download PDF",
                                        data=file,
                                        file_name=pdf_info['name'],
                                        mime="application/pdf",
                                        key=f"download_btn_{idx}",
                                        use_container_width=True
                                    )
                                st.info("💡 Click the button above to download the PDF to your device.")
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
            
            # Raw Chunks Section (expandable)
            if st.session_state.raw_chunks:
                with st.expander("🔍 View Retrieved Chunks (Advanced)", expanded=False):
                    st.markdown("<p style='color: #888; font-size: 0.9rem;'>These are the raw text chunks retrieved from the document index:</p>", unsafe_allow_html=True)
                    for chunk in st.session_state.raw_chunks:
                        st.markdown(f"""
                        <div style='background: rgba(0, 31, 33, 0.8); border: 1px solid rgba(0, 163, 173, 0.2); border-radius: 8px; padding: 15px; margin: 10px 0;'>
                            <div style='color: #00A3AD; font-weight: 600; margin-bottom: 8px;'>Source {chunk['source_num']}</div>
                            <div style='color: #888; font-size: 0.8rem; margin-bottom: 8px;'>Metadata: {json.dumps(chunk['metadata'], indent=2)}</div>
                            <div style='color: #d0d0d0; font-size: 0.9rem; border-top: 1px solid rgba(0, 163, 173, 0.2); padding-top: 10px;'>{chunk['content'][:500]}{'...' if len(chunk['content']) > 500 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No references found. Try rephrasing your question.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.85rem; padding: 20px;'>
        <p>NRC Technical Document Query System | Built with Streamlit | AI-Powered Search</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
