import os  # keys are read from the environment (see .env.example)
import requests
from llama_cloud_services import LlamaCloudIndex

# --- Your Existing LLM Function ---
def get_core42_response(role: str, content: str, system_instruction: str = None) -> str:
    """Sends a message to the Core42 API."""
    API_KEY = os.environ.get("CORE42_API_KEY", "")  # moved to .env - never hard-code secrets
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

# --- Helper Functions ---
def clean_json_response(response: str) -> str:
    """Remove markdown code fences from JSON responses."""
    import re
    # Remove ```json and ``` markers
    cleaned = re.sub(r'^```json\s*\n?', '', response.strip())
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    return cleaned.strip()


# --- LlamaCloud Integration Logic ---

# STEP 1: Identify Relevant Section(s)
def identify_relevant_sections(query: str, index):
    """First pass: identify which sections contain relevant information."""
    # Retrieve initial chunks to identify relevant sections
    nodes = index.as_retriever(similarity_top_k=5).retrieve(query)
    
    # Combine text AND metadata from all retrieved nodes
    context_text = ""
    for i, node_with_score in enumerate(nodes):
        chunk_content = node_with_score.node.get_content()
        metadata = node_with_score.node.metadata 
        
        context_text += f"\n--- Source {i+1} ---\n"
        context_text += f"METADATA: {metadata}\n"
        context_text += f"CONTENT: {chunk_content}\n"

    # Prepare the prompt to identify sections
    system_prompt = (
        "You are a technical document assistant for nuclear energy regulations. "
        "Your task is to identify the specific document name and section number "
        "that contains the information relevant to the user's question, and provide "
        "a detailed answer based on the available context. "
        "\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the provided context chunks.\n"
        "2. Extract the 'Document Name' and 'Section Number' for relevant sources.\n"
        "3. Provide a comprehensive, detailed answer to the user's question based on the context.\n"
        "4. Return ONLY a valid JSON object. Do not include any conversational text.\n"
        "5. If the answer is not in the context, return: {\"error\": \"Information not found in available documents.\"}\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "{\n"
        "  \"references\": [\n"
        "    {\"document_name\": \"string\", \"section_number\": \"string\", \"detailed_answer\": \"string\"}\n"
        "  ]\n"
        "}"
    )
    
    final_user_content = f"CONTEXT:\n{context_text}\n\nUSER QUESTION: {query}"
    
    return get_core42_response(
        role="user", 
        content=final_user_content, 
        system_instruction=system_prompt
    )


# STEP 2: Retrieve Full Section Text
def retrieve_full_section_text(section_number: str, document_name: str, index):
    """Second pass: retrieve all chunks from the identified section to get complete text."""
    try:
        # Retrieve many chunks filtered by section metadata
        # Note: The metadata key might vary - common ones are 'section_number', 'Section', 'page_label'
        retriever = index.as_retriever(similarity_top_k=50)
        
        # Retrieve all nodes and filter manually by section
        # (LlamaCloud filtering syntax may vary based on your index configuration)
        all_nodes = retriever.retrieve(f"section {section_number}")
        
        # Filter nodes that match the section number in metadata
        section_nodes = []
        for node_with_score in all_nodes:
            metadata = node_with_score.node.metadata
            # Check various possible metadata keys for section number
            node_section = metadata.get('section_number') or metadata.get('Section') or metadata.get('page_label')
            node_doc = metadata.get('file_name') or metadata.get('document_name') or metadata.get('Document')
            
            # Match section and document
            if (node_section and section_number in str(node_section)) and \
               (not document_name or (node_doc and document_name in str(node_doc))):
                section_nodes.append(node_with_score)
        
        # Combine all chunks from the section
        full_text = "\n\n".join([node.node.get_content() for node in section_nodes])
        
        return {
            "section_number": section_number,
            "document_name": document_name,
            "full_text": full_text,
            "chunk_count": len(section_nodes)
        }
    
    except Exception as e:
        return {
            "error": f"Failed to retrieve full section text: {str(e)}",
            "section_number": section_number,
            "document_name": document_name
        }


# COMBINED TWO-STEP RAG
def run_custom_rag(query: str):
    """Two-step RAG: 1) Identify sections, 2) Retrieve full section text."""
    import json
    
    # Initialize the LlamaCloud Index
    index = LlamaCloudIndex(
      name="nrc",
      project_name="Default",
      organization_id=os.environ.get("LLAMA_CLOUD_ORGANIZATION_ID", ""),
      api_key=os.environ.get("LLAMA_CLOUD_API_KEY", ""),
    )
    
    print("STEP 1: Identifying relevant sections...")
    section_identification = identify_relevant_sections(query, index)
    print(f"Section Identification Result:\n{section_identification}\n")
    
    # Parse the JSON response
    try:
        # Clean the response to remove markdown code fences
        cleaned_response = clean_json_response(section_identification)
        section_data = json.loads(cleaned_response)
        
        if "error" in section_data:
            return section_data
        
        # Step 2: Retrieve full text for each identified section
        print("STEP 2: Retrieving full section text...")
        full_sections = []
        
        for ref in section_data.get("references", []):
            section_num = ref.get("section_number", "")
            doc_name = ref.get("document_name", "")
            
            print(f"  - Retrieving: {doc_name} - Section {section_num}")
            full_section = retrieve_full_section_text(section_num, doc_name, index)
            full_section["detailed_answer"] = ref.get("detailed_answer", "")
            full_sections.append(full_section)
        
        return {
            "query": query,
            "identified_sections": section_data["references"],
            "full_sections": full_sections
        }
    
    except json.JSONDecodeError as e:
        return {
            "error": "Failed to parse section identification response",
            "raw_response": section_identification,
            "parse_error": str(e)
        }


# --- Example Usage ---
if __name__ == "__main__":
    user_question = "where can i find info on predominance area diagram for a metal, chlorine, oxygen solute"
    result = run_custom_rag(user_question)
    
    import json
    print("\n" + "="*80)
    print("FINAL RESULT:")
    print("="*80)
    print(json.dumps(result, indent=2))