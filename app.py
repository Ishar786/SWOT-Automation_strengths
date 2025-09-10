# --- IMPORTS AND CONFIGURATION ---
import streamlit as st
import google.generativeai as genai
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import re
import time
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Expert Strength Analyzer",
    page_icon="📄",
    layout="wide"
)

# --- STEP 1: DEFINE STYLE GUIDE & INDUSTRY MAPPING ---

# The complete list of strength categories derived from your examples.
STRENGTH_CATEGORIES = [
    "R&D Focus", "Financial Strengths", "Credit Ratings", "Brand Value",
    "Product/Service Spread", "Market Share & Market Position",
    "Geographic Reach & Channels", "Customer Base & Digital Strength",
    "Orderbook Health", "CET Values and Performances", "Subscriber Base",
    "Diversified Business Segments", "Operational Network"
]

# This is the complete and definitive style guide, built from all 13 of your verbatim examples.
# This will be used in the final generation step to ensure perfect formatting.
WRITING_STYLE_GUIDE = {
    "R&D Focus": "PetroChina has one of the best-in-class R&D that could give the company a competitive advantage. In FY2024, the group has incurred R&D expenses of RMB23,014 million (~$3,197.89 million), which grew ~4.8%  year on year (YoY). As of December 2024, the group had 84 research institutes, 21 national R&D institutions and 54 company-level key laboratories. In FY2024, the group owns a total of ~20,000 patents obtained in China and overseas. Such best-in-class R&D could help gain a competitive advantage.",
    "Financial Strengths": "China Mobile’s revenue was recorded at RMB1,040,759 million (~$144,617.9 million) in FY2024, which increased by ~3.1% YoY from RMB1,009,309 million (~$142,532 million) in FY2023. The group recorded the profit for the year at RMB138,526 million (~$19,248.8 million) in FY2024, against RMB131,935 (~$18,631.5 million) in FY2023, indicating a growth of ~5% YoY. Such an increase in the financial performance of China Mobile could enhance its market position and boost the investors’ confidence.",
    "Credit Ratings": "As of December 2024, CCB’s long-term rating was rated A with a stable outlook by Standard & Poor’s Global Ratings (S&P), A1 with a stable outlook by Moody’s Corporation (Moody’s), and an A with a stable outlook from Fitch Ratings Inc. (Fitch). In 2025, the group was ranked at #12 in the BrandFinance global 500. Thus, stable credit ratings would help the group boost investment grade profile.",
    "Brand Value": "China Mobile has a strong brand equity evident through its recognition by various leading evaluation players. The group’s mobile customers stood at 1,004 million in FY2024, from 991 million in FY2023, which increased ~1.3% YoY. In 2025, the group was ranked at #27 position by Brand Finance in the BrandFinance Global 500 (100) list.",
    "Product/Service Spread": "CCB has four reportable business segments that could boost the group’s operations through their wide range of product portfolio. In FY2024 (January – December 2024), the Personal Finance Business segment (which accounted for ~49.3% of the group’s total operating income) provides a range of financial products and services to individual customers. The Corporate Finance Business segment (~31.3%) serves corporations, government agencies and financial institutions. The Treasury and Asset Management Business segment (~16.3%) includes the group’s treasury operations. The Others segment (~3.1%) covers equity investments. Having such strong business segment performance and product/service offerings could boost financial flexibility.",
    "Market Share & Market Position": "POSCO maintains a dominant position in the Korean steel market, reinforcing its leadership across multiple product categories. In FY2024 (January – December 2024), POSCO’s total market share of steel products in Korea stood at ~46.0%, an increase from ~45.5% in FY2023. For instance, its cold rolled products achieved a domestic market share of ~56%, while its hot rolled products commanded a ~52% market share. Such a leading market position enhances the company’s competitive profile and strengthens its bargaining power.",
    "Geographic Reach & Channels": "TJX remains one of the largest off-price retailers with an integrated network of 5,085 stores as of January 2025. The company’s T.J. Maxx and Marshalls chains are together the largest off-price retailers in the US (with a total of 2,563 stores). The company also operated ~30 million square feet (sq. ft.) of distribution centers across six countries. Such a robust distribution system and network of stores would enable TJX to serve its customers effectively by attracting a huge customer base.",
    "Customer Base & Digital Strength": "In FY2024, AT&T served 141 million wireless subscribers in North America, with ~118 million subscribers in the US. As of December 2024, its network covers more than 314 million people with 5G technology in the US and North America. AT&T served  ~15.3 million customer locations by offering broadband and internet services, along with ~9.3 million fiber broadband connections.",
    "Orderbook Health": "Accenture reported new order bookings of $81.2 billion in FY2024 (September 2023 – August 2024) which increased by ~12.5% year-on-year (YoY) from $72.2 billion in FY2023. Accenture’s new order bookings comprised $37 billion of consulting bookings and $44.2 billion of managed services bookings.",
    "CET Values and Performances": "In FY2023, Absa group reported a Common Equity Tier 1 (CET 1) ratio of 12.5%. The group also reported a Tier 1 capital ratio of 13.8% and a total capital ratio of 15.4% in FY2023. According to the transitional Basel III framework, the minimum requirement of the CET1 capital ratio is 8.5%. Such adequate capital requirements could mitigate business risks.",
    "Subscriber Base": "In FY2024 (January – December 2024), AT&T served 141 million wireless subscribers in North America, with ~118 million subscribers in the US. The company’s mobile customers stood at over 118 million, which include 19 million prepaid subscribers, 89 million postpaid subscribers, and 10 million resellers.",
    "Diversified Business Segments": "AT&T has a diversified business segment portfolio. Mobility segment delivers nationwide wireless voice and data services. The Business Wireline segment provides advanced connectivity solutions such as AT&T Dedicated Internet and fiber ethernet. Consumer Wireline segment delivers broadband services through fiber connections and AT&T Internet Air (AIA). Such diversified business segments could enhance revenue stability.",
    "Operational Network": "As of December 2024, BOC had ~11,507 institutions globally, including ~10,279 institutions in the Chinese mainland and ~543 institutions in Hong Kong, Taiwan, Macao, and other countries and regions. In FY2024, BOC continued to enhance the competitiveness of its domestic institutions in green finance by selecting 28 tier-1 and tier-2 branches as model institutions in green finance and establishing a total of 456 specialized green finance outlets."
}

INDUSTRY_KPI_PRIORITY = {
    "Banking & Financial Services": ["CET Values and Performances", "Financial Strengths", "Credit Ratings", "Product/Service Spread", "Operational Network"],
    "Technology & Telecom": ["Customer Base & Digital Strength", "Subscriber Base", "R&D Focus", "Financial Strengths", "Brand Value"],
    "Retail & Consumer Goods": ["Geographic Reach & Channels", "Market Share & Market Position", "Financial Strengths", "Customer Base & Digital Strength"],
    "Manufacturing & Industrials": ["Market Share & Market Position", "R&D Focus", "Orderbook Health", "Product/Service Spread"],
    "Energy & Resources": ["Market Share & Market Position", "Financial Strengths", "R&D Focus", "Geographic Reach & Channels"],
    "Professional Services": ["Orderbook Health", "Financial Strengths", "Brand Value"],
    "Healthcare & Pharmaceuticals": ["R&D Focus", "Product/Service Spread", "Brand Value", "Financial Strengths"],
    "Automotive": ["Geographic Reach & Channels", "Market Share & Market Position", "Financial Strengths", "R&D Focus"]
}


# --- CORE LANGCHAIN FUNCTIONS ---

def load_pdf_chunks(pdf_file):
    """Splits the uploaded PDF into manageable chunks."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_file.getvalue())
            tmp_pdf_path = tmpfile.name
        
        loader = PyMuPDFLoader(tmp_pdf_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        os.remove(tmp_pdf_path)
        return chunks
    except Exception as e:
        st.error(f"Error loading or splitting PDF: {e}")
        return []

def get_strengths_analysis(chunks, company_name, industry, api_key):
    """Performs the full Classify -> Rank -> Generate pipeline for Strengths."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=api_key, convert_system_message_to_human=True)

    # --- Chain 1: "Mega-Batch" Classifier ---
    mega_batch_classifier_prompt = PromptTemplate(
        input_variables=["text_batch", "categories"],
        template="""You are an expert document analyst. For each numbered text chunk below, identify the single best matching category from the provided list.
        Your response MUST be a valid JSON object where keys are the chunk numbers (e.g., "Chunk 1") and values are the category name or 'None'. Do not add any commentary before or after the JSON.

        Categories: {categories}
        ---
        {text_batch}
        ---
        JSON Response:"""
    )
    mega_batch_classifier_chain = LLMChain(llm=llm, prompt=mega_batch_classifier_prompt)

    # --- Chain 2: Generator ---
    generator_prompt = PromptTemplate(
        input_variables=["company_name", "text", "style_example"],
        template="""As an expert business analyst, analyze the following source text about {company_name} and write a detailed, data-rich paragraph. Your writing MUST perfectly mimic the narrative style of the provided example. Start with a strategic claim, support it with hard numbers (financials, operational stats, dates, rankings), and end with a sentence explaining the business impact.

        **STYLE GUIDE EXAMPLE TO MIMIC PERFECTLY:**
        "{style_example}"
        
        **Source Text to Analyze:**
        "{text}"
        
        **Generated Paragraph:**
        """
    )
    generator_chain = LLMChain(llm=llm, prompt=generator_prompt)

    # --- OPTIMIZATION: Pre-filter chunks ---
    st.write("🔍 Pre-filtering chunks for relevant keywords...")
    keywords = ['revenue', 'profit', 'margin', 'financial', 'market share', 'rankings', 'stores', 'locations', 'subscribers', 'users', 'patents', 'R&D', 'order book', 'bookings', 'CET1', 'capital ratio']
    relevant_chunks = [
        chunk for chunk in chunks 
        if any(re.search(r'\b' + keyword + r'\b', chunk.page_content, re.IGNORECASE) for keyword in keywords)
    ]
    st.info(f"Found {len(relevant_chunks)} potentially relevant chunks out of {len(chunks)} total.")

    # --- Pipeline Execution: "Mega-Batch" Classification ---
    st.write("🔍 Classifying all relevant chunks in a single, efficient API call...")
    classified_strengths = {}
    
    # Format all relevant chunks into one large string for a single API call
    formatted_mega_batch = "\n---\n".join([f'Chunk {i+1}:\n"""{chunk.page_content}"""' for i, chunk in enumerate(relevant_chunks)])
    
    if len(formatted_mega_batch) > 900000: # Leave a buffer for the prompt template
        st.error(f"The filtered content ({len(formatted_mega_batch)} characters) is too large for the model's 1 million token context window. Please try a smaller document.")
        return None

    try:
        response_text = mega_batch_classifier_chain.invoke({
            "text_batch": formatted_mega_batch,
            "categories": ", ".join(STRENGTH_CATEGORIES)
        })['text']
        
        clean_response = response_text.strip().replace("```json", "").replace("```", "")
        classifications = json.loads(clean_response)
        
        for chunk_num_str, category in classifications.items():
            chunk_index = int(re.search(r'\d+', chunk_num_str).group()) - 1
            if 0 <= chunk_index < len(relevant_chunks):
                if category != "None" and category in STRENGTH_CATEGORIES:
                    if category not in classified_strengths:
                        classified_strengths[category] = []
                    classified_strengths[category].append(relevant_chunks[chunk_index].page_content)
        st.success("Classification successful.")

    except Exception as e:
        st.error(f"CRITICAL ERROR during the single classification call: {e}.")
        st.warning("The process cannot continue. This might be due to an API rate limit, an extremely large document, or an invalid API Key.")
        return None
    
    # --- Rank and Select Top 2 Strengths ---
    st.write("📊 Ranking findings based on industry priorities...")
    priority_strengths = INDUSTRY_KPI_PRIORITY.get(industry, STRENGTH_CATEGORIES)
    final_strength_chunks = []
    found_categories = []
    for cat in priority_strengths:
        if cat in classified_strengths:
            combined_text = "\n\n".join(classified_strengths[cat])
            final_strength_chunks.append({"category": cat, "text": combined_text})
            found_categories.append(cat)
        if len(final_strength_chunks) == 2:
            break
    
    st.info(f"Selected top categories for {industry}: {found_categories}")

    # --- Generate Final Output ---
    st.write("✍️ Generating final analysis...")
    strengths_output = ""
    for item in final_strength_chunks:
        category = item["category"]
        text = item["text"]
        style_example = WRITING_STYLE_GUIDE.get(category, "Write a detailed, data-rich paragraph.")
        
        generated_result = generator_chain.invoke({
            "company_name": company_name,
            "text": text,
            "style_example": style_example
        })
        strengths_output += generated_result['text'] + "\n\n"

    return strengths_output.strip()

# --- STREAMLIT USER INTERFACE ---
st.title("📄 Expert Strength Analyzer")
st.markdown("This tool uses a multi-step AI pipeline (**Classify -> Rank -> Generate**) to analyze an annual report and extract the two most relevant **Strengths** in a specific, expert-defined style.")

# --- API Key Handling ---
st.sidebar.header("Configuration")
# Use a different key for the widget to avoid potential name clashes with the session_state key
api_key_input = st.sidebar.text_input("Enter your Google API Key:", type="password", key="api_key_input_widget")

# THE FIX: Remove the immediate st.rerun() call for more stable state handling
if st.sidebar.button("Set API Key", type="primary"):
    if api_key_input:
        st.session_state.google_api_key = api_key_input
        # The script will rerun naturally after the button click, no need to force it.
    else:
        st.sidebar.error("Please enter an API key.")

# --- Main App Logic ---
# The rest of the script now checks for the key set in st.session_state
if 'google_api_key' in st.session_state and st.session_state.google_api_key:
    try:
        genai.configure(api_key=st.session_state.google_api_key)
        st.sidebar.success("API Key accepted. Ready to generate.")
        
        st.header("1. Enter Company & Industry")
        company_name = st.text_input("Company Name:", placeholder="e.g., AT&T Inc.")
        
        industry_options = list(INDUSTRY_KPI_PRIORITY.keys())
        industry = st.selectbox("Select Industry:", options=industry_options)
        
        st.header("2. Upload Annual Report")
        pdf_file = st.file_uploader("Upload the PDF document", type="pdf")
        
        if st.button("✨ Generate Strengths Analysis", type="primary", use_container_width=True):
            if not company_name: st.error("Please enter a company name.")
            elif not industry: st.error("Please select an industry.")
            elif not pdf_file: st.error("Please upload a PDF file.")
            else:
                with st.spinner("Step 1/3: Processing and chunking PDF..."):
                    chunks = load_pdf_chunks(pdf_file)
                
                if chunks:
                    strengths = get_strengths_analysis(chunks, company_name, industry, st.session_state.google_api_key)
                    
                    if strengths is not None:
                        st.header(f"Strengths Analysis for {company_name}")
                        with st.expander("**Strengths** 💪 (from Annual Report)", expanded=True):
                            st.markdown(strengths)

    except Exception as e:
        st.sidebar.error(f"An error occurred. Please check your API Key and try again.")
        st.error(f"Details: {e}")
        if 'google_api_key' in st.session_state:
            del st.session_state.google_api_key
        # Use st.rerun() here to reset the app state after an error
        st.rerun()

else:
    st.info("Please enter your Google API Key in the sidebar to start.")

