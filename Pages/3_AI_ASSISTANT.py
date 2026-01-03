"""
AI Assistant Page - Queries & Reports
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add Repository to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Repository"))

from Repository.Query_Processor import QueryProcessor
from Repository.Summary_Generator import SummaryGenerator
from Repository.LLM_Monitoring_Agent import LLMMonitoringAgent

# Page code runs directly (no show() function needed for Streamlit pages)
st.title("🤖 AI Assistant - Chat Interface")

# Custom CSS for chat-like interface
st.markdown("""
<style>
    .chat-container {
        height: 500px;
        overflow-y: auto;
        overflow-x: hidden;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #E3F2FD;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #F5F5F5;
        margin-right: 20%;
    }
    .message-content {
        flex: 1;
    }
    .message-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #666;
    }
    .message-text {
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .empty-chat {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #999;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for conversation history and query processing
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'process_query_now' not in st.session_state:
    st.session_state.process_query_now = False
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'query_input_key' not in st.session_state:
    st.session_state.query_input_key = 0

# Initialize components
try:
    processor = QueryProcessor()
    generator = SummaryGenerator()
    agent = LLMMonitoringAgent()
    llm_available = agent.client is not None
except Exception as e:
    st.warning(f"⚠️ Some components may not be available: {str(e)}")
    processor = None
    generator = None
    agent = None
    llm_available = False

# Quick Questions Section (collapsible)
with st.expander("💡 Quick Questions", expanded=False):
    col1, col2 = st.columns(2)
    
    quick_questions = [
        "Which machine has the highest defect rate?",
        "What are the most common defect types?",
        "Show me recent anomalies",
        "Generate recommendations for improvement"
    ]
    
    # Check if any quick question button was clicked
    for i, question in enumerate(quick_questions):
        col_idx = i % 2
        button_key = f"quick_{i}"
        
        if col_idx == 0:
            if col1.button(question, key=button_key, width='stretch'):
                st.session_state.current_query = question
                st.session_state.process_query_now = True
                st.rerun()
        else:
            if col2.button(question, key=button_key, width='stretch'):
                st.session_state.current_query = question
                st.session_state.process_query_now = True
                st.rerun()

st.markdown("---")

# Display Conversation History
st.markdown("### 💬 Conversation")

# Helper function to escape HTML and preserve line breaks
def escape_html(text):
    """Escape HTML characters and convert newlines to <br>"""
    if not text:
        return ""
    import html
    import re
    # Remove the equals signs separator lines (they're decorative)
    text = re.sub(r'=+\s*\n?', '', str(text))
    # Remove "Detailed Analysis:" and "Analysis:" headers with separators
    text = re.sub(r'Detailed Analysis:\s*\n?', '', text)
    text = re.sub(r'Analysis:\s*\n?', '', text)
    text = re.sub(r'Enhanced Analysis:\s*\n?', '', text)
    # Escape HTML characters to prevent XSS
    escaped = html.escape(text)
    # Convert newlines to <br> for proper display
    escaped = escaped.replace('\n', '<br>')
    # Clean up multiple consecutive <br> tags
    escaped = re.sub(r'(<br>\s*){3,}', '<br><br>', escaped)
    return escaped

# Create scrollable chat container
if st.session_state.conversation_history:
    # Build HTML for all messages with proper formatting
    chat_html_parts = ['<div class="chat-container">']
    
    for i, conv in enumerate(st.session_state.conversation_history):
        # Escape HTML in messages (only the text content, not the structure)
        question_escaped = escape_html(conv['question'])
        answer_escaped = escape_html(conv['answer'])
        timestamp_escaped = escape_html(conv['timestamp'])
        
        # User message - build HTML properly
        chat_html_parts.append('<div class="chat-message user-message">')
        chat_html_parts.append('<div class="message-content">')
        chat_html_parts.append(f'<div class="message-header">👤 You • {timestamp_escaped}</div>')
        chat_html_parts.append(f'<div class="message-text">{question_escaped}</div>')
        chat_html_parts.append('</div></div>')
        
        # Assistant message - build HTML properly
        chat_html_parts.append('<div class="chat-message assistant-message">')
        chat_html_parts.append('<div class="message-content">')
        chat_html_parts.append(f'<div class="message-header">🤖 AI Assistant • {timestamp_escaped}</div>')
        chat_html_parts.append(f'<div class="message-text">{answer_escaped}</div>')
        chat_html_parts.append('</div></div>')
    
    chat_html_parts.append('</div>')
    
    # Join all parts
    chat_html = ''.join(chat_html_parts)
    
    # Add JavaScript to auto-scroll to bottom (minified to avoid issues)
    scroll_script = '<script>setTimeout(function(){var c=document.querySelector(".chat-container");if(c)c.scrollTop=c.scrollHeight;},100);</script>'
    chat_html += scroll_script
    
    # Render the HTML using st.components.v1.html or st.markdown
    st.markdown(chat_html, unsafe_allow_html=True)
else:
    # Empty state inside scrollable container
    st.markdown("""
    <div class="chat-container">
        <div class="empty-chat">
            👋 Start a conversation by asking a question below!
        </div>
    </div>
    """, unsafe_allow_html=True)

# st.markdown("---")

# # Query Input Section (Fixed at bottom)
# st.markdown("### Ask a Question")

# Use session state query or allow manual input
# Use dynamic key to force reset when clearing
user_query = st.text_area(
    "Your Question:",
    value=st.session_state.current_query if st.session_state.current_query else "",
    height=100,
    key=f"query_input_{st.session_state.query_input_key}",
    label_visibility="collapsed",
    placeholder="Type your question here... (Press Enter for new line, click Ask to send)"
)

# Update session state if user types manually (only if not empty and different)
if user_query and user_query != st.session_state.current_query:
    st.session_state.current_query = user_query

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear Input", use_container_width=True):
        st.session_state.current_query = ""
        st.session_state.process_query_now = False
        st.session_state.query_input_key += 1  # Force text_area to reset by changing key
        st.rerun()
with col3:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.current_query = ""
        st.session_state.process_query_now = False
        st.session_state.query_input_key += 1  # Force text_area to reset
        st.rerun()

# Process query if button clicked or quick question selected
should_process = (ask_button or st.session_state.process_query_now) and user_query and processor

if should_process:
    # Use the current query
    query_to_process = user_query
    
    # Reset the flag
    if st.session_state.process_query_now:
        st.session_state.process_query_now = False
    
    with st.spinner("🤔 Processing your question..."):
        try:
            result = processor.process_query(query_to_process, use_llm=llm_available)
            
            answer = result.get('answer', 'No answer available')
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                'question': query_to_process,
                'answer': answer,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Clear current query after processing
            st.session_state.current_query = ""
            
            # Rerun to show the new message in the chat
            st.rerun()
        
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.error(error_msg)
            st.info("Make sure you have run the manufacturing simulation and have data available.")
            
            # Add error to conversation history
            st.session_state.conversation_history.append({
                'question': query_to_process,
                'answer': f"❌ {error_msg}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            st.session_state.current_query = ""
            st.rerun()

st.markdown("---")

# Report Generation Section
st.markdown("### Generate Report")

report_type = st.selectbox(
    "Report Type:",
    ["Daily Summary Report", "Comprehensive PDF Report"]
)

# Get available simulation dates for filtering
if generator:
    try:
        generator.aggregator.load_results()
        available_dates = generator.aggregator.get_available_simulation_dates()
    except:
        available_dates = []
else:
    available_dates = []

# Date selection (for both report types)
selected_simulation_date = None
if available_dates:
    date_options = ["All Dates"] + available_dates
    selected_date_option = st.selectbox(
        "📅 Select Simulation Date:",
        options=date_options,
        index=0,
        help="Select a specific simulation date or 'All Dates' to include all data"
    )
    if selected_date_option != "All Dates":
        selected_simulation_date = selected_date_option
else:
    st.info("📅 No simulation dates available. Run a simulation first to generate reports.")

# PDF-specific options (only show for PDF reports)
pdf_options = None
if report_type == "Comprehensive PDF Report":
    pdf_options = st.radio(
        "PDF Report Detail Level:",
        ["Summary Only (No Per-Wafer Details)", "Full Report (With Per-Wafer Details)"],
        help="Summary Only: Faster generation, smaller file size. Full Report: Includes detailed information for each wafer."
    )

use_llm = st.checkbox("Include LLM-Enhanced Summary", value=True, disabled=not llm_available)
if not llm_available:
    st.caption("⚠️ LLM not available - will use fallback summary")

col1, col2 = st.columns(2)
with col1:
    if st.button("📄 Generate Report", type="primary", width='stretch'):
        if generator:
            with st.spinner("📊 Generating report..."):
                try:
                    if report_type == "Daily Summary Report":
                        summary = generator.generate_text_summary(use_llm=use_llm)
                        st.markdown("### Generated Summary Report")
                        st.markdown("---")
                        st.text(summary)
                        
                        # Download button
                        st.download_button(
                            label="💾 Download as Text",
                            data=summary,
                            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    elif report_type == "Comprehensive PDF Report":
                        # Determine if per-wafer details should be included
                        include_per_wafer = pdf_options == "Full Report (With Per-Wafer Details)" if pdf_options else True
                        pdf_path = generator.generate_pdf_report(
                            use_llm=use_llm, 
                            include_per_wafer_details=include_per_wafer,
                            simulation_date=selected_simulation_date
                        )
                        report_type_text = "Summary Only" if not include_per_wafer else "Full Report"
                        date_text = f" for {selected_simulation_date}" if selected_simulation_date else " (All Dates)"
                        st.success(f"✅ PDF report generated ({report_type_text}{date_text}): {pdf_path.name}")
                        
                        # Read and provide download
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="💾 Download PDF",
                                data=pdf_file.read(),
                                file_name=pdf_path.name,
                                mime="application/pdf"
                            )
                
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        else:
            st.error("Report generator not available")

with col2:
    if st.button("🔄 Refresh Data", width='stretch'):
        st.rerun()

# Navigation buttons at the bottom
st.markdown("---")
st.markdown("### 🔗 Page Navigation")
st.info("💡 Use the buttons below to navigate between pages")
col_nav = st.columns(3)
with col_nav[0]:
    if st.button("🏠 Welcome Page", use_container_width=True, type="secondary"):
        st.switch_page("WELCOME.py")
with col_nav[1]:
    if st.button("← Previous: 📊 Defect Analytics", use_container_width=True, type="secondary"):
        st.switch_page("Pages/2_DEFECT ANALYTICS.py")
with col_nav[2]:
    st.empty()  # Empty column for spacing (last page)

