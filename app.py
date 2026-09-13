"""RightsGuide AI / Qanoon Dost AI Streamlit Application."""

from collections import OrderedDict
import hashlib
import logging
import uuid

import streamlit as st

from chat_storage import delete_conversation, load_conversations, save_conversation
from llm_response import get_legal_answer
from retrieval import DEFAULT_TOP_K as KB_TOP_K
from user_documents import (
    DEFAULT_TOP_K as DOCUMENT_TOP_K,
    build_document,
    get_document_answer,
    load_embedding_model,
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Qanoon Dost AI - RightsGuide",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE THEME CSS (LIGHT, DARK & SYSTEM RESILIENT)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* 1. Global Font & Reset */
    * {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* 2. SIDEBAR UNIVERSAL CONTRAST (Works in both Dark & Light themes) */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #334155 !important;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    /* Sidebar Brand Box */
    .brand-box {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 14px;
        padding: 1.1rem;
        margin-bottom: 1rem;
    }
    .brand-box .brand-name {
        font-weight: 800;
        font-size: 1.25rem;
        color: #38bdf8 !important;
    }
    .brand-box .brand-tag {
        color: #cbd5e1 !important;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    /* Sidebar Buttons (Fixing Invisible/White Boxes) */
    [data-testid="stSidebar"] button {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebar"] button:hover {
        background-color: #0284c7 !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25) !important;
    }

    [data-testid="stSidebar"] button p {
        color: inherit !important;
    }

    /* 3. HERO BANNER */
    .hero {
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0369a1 100%) !important;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15);
    }
    .hero h1 {
        margin: 0;
        color: #ffffff !important;
        font-weight: 800;
        font-size: clamp(1.7rem, 2.2vw, 2.2rem);
    }
    .hero p {
        margin: 0.35rem 0 0.8rem;
        color: #e0f2fe !important;
        font-size: 0.95rem;
    }
    .pillar-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .pillar-tag {
        background: rgba(255, 255, 255, 0.18) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* 4. CHAT BUBBLES: FULL CONTRAST SAFEGUARD */
    [class*="st-key-chat-panel"] {
        padding: 0.5rem 0.5rem 6.5rem 0.5rem !important; /* Prevents input overlap */
    }

    /* User Chat Bubble (High-contrast Blue) */
    [class*="st-key-chat-bubble-user-"] {
        background: #0284c7 !important;
        border-radius: 16px 16px 4px 16px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.2) !important;
    }
    [class*="st-key-chat-bubble-user-"] * {
        color: #ffffff !important;
        font-size: 0.98rem;
        line-height: 1.6;
    }

    /* Assistant Chat Bubble (Card styling with dark text) */
    [class*="st-key-chat-bubble-assistant-"] {
        background: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 16px 16px 16px 4px !important;
        padding: 1.2rem 1.3rem !important;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05) !important;
    }

    /* Force all text inside assistant bubble to be deep charcoal */
    [class*="st-key-chat-bubble-assistant-"] * {
        color: #0f172a !important;
        font-size: 0.98rem;
        line-height: 1.65;
    }

    /* Sources / Citations Card */
    .source-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 4px solid #0284c7 !important;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.35rem 0;
    }
    .source-card strong {
        color: #0f172a !important;
        font-size: 0.9rem;
    }
    .source-card small {
        color: #64748b !important;
        font-size: 0.8rem;
    }

    /* 5. FLOATING BOTTOM COMPOSER */
    [class*="st-key-legal-chat-composer"] {
        position: fixed;
        bottom: 0.75rem;
        z-index: 9999;
        left: calc(18rem + 1.2rem);
        right: 1.5rem;
        max-width: calc(100% - (18rem + 2.7rem));
        padding: 0.45rem;
        background: #0f172a !important;
        border: 1px solid #334155;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    /* Text Input inside Composer */
    [class*="st-key-law_questions_form"] [data-testid="stTextInput"] input {
        background: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
        border-radius: 12px !important;
        height: 2.8rem !important;
    }
    [class*="st-key-law_questions_form"] [data-testid="stTextInput"] input::placeholder {
        color: #94a3b8 !important;
    }

    /* Send Button */
    [class*="st-key-law_questions_form"] button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        height: 2.8rem !important;
        font-weight: 600 !important;
    }
    [class*="st-key-law_questions_form"] button:hover {
        background-color: #0369a1 !important;
    }

    /* Suggestion Chips */
    [class*="st-key-empty-state-suggestions"] button {
        background: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        color: #0f172a !important;
        font-weight: 500 !important;
        padding: 0.7rem 0.9rem !important;
    }
    [class*="st-key-empty-state-suggestions"] button:hover {
        background: #e0f2fe !important;
        border-color: #0284c7 !important;
        color: #0369a1 !important;
    }

    @media (max-width: 768px) {
        [class*="st-key-legal-chat-composer"] {
            left: 0.75rem !important;
            right: 0.75rem !important;
            max-width: calc(100% - 1.5rem) !important;
            bottom: 0.5rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "uploaded_documents": {},
        "document_model": None,
        "language": "roman_urdu",
        "chat_sessions": OrderedDict(),
        "active_chat_id": None,
        "chat_sessions_loaded": False,
        "enable_web_search": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.chat_sessions_loaded:
        persisted = load_conversations()
        if persisted:
            st.session_state.chat_sessions = persisted
        st.session_state.chat_sessions_loaded = True

    if (
        st.session_state.chat_sessions
        and st.session_state.active_chat_id not in st.session_state.chat_sessions
    ):
        st.session_state.active_chat_id = next(
            reversed(st.session_state.chat_sessions)
        )


def create_chat_session(title="New Consultation"):
    active_chat_id = st.session_state.active_chat_id
    if active_chat_id in st.session_state.chat_sessions:
        active_session = st.session_state.chat_sessions[active_chat_id]
        if not active_session["messages"]:
            active_session["title"] = title
            return active_chat_id

    chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[chat_id] = {"title": title, "messages": []}
    st.session_state.active_chat_id = chat_id
    return chat_id


def mark_chat_active(chat_id):
    if chat_id not in st.session_state.chat_sessions:
        return
    st.session_state.chat_sessions.move_to_end(chat_id)
    st.session_state.active_chat_id = chat_id
    save_conversation(chat_id, st.session_state.chat_sessions[chat_id])


def delete_chat_session(chat_id):
    if chat_id not in st.session_state.chat_sessions:
        return
    del st.session_state.chat_sessions[chat_id]
    delete_conversation(chat_id)
    if st.session_state.active_chat_id == chat_id:
        if st.session_state.chat_sessions:
            st.session_state.active_chat_id = next(
                reversed(st.session_state.chat_sessions)
            )
        else:
            st.session_state.active_chat_id = None


def get_real_conversations():
    return [
        (chat_id, conv)
        for chat_id, conv in reversed(list(st.session_state.chat_sessions.items()))
        if conv["messages"]
    ]


def update_session_title(chat_id):
    session = st.session_state.chat_sessions.get(chat_id)
    if not session or not session["messages"]:
        return
    if session.get("title") not in (None, "", "New Consultation"):
        return
    try:
        first_user = next(
            msg for msg in session["messages"] if msg["role"] == "user"
        )
        cleaned = " ".join(first_user["content"].split())
        session["title"] = (
            (cleaned[:26] + "...") if len(cleaned) > 26 else cleaned
        )
    except StopIteration:
        session["title"] = "Legal Query"


def add_message(
    role,
    content,
    sources=None,
    provider_used=None,
    web_search_used=False,
):
    chat_id = st.session_state.active_chat_id
    if chat_id not in st.session_state.chat_sessions:
        create_chat_session()
        chat_id = st.session_state.active_chat_id
    else:
        mark_chat_active(chat_id)

    message = {
        "role": role,
        "content": content,
        "sources": sources or [],
        "provider_used": provider_used,
        "web_search_used": web_search_used,
    }
    st.session_state.chat_sessions[chat_id]["messages"].append(message)
    if role == "user":
        update_session_title(chat_id)
    save_conversation(chat_id, st.session_state.chat_sessions[chat_id])


def get_active_messages():
    chat_id = st.session_state.active_chat_id
    if chat_id not in st.session_state.chat_sessions:
        return []
    return st.session_state.chat_sessions[chat_id]["messages"]


def export_active_conversation():
    messages = get_active_messages()
    if not messages:
        return ""
    text_export = "# Qanoon Dost AI - Legal Advisory Summary\n\n"
    for msg in messages:
        role = "Citizen Query" if msg["role"] == "user" else "Legal Response"
        text_export += f"### {role}:\n{msg['content']}\n\n"
        if msg.get("sources"):
            text_export += "**Statutory References:**\n"
            for s in msg["sources"]:
                text_export += f"- {s.get('law_name')} (Page {s.get('page_start', 'N/A')})\n"
            text_export += "\n"
    text_export += "---\n*Generated by Qanoon Dost AI (informational guidance only)*"
    return text_export


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📑 Legal Citations ({len(sources)} Authorities)"):
        for source in sources:
            pages = source.get("page_start", "N/A")
            if source.get("page_end") and source["page_end"] != pages:
                pages = f"{pages}-{source['page_end']}"
            st.markdown(
                f'<div class="source-card">'
                f'<strong>⚖️ {source.get("law_name", "Statute")}</strong>'
                f"<small>Statutory Reference · Page {pages}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_message(message, message_index):
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            with st.container(key=f"chat-bubble-user-{message_index}"):
                st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="⚖️"):
            with st.container(key=f"chat-bubble-assistant-{message_index}"):
                st.markdown(message["content"])
            if message.get("web_search_used"):
                st.caption("🌐 Verified via Live Search · Tavily")
            elif message.get("provider_used"):
                st.caption(
                    f"⚡ Verified via {message['provider_used'].upper()} RAG Engine"
                )
            render_sources(message.get("sources", []))


def handle_legal_question(question, language):
    question = (question or "").strip()
    if not question:
        st.warning("Please enter a legal query.")
        return

    add_message("user", question)
    with st.spinner("Searching Pakistani statutes..."):
        try:
            result = get_legal_answer(
                question,
                language=language,
                top_k=KB_TOP_K,
                enable_web_search=st.session_state.get(
                    "enable_web_search", False
                ),
            )
            answer_text = (
                result.get("answer")
                or "I could not find enough relevant Pakistani legal material to answer confidently."
            )
            sources = result.get("sources", [])
            provider = result.get("provider_used")
            web_used = result.get("web_search_used", False)
            add_message(
                "assistant",
                answer_text,
                sources=sources,
                provider_used=provider,
                web_search_used=web_used,
            )
        except Exception as e:
            logger.exception("Error answering question: %s", e)
            add_message(
                "assistant",
                "Service momentarily unavailable. Please check your query or retry.",
            )


def set_question_from_example(question):
    st.session_state.legal_question_input = question


def get_document_id(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def get_document_options(documents):
    names = {}
    for doc_id, doc in documents.items():
        names.setdefault(doc.name, []).append(doc_id)
    options = {}
    for name, doc_ids in names.items():
        for doc_id in doc_ids:
            label = name if len(doc_ids) == 1 else f"{name} ({doc_id[:8]})"
            options[label] = doc_id
    return options


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION UI
# ─────────────────────────────────────────────────────────────────────────────
init_state()

# Top Hero Header
with st.container():
    st.markdown(
        """
        <div class="hero">
            <h1>⚖️ Qanoon Dost AI (قانون دوست)</h1>
            <p>Pakistan's AI Legal Rights Assistant · Grounded in Official Statutes</p>
            <div class="pillar-tags">
                <span class="pillar-tag">🏠 Tenant Rights</span>
                <span class="pillar-tag">👨‍👩‍👧 Family Laws</span>
                <span class="pillar-tag">💼 Labor & Minimum Wage</span>
                <span class="pillar-tag">🛒 Consumer Courts</span>
                <span class="pillar-tag">📜 CrPC & Police Rights</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-name">⚖️ Qanoon Dost AI</div>
            <div class="brand-tag">Access to justice for 220M+ Pakistani citizens.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "➕ New Consultation", key="new_chat_button", use_container_width=True
    ):
        create_chat_session()
        st.rerun()

    active_msgs = get_active_messages()
    if active_msgs:
        chat_markdown = export_active_conversation()
        st.download_button(
            label="📥 Export Consultation",
            data=chat_markdown,
            file_name="Qanoon_Dost_Legal_Advice.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption("Recent Consultations")
    recent_conversations = get_real_conversations()
    for chat_id, conv in recent_conversations:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                conv["title"], key=f"chat_{chat_id}", use_container_width=True
            ):
                mark_chat_active(chat_id)
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{chat_id}", help="Delete chat"):
                delete_chat_session(chat_id)
                st.rerun()

    st.markdown("---")
    st.caption("Settings")
    st.session_state.language = st.selectbox(
        "Response Language",
        options=["roman_urdu", "english", "urdu"],
        index=["roman_urdu", "english", "urdu"].index(
            st.session_state.language
        ),
        format_func=lambda x: {
            "roman_urdu": "Roman Urdu (آسان اردو)",
            "english": "English",
            "urdu": "Urdu Script (اردو)",
        }[x],
    )

    st.session_state.enable_web_search = st.toggle(
        "🌐 Live Web Search",
        value=st.session_state.get("enable_web_search", False),
        help="Search web when local legal knowledge base confidence is low.",
    )

    st.markdown(
        """
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:1rem; padding:0.6rem; border:1px solid #334155; border-radius:8px;">
        ⚠️ <strong>Notice:</strong> Statutory guidance only; consult a licensed advocate for legal proceedings.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main Navigation
tab_laws, tab_upload = st.tabs(
    ["💬 Pakistani Law Chat", "📄 Private PDF Analysis"]
)

with tab_laws:
    chat_messages = get_active_messages()

    if chat_messages:
        with st.container(key="chat-panel"):
            for i, msg in enumerate(chat_messages):
                render_message(msg, i)
    else:
        st.markdown(
            """
            <div style="text-align:center; padding: 2rem 1rem;">
                <h3 style="margin-bottom:0.25rem;">How can Qanoon Dost assist you today?</h3>
                <p style="color:#64748b;">Select an everyday scenario below or type your situation:</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(key="empty-state-suggestions"):
            c1, c2 = st.columns(2)
            with c1:
                st.button(
                    "🏠 Landlord notice ke baghair rent barha sakta hai?",
                    on_click=set_question_from_example,
                    args=(
                        "Landlord notice diye baghair rent barha sakta hai ya nahi? Rent laws kya kehte hain?",
                    ),
                    use_container_width=True,
                )
                st.button(
                    "⚖️ Police FIR darj karne se mana kare to kya karein?",
                    on_click=set_question_from_example,
                    args=(
                        "Agar police station FIR darj karne se inkar kar de to CrPC ke teht kya rasta hai?",
                    ),
                    use_container_width=True,
                )
            with c2:
                st.button(
                    "🛒 Online fraud ya kharab saman ki refund complaint?",
                    on_click=set_question_from_example,
                    args=(
                        "Online seller ne kharab cheez bheji aur refund nahi de raha. Consumer court kaise approach karein?",
                    ),
                    use_container_width=True,
                )
                st.button(
                    "👨‍👩‍👧 Khula aur child custody (hizanat) ke rules?",
                    on_click=set_question_from_example,
                    args=(
                        "Khula lene ka legal procedure aur bache ki custody (hizanat) mother ko kab tak milti hai?",
                    ),
                    use_container_width=True,
                )

    # Bottom Fixed Input Form
    with st.container(key="legal-chat-composer"):
        with st.form("law_questions_form", clear_on_submit=True):
            cols = st.columns([5, 1])
            with cols[0]:
                q_input = st.text_input(
                    "Ask Pakistani Legal Question",
                    placeholder="Apna masla yahan likhein (e.g. Kirayadar shop khali nahi kar raha...)",
                    label_visibility="collapsed",
                    key="legal_question_input",
                )
            with cols[1]:
                send_pressed = st.form_submit_button(
                    "Send ⚖️", use_container_width=True
                )

    if send_pressed and q_input.strip():
        handle_legal_question(q_input, st.session_state.language)
        st.rerun()

with tab_upload:
    st.subheader("📄 Analyze Tenancy Agreement or Legal Notice")
    st.caption(
        "Upload PDF documents for confidential, session-only legal analysis."
    )

    uploads = st.file_uploader(
        "Upload PDF document",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploads:
        if st.session_state.document_model is None:
            with st.spinner("Initializing parser..."):
                st.session_state.document_model = load_embedding_model()

        for upload in uploads:
            file_bytes = upload.getvalue()
            doc_id = get_document_id(file_bytes)
            if doc_id not in st.session_state.uploaded_documents:
                try:
                    with st.spinner(f"Indexing {upload.name}..."):
                        st.session_state.uploaded_documents[doc_id] = (
                            build_document(
                                upload.name,
                                file_bytes,
                                st.session_state.document_model,
                            )
                        )
                except ValueError as err:
                    st.error(f"{upload.name}: {err}")

        if st.session_state.uploaded_documents:
            doc_options = get_document_options(
                st.session_state.uploaded_documents
            )
            chosen_label = st.selectbox(
                "Select active document:", list(doc_options)
            )
            chosen_id = doc_options[chosen_label]

            with st.form("pdf_qa_form", clear_on_submit=True):
                p_cols = st.columns([5, 1])
                with p_cols[0]:
                    pdf_q = st.text_input(
                        "Question about this document",
                        placeholder="What does this contract say about security deposit?",
                        label_visibility="collapsed",
                    )
                with p_cols[1]:
                    pdf_btn = st.form_submit_button(
                        "Analyze 📄", use_container_width=True
                    )

            if pdf_btn and pdf_q.strip():
                with st.spinner("Extracting contract clauses..."):
                    try:
                        res = get_document_answer(
                            st.session_state.uploaded_documents[chosen_id],
                            pdf_q.strip(),
                            st.session_state.document_model,
                            language=st.session_state.language,
                            top_k=DOCUMENT_TOP_K,
                        )
                        st.markdown("### 📋 Contract Analysis Result")
                        st.write(res["answer"])
                        if res.get("sources"):
                            render_sources(res["sources"])
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

            if st.button("🗑️ Clear Uploaded Documents", key="clear_docs_btn"):
                st.session_state.uploaded_documents = {}
                st.rerun()
