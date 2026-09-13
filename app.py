"""RightsGuide AI Streamlit application."""

from collections import OrderedDict
import hashlib
import logging
import uuid

import streamlit as st

from llm_response import get_legal_answer
from retrieval import DEFAULT_TOP_K as KB_TOP_K
from chat_storage import delete_conversation, load_conversations, save_conversation
from user_documents import (
    DEFAULT_TOP_K as DOCUMENT_TOP_K,
    build_document,
    get_document_answer,
    load_embedding_model,
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RightsGuide AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #f3f7fb;
        --panel: #ffffff;
        --navy-900: #0d1b2a;
        --navy-800: #12263d;
        --navy-700: #18324b;
        --blue-600: #1d73c9;
        --blue-500: #3c9fe8;
        --cyan-400: #6ad4ff;
        --sky-100: #eff7ff;
        --slate-200: #dfe7f1;
        --slate-400: #7d8997;
        --slate-600: #475569;
        --success: #1b7a52;
        --shadow: 0 12px 32px rgba(11, 30, 48, 0.08);
        --composer-height: 3.9rem;
        --composer-gap: 0.9rem;
    }

    .stApp {
        background: var(--bg);
        color: var(--navy-800);
        overflow-x: hidden;
    }

    *, *::before, *::after {
        box-sizing: border-box;
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0);
        box-shadow: none;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #12263d 100%);
        border-right: 1px solid rgba(138, 164, 189, 0.18);
    }

    [data-testid="stSidebar"] * {
        color: #edf5ff;
    }

    .brand-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(141, 182, 218, 0.18);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem;
        margin-bottom: 1rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }

    .brand-box .brand-name {
        display: flex;
        align-items: center;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.01em;
        margin-bottom: 0.25rem;
    }

    .brand-box .brand-tag {
        color: rgba(237, 245, 255, 0.82);
        font-size: 0.8rem;
        line-height: 1.5;
    }

    .st-key-new_chat_button button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        background: rgba(60, 159, 232, 0.12) !important;
        border: 1px solid rgba(154, 206, 255, 0.22) !important;
        color: #edf5ff !important;
        transition: background 0.2s ease !important;
    }

    .st-key-new_chat_button button:hover {
        background: rgba(60, 159, 232, 0.2) !important;
    }



    [class*="st-key-history-item-"] {
        margin: 0.35rem 0;
        width: 100%;
        overflow: visible !important;
    }

    [class*="st-key-history-item-"] [data-testid="stHorizontalBlock"] {
        width: 100%;
        min-width: 0;
        overflow: visible !important;
        align-items: center;
        flex-wrap: nowrap !important;
        gap: 0.25rem !important;
        display: flex !important;
    }

    [class*="st-key-history-item-"] [data-testid="stHorizontalBlock"] > div {
        min-width: 0 !important;
        overflow: hidden !important;
    }

    [class*="st-key-history-item-"] [data-testid="stHorizontalBlock"] > div:not(:has([data-testid="stPopover"])) {
        flex: 1 1 auto !important;
    }

    [class*="st-key-history-item-"] [data-testid="stHorizontalBlock"] > div:not(:has([data-testid="stPopover"])) button {
        min-width: 0 !important;
        max-width: 100% !important;
        overflow: hidden !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
    }

    [class*="st-key-history-item-"] [data-testid="stHorizontalBlock"] > div:not(:has([data-testid="stPopover"])) button p {
        min-width: 0 !important;
        max-width: 100% !important;
        overflow: hidden !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
    }

    [class*="st-key-history-item-"] [data-testid="stHorizontalBlock"] > div:has([data-testid="stPopover"]) {
        flex: 0 0 2.6rem !important;
        width: 2.6rem !important;
        min-width: 2.6rem !important;
        max-width: 2.6rem !important;
        padding: 0 !important;
        overflow: visible !important;
        border: 0 !important;
        margin: 0 !important;
    }

    [class*="st-key-history-item-"] button {
        width: 100%;
        justify-content: flex-start !important;
        border-radius: 10px !important;
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(147, 170, 196, 0.12) !important;
        color: #edf5ff !important;
        padding: 0.7rem 0.8rem !important;
        text-align: left;
        transition: background 0.2s ease, border-color 0.2s ease !important;
    }

    [class*="st-key-history-item-"] button:hover {
        background: rgba(60, 159, 232, 0.14) !important;
        border-color: rgba(154, 206, 255, 0.2) !important;
    }

    [class*="st-key-history-item-"] [data-testid="stPopover"] > button {
        width: 2.6rem !important;
        min-width: 2.6rem !important;
        max-width: 2.6rem !important;
        height: 2.35rem !important;
        min-height: 2.35rem !important;
        padding: 0 0.25rem !important;
        margin: 0 auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 0 !important;
        background: transparent !important;
        border: 0 !important;
        border-bottom: 0 !important;
        outline: 0 !important;
        box-shadow: none !important;
        color: rgba(237, 245, 255, 0.7) !important;
        transition: color 0.2s ease !important;
    }

    [class*="st-key-history-item-"] [data-testid="stPopover"] > button:hover {
        background: transparent !important;
        border: 0 !important;
        border-bottom: 0 !important;
        outline: 0 !important;
        box-shadow: none !important;
        color: #ffffff !important;
    }

    [class*="st-key-history-item-"] [data-testid="stPopover"] > button::before,
    [class*="st-key-history-item-"] [data-testid="stPopover"] > button::after {
        display: none !important;
        content: none !important;
    }

    [class*="st-key-history-item-"] [data-testid="stPopover"] > button p {
        margin: 0 !important;
        line-height: 1 !important;
    }

    [class*="st-key-history-item-"] [data-testid="stPopover"] {
        width: 2.6rem !important;
        min-width: 2.6rem !important;
        max-width: 2.6rem !important;
        overflow: visible !important;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 0 !important;
        border-bottom: 0 !important;
        box-shadow: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [class*="st-key-history-item-"] [data-testid="stHorizontalBlock"]::before,
    [class*="st-key-history-item-"] [data-testid="stHorizontalBlock"]::after {
        display: none !important;
        content: none !important;
    }

    /* The history row is the horizontal block itself, not a descendant block. */
    [class*="st-key-history-item-"][data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 0.25rem !important;
        width: 100% !important;
        min-width: 0 !important;
        overflow: visible !important;
    }

    [class*="st-key-history-item-"][data-testid="stHorizontalBlock"] > [data-testid="stElementContainer"] {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        max-width: none !important;
        overflow: hidden !important;
    }

    [class*="st-key-history-item-"][data-testid="stHorizontalBlock"] > [data-testid="stElementContainer"] button {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        overflow: hidden !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
    }

    [class*="st-key-history-item-"][data-testid="stHorizontalBlock"] > [data-testid="stElementContainer"] button p {
        min-width: 0 !important;
        max-width: 100% !important;
        overflow: hidden !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
    }

    [class*="st-key-history-item-"][data-testid="stHorizontalBlock"] > [data-testid="stLayoutWrapper"] {
        flex: 0 0 2.6rem !important;
        width: 2.6rem !important;
        min-width: 2.6rem !important;
        max-width: 2.6rem !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
    }

    [data-theme="light"] [data-testid="stSidebar"] [class*="st-key-history-item-"] [data-testid="stPopover"] > button,
    [data-theme="light"] [data-testid="stSidebar"] [class*="st-key-history-item-"] [data-testid="stPopover"] > button:hover {
        background: transparent !important;
        border: 0 !important;
        border-bottom: 0 !important;
        outline: 0 !important;
        box-shadow: none !important;
        color: #123d5d !important;
    }

    [data-theme="light"] [data-testid="stSidebar"] .st-key-new_chat_button button,
    [data-theme="light"] [data-testid="stSidebar"] [class*="st-key-history-item-"] button {
        background: rgba(60, 159, 232, 0.12) !important;
        border-color: rgba(29, 115, 201, 0.22) !important;
        color: #123d5d !important;
    }

    [data-theme="light"] [data-testid="stSidebar"] .st-key-new_chat_button button:hover,
    [data-theme="light"] [data-testid="stSidebar"] [class*="st-key-history-item-"] button:hover {
        background: rgba(60, 159, 232, 0.2) !important;
        border-color: rgba(29, 115, 201, 0.3) !important;
        color: #123d5d !important;
    }

    [class*="st-key-settings-panel"] {
        margin-top: 0.25rem;
        padding: 0.2rem 0.1rem 0;
    }

    [class*="st-key-settings-panel"] h2 {
        margin-bottom: 0.8rem;
        color: #edf5ff;
        font-size: 1.05rem;
        letter-spacing: 0.01em;
    }

    [class*="st-key-settings-panel"] [data-testid="stSelectbox"] {
        margin-bottom: 0.25rem;
    }

    [class*="st-key-settings-panel"] [data-testid="stSelectbox"] label {
        color: rgba(237, 245, 255, 0.78) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }

    [class*="st-key-settings-panel"] [data-testid="stSelectbox"] [role="combobox"] {
        background: #ffffff !important;
        border: 1px solid rgba(141, 182, 218, 0.42);
        border-radius: 10px;
        color: var(--navy-800) !important;
        min-width: 0;
        padding-right: 0.7rem !important;
        overflow: visible !important;
        display: flex !important;
        align-items: center !important;
        transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    [class*="st-key-settings-panel"] [data-testid="stSelectbox"] [role="combobox"] * {
        color: var(--navy-800) !important;
    }

    [class*="st-key-settings-panel"] [data-testid="stSelectbox"] [role="combobox"] svg {
        fill: var(--blue-600) !important;
        color: var(--blue-600) !important;
        flex: 0 0 auto !important;
        width: 1.25rem !important;
        height: 1.25rem !important;
        opacity: 1 !important;
    }

    [class*="st-key-settings-panel"] [data-testid="stSelectbox"] [role="combobox"]:hover,
    [class*="st-key-settings-panel"] [data-testid="stSelectbox"] [role="combobox"]:focus-within {
        background: #ffffff !important;
        border-color: rgba(154, 206, 255, 0.46);
        box-shadow: 0 0 0 2px rgba(60, 159, 232, 0.1);
    }

    body:has([class*="st-key-settings-panel"] [data-testid="stSelectbox"] [aria-expanded="true"]) [data-baseweb="menu"] {
        background: #ffffff !important;
        border: 1px solid rgba(141, 182, 218, 0.3);
    }

    body:has([class*="st-key-settings-panel"] [data-testid="stSelectbox"] [aria-expanded="true"]) [data-baseweb="menu"] [role="option"] {
        background: #ffffff !important;
        color: var(--navy-800) !important;
    }

    body:has([class*="st-key-settings-panel"] [data-testid="stSelectbox"] [aria-expanded="true"]) [data-baseweb="menu"] [role="option"]:hover,
    body:has([class*="st-key-settings-panel"] [data-testid="stSelectbox"] [aria-expanded="true"]) [data-baseweb="menu"] [role="option"][aria-selected="true"] {
        background: var(--sky-100) !important;
        color: var(--navy-800) !important;
    }

    .disclaimer-box {
        margin-top: 1rem;
        padding: 0.8rem 0.9rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(147, 170, 196, 0.12);
        color: rgba(237,245,255,0.9);
        font-size: 0.76rem;
        line-height: 1.55;
    }

    .hero {
        border-radius: 20px;
        background: linear-gradient(135deg, #0f2438 0%, #123d5d 50%, #1a5b86 100%);
        padding: 1.15rem 1.5rem;
        margin-bottom: 0.25rem;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
        max-width: 100%;
    }

    /* Consolidated header/navigation spacing - controlled from one place */
    [class*="st-key-main-header"], [class*="st-key-main-navigation"], [class*="st-key-main-navigation"] [data-testid="stTabs"] {
        margin: 0 !important;
        padding: 0 !important;
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top right, rgba(118, 211, 255, 0.2), transparent 28%);
        pointer-events: none;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        font-size: clamp(2rem, 2.5vw, 2.5rem);
        color: white;
        letter-spacing: -0.03em;
    }

    .hero p {
        position: relative;
        z-index: 1;
        margin: 0.5rem 0 0;
        color: rgba(255,255,255,0.86);
        font-size: 0.98rem;
    }

[class*="st-key-chat-panel"] {
    background: rgba(255,255,255,0.54);
    border: 1px solid rgba(142, 165, 183, 0.22);
    border-radius: 18px;
    box-shadow: var(--shadow);
    padding: 1rem;
    padding-bottom: calc(var(--composer-height) + var(--composer-gap));
    min-height: 0;
    width: 100%;
    box-sizing: border-box;
    overflow: visible;
}

[class*="st-key-legal-chat-workspace"] {
    position: relative;
    width: 100%;
    overflow: visible !important;
}

[data-testid="stTabContent"]:has([class*="st-key-legal-chat-workspace"]),
[data-testid="stTabs"] > div:has([class*="st-key-legal-chat-workspace"]) {
    overflow: visible !important;
}

.empty-state {
    min-height: clamp(240px, 30vh, 330px);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 1.15rem 1.5rem;
}

.empty-state-icon {
    font-size: 2.5rem;
    margin-bottom: 0.45rem;
}

.empty-state h2 {
    margin: 0;
    color: var(--navy-800);
    font-size: 1.45rem;
}

.empty-state p {
    margin: 0.35rem 0 0.85rem;
    color: var(--slate-600);
}

.example-question {
    display: inline-block;
    background: white;
    border: 1px solid var(--slate-200);
    border-radius: 12px;
    padding: 0.7rem 1rem;
    margin: 0.3rem;
    color: var(--navy-800);
    font-size: 0.9rem;
}

[class*="st-key-empty-state-suggestions"] {
    width: 100%;
    margin-top: -0.1rem;
}

[class*="st-key-empty-state-suggestions"] button {
    width: 100%;
    min-height: 2.8rem;
    background: white !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: 12px !important;
    color: var(--navy-800) !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 0.65rem !important;
    transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease !important;
}

[class*="st-key-empty-state-suggestions"] button:hover {
    background: var(--sky-100) !important;
    border-color: rgba(60, 159, 232, 0.45) !important;
    box-shadow: 0 6px 14px rgba(29, 115, 201, 0.1);
    transform: translateY(-1px);
}

    [data-testid="stChatMessage"] {
        border-radius: 16px !important;
        margin: 0.4rem 0 0.75rem 0;
    }

    [data-testid="stChatMessage"] .stChatMessageContent {
        background: transparent !important;
    }

    [class*="st-key-chat-bubble-user-"] {
        background: linear-gradient(135deg, #0f4d7a, #1d73c9);
        color: white !important;
        border-radius: 18px 18px 6px 18px !important;
        padding: 0.9rem 1rem !important;
        box-shadow: 0 10px 22px rgba(29, 115, 201, 0.18);
    }

    [class*="st-key-chat-bubble-assistant-"] {
        background: white;
        border: 1px solid rgba(130, 157, 179, 0.23);
        border-radius: 18px 18px 18px 6px !important;
        padding: 0.9rem 1rem !important;
        box-shadow: 0 8px 18px rgba(15, 36, 56, 0.06);
    }

    [class*="st-key-chat-bubble-assistant-"] p,
    [class*="st-key-chat-bubble-user-"] p {
        margin: 0;
        line-height: 1.65;
        font-size: 0.98rem;
    }

    .source-card {
        background: linear-gradient(135deg, #ffffff, #f6faff);
        border: 1px solid rgba(119, 141, 167, 0.2);
        border-left: 4px solid var(--blue-500);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        margin: 0.5rem 0 0.35rem;
        box-shadow: 0 4px 12px rgba(15, 36, 56, 0.04);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .source-card:hover {
        border-color: rgba(60, 159, 232, 0.4);
        border-left-color: var(--blue-600);
        box-shadow: 0 6px 16px rgba(29, 115, 201, 0.08);
    }

    .source-card strong {
        color: var(--navy-800);
        display: block;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .source-card small {
        color: var(--slate-600);
        display: block;
        margin-top: 0.2rem;
        font-size: 0.82rem;
    }

    .typing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: white;
        border: 1px solid rgba(130, 157, 179, 0.23);
        border-radius: 999px;
        padding: 0.7rem 0.9rem;
        box-shadow: 0 8px 20px rgba(21, 45, 70, 0.06);
    }

    .typing-indicator span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #7bb2dc;
        display: inline-block;
        animation: blink 1.2s infinite ease-in-out;
    }

    .typing-indicator span:nth-child(2) { animation-delay: 0.15s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.3s; }

    @keyframes blink {
        0%, 80%, 100% { opacity: 0.3; transform: translateY(0); }
        40% { opacity: 1; transform: translateY(-2px); }
    }

    .compact-note {
        color: var(--slate-600);
        font-size: 0.78rem;
        margin-top: 0.4rem;
    }

    [class*="st-key-knowledge-base-note"] {
        margin: 0.65rem 0 0;
        padding: 0.65rem 0.85rem;
        border-top: 1px solid rgba(142, 165, 183, 0.18);
        color: var(--slate-600);
        text-align: center;
    }

    [class*="st-key-knowledge-base-note"] p {
        margin: 0;
        line-height: 1.45;
        font-size: 0.78rem;
    }

    [class*="st-key-knowledge-base-note"] strong {
        color: var(--navy-700);
        font-size: 0.8rem;
        font-weight: 700;
    }

    .input-wrap {
        background: rgba(255,255,255,0.74);
        border: 1px solid rgba(138, 158, 176, 0.22);
        border-radius: 16px;
        padding: 0.5rem 0.5rem 0.35rem;
        box-shadow: 0 10px 24px rgba(15, 36, 56, 0.06);
    }

    [class*="st-key-law_questions_form"] {
        margin-top: 0.9rem;
        padding: 0.5rem 0.65rem 0.45rem;
        background: rgba(255,255,255,0.8);
        border: 1px solid rgba(60, 159, 232, 0.2);
        border-radius: 18px;
        box-shadow: 0 10px 24px rgba(15, 36, 56, 0.07);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    /* Fixed composer: stays attached to the bottom of the viewport and avoids overlapping the sidebar */
    [class*="st-key-legal-chat-composer"] {
        position: fixed;
        bottom: 1rem;
        z-index: 9999;
        /* Place to the right of the sidebar (default sidebar width ~18rem) and keep spacing from right edge */
        left: calc(18rem + 1rem);
        right: 1.5rem;
        max-width: calc(100% - (18rem + 2.5rem));
        padding: 0.35rem;
        background: rgba(243,247,251,0.92);
        border-radius: 20px;
        box-shadow: 0 8px 22px rgba(15, 36, 56, 0.08);
        backdrop-filter: blur(10px);
        box-sizing: border-box;
    }

    /* Ensure nested selector doesn't override fixed positioning */
    [class*="st-key-legal-chat-workspace"] [class*="st-key-legal-chat-composer"] {
        position: fixed;
    }

    [class*="st-key-legal-chat-composer"] [class*="st-key-law_questions_form"] {
        margin-top: 0;
    }

    /* Add bottom padding to chat panel so final message is not hidden under fixed composer */
    [class*="st-key-chat-panel"] {
        padding-bottom: calc(var(--composer-height) + var(--composer-gap));
    }

    [class*="st-key-chat-panel-empty"] {
        padding-bottom: 1rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
    }

    [class*="st-key-chat-panel-empty"] > [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex: 1 1 auto !important;
        flex-direction: column !important;
        align-items: stretch !important;
    }

    [class*="st-key-empty-chat-content"] {
        width: 100%;
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: stretch !important;
    }

    [class*="st-key-empty-chat-content"] > [data-testid="stVerticalBlock"] {
        width: 100%;
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: stretch !important;
    }

    [class*="st-key-legal-chat-flow"] {
        position: static !important;
        width: 100%;
        margin-top: 0.9rem;
        padding: 0;
        background: transparent;
        border-radius: 0;
        box-shadow: none;
        backdrop-filter: none;
    }

    @media (max-width: 768px) {
        [class*="st-key-legal-chat-composer"] {
            left: 1rem !important;
            right: 1rem !important;
            max-width: calc(100% - 2rem) !important;
            bottom: 0.6rem !important;
        }
        [class*="st-key-chat-panel"] {
            padding-bottom: calc(var(--composer-height) + var(--composer-gap));
        }
    }

    @media (max-width: 520px) {
        [class*="st-key-legal-chat-composer"] {
            left: 0.6rem !important;
            right: 0.6rem !important;
            bottom: 0.5rem !important;
        }
        [class*="st-key-chat-panel"] {
            padding-bottom: calc(var(--composer-height) + var(--composer-gap));
        }
    }

    [class*="st-key-law_questions_form"]:focus-within {
        border-color: rgba(60, 159, 232, 0.42);
        box-shadow: 0 12px 28px rgba(29, 115, 201, 0.11);
    }

    [class*="st-key-law_questions_form"] [data-testid="stTextInput"] input {
        border-color: rgba(138, 158, 176, 0.3) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.92) !important;
        min-height: 2.2rem !important;
        padding: 0.3rem 0.7rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    [class*="st-key-law_questions_form"] [data-testid="stTextInput"] input:hover {
        border-color: rgba(60, 159, 232, 0.45) !important;
    }

    [class*="st-key-law_questions_form"] [data-testid="stTextInput"] input:focus {
        border-color: var(--blue-500) !important;
        box-shadow: 0 0 0 2px rgba(60, 159, 232, 0.16) !important;
    }

    [class*="st-key-law_questions_form"] [data-testid="stFormSubmitButton"] button {
        min-height: 2.2rem !important;
        padding: 0.3rem 0.85rem !important;
        border-radius: 12px !important;
        background: var(--blue-600) !important;
        border: 1px solid var(--blue-600) !important;
        color: white !important;
        box-shadow: 0 6px 14px rgba(29, 115, 201, 0.16);
        transition: background 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease !important;
    }

    [class*="st-key-law_questions_form"] [data-testid="stFormSubmitButton"] button:hover {
        background: var(--blue-500) !important;
        border-color: var(--blue-500) !important;
        box-shadow: 0 8px 18px rgba(29, 115, 201, 0.24);
        transform: translateY(-1px);
    }

    [class*="st-key-law_questions_form"] [data-testid="stFormSubmitButton"] button:active {
        background: var(--blue-600) !important;
        box-shadow: 0 3px 8px rgba(29, 115, 201, 0.16);
        transform: translateY(1px);
    }

    .doc-card {
        background: white;
        border: 1px solid rgba(138, 158, 176, 0.2);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(15,36,56,0.04);
    }

    .pdf-tab-intro {
        background: linear-gradient(135deg, rgba(239,247,255,0.9), rgba(255,255,255,0.82));
        border: 1px solid rgba(60, 159, 232, 0.2);
        border-left: 4px solid var(--blue-500);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin: 0.25rem 0 1rem;
        color: var(--slate-600);
        line-height: 1.5;
    }

    [class*="st-key-pdf-selected-document"] {
        background: rgba(239,247,255,0.72);
        border: 1px solid rgba(60, 159, 232, 0.22);
        border-radius: 12px;
        padding: 0.65rem 0.85rem;
        margin: 0.4rem 0 0.75rem;
    }

    [class*="st-key-pdf-selected-document"] p {
        margin: 0;
        overflow-wrap: anywhere;
    }

    [class*="st-key-pdf-answer-card"] {
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(60, 159, 232, 0.2);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-top: 1rem;
        box-shadow: 0 8px 20px rgba(15, 36, 56, 0.06);
    }

    [data-testid="stTabs"] [role="tab"] {
        color: var(--slate-600);
        font-weight: 600;
        transition: color 0.2s ease, background 0.2s ease;
    }

    [data-testid="stTabs"] [role="tab"]:hover {
        color: var(--blue-600);
        background: rgba(60, 159, 232, 0.06);
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--blue-600);
    }

    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            min-width: 15rem;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 1rem 0.8rem;
        }

        .hero {
            margin-left: 0;
            margin-right: 0;
            padding: 1rem 1.1rem;
        }

        [class*="st-key-chat-panel"] {
            min-height: auto;
            padding: 0.7rem;
            padding-bottom: calc(var(--composer-height) + var(--composer-gap));
        }

        .empty-state {
            min-height: 240px;
        }

        .hero h1 {
            font-size: 1.7rem;
        }

        [class*="st-key-law_questions_form"] {
            padding: 0.45rem 0.5rem 0.4rem;
        }

        [class*="st-key-legal-chat-composer"] {
            bottom: 0.4rem;
            margin-left: -0.15rem;
            margin-right: -0.15rem;
            padding: 0.25rem;
        }

        [class*="st-key-empty-state-suggestions"] [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: 0.35rem;
        }

        [class*="st-key-empty-state-suggestions"] [data-testid="stColumn"] {
            flex: 1 1 calc(50% - 0.35rem);
            min-width: 0;
        }

        [class*="st-key-law_questions_form"] [data-testid="stHorizontalBlock"] {
            align-items: stretch;
            gap: 0.5rem;
        }

        [class*="st-key-law_questions_form"] [data-testid="stColumn"] {
            min-width: 0;
        }

        [data-testid="stFileUploader"] section {
            min-width: 0;
        }

        [data-testid="stTabs"] [role="tablist"] {
            overflow-x: auto;
            scrollbar-width: thin;
        }
    }

    @media (max-width: 520px) {
        .hero h1 {
            font-size: 1.45rem;
        }

        .hero p {
            font-size: 0.88rem;
        }

        .empty-state {
            padding: 0.9rem 0.5rem;
        }

        .empty-state h2 {
            font-size: 1.35rem;
        }

        [class*="st-key-empty-state-suggestions"] [data-testid="stColumn"] {
            flex-basis: 100%;
        }

        [class*="st-key-law_questions_form"] [data-testid="stHorizontalBlock"],
        [class*="st-key-pdf_questions_form"] [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }

        [class*="st-key-law_questions_form"] [data-testid="stColumn"],
        [class*="st-key-pdf_questions_form"] [data-testid="stColumn"] {
            flex-basis: 100% !important;
            width: 100% !important;
        }

        [class*="st-key-pdf_questions_form"] [data-testid="stFormSubmitButton"] button {
            min-height: 2.7rem;
        }

        [data-testid="stChatMessage"] {
            max-width: 100%;
        }

        .source-card {
            padding: 0.7rem 0.75rem;
        }
    }

    /* Composer size overrides for breakpoints */
    @media (max-width: 768px) {
        :root { --composer-height: 4.9rem; --composer-gap: 1rem; }
    }

    @media (max-width: 520px) {
        :root { --composer-height: 6rem; --composer-gap: 1.1rem; }
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
        "delete_popover_versions": {},
        "enable_web_search": False,   # Phase 8: Live Web Search toggle
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.chat_sessions_loaded:
        persisted_conversations = load_conversations()
        if persisted_conversations:
            st.session_state.chat_sessions = persisted_conversations
        st.session_state.chat_sessions_loaded = True

    if (
        st.session_state.chat_sessions
        and st.session_state.active_chat_id not in st.session_state.chat_sessions
    ):
        st.session_state.active_chat_id = next(reversed(st.session_state.chat_sessions))


def create_chat_session(title="New chat"):
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
    st.session_state.delete_popover_versions.pop(chat_id, None)

    if st.session_state.active_chat_id == chat_id:
        if st.session_state.chat_sessions:
            st.session_state.active_chat_id = next(
                reversed(st.session_state.chat_sessions)
            )
        else:
            st.session_state.active_chat_id = None


def has_real_conversations():
    return any(
        conversation["messages"]
        for conversation in st.session_state.chat_sessions.values()
    )


def get_real_conversations():
    return [
        (chat_id, conversation)
        for chat_id, conversation in reversed(list(st.session_state.chat_sessions.items()))
        if conversation["messages"]
    ]


def update_session_title(chat_id, title_text):
    session = st.session_state.chat_sessions.get(chat_id)
    if session is None:
        return
    if not session["messages"]:
        session["title"] = title_text or "New chat"
        return
    if session.get("title") not in (None, "", "New chat"):
        return
    try:
        first_user = next(msg for msg in session["messages"] if msg["role"] == "user")
        cleaned = " ".join(first_user["content"].split())
        session["title"] = (
            (cleaned[:28] + "...") if len(cleaned) > 28 else cleaned
        ) or "Conversation"
    except StopIteration:
        session["title"] = title_text or "Conversation"


def add_message(role, content, sources=None, provider_used=None, web_search_used=False):
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
        update_session_title(chat_id, None)
    save_conversation(chat_id, st.session_state.chat_sessions[chat_id])


def get_active_messages():
    chat_id = st.session_state.active_chat_id
    if chat_id not in st.session_state.chat_sessions:
        return []
    return st.session_state.chat_sessions[chat_id]["messages"]


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            pages = source["page_start"]
            if source["page_end"] != pages:
                pages = f"{pages}-{source['page_end']}"
            st.markdown(
                f'<div class="source-card"><strong>{source["law_name"]}</strong>'
                f'<small>Page {pages}</small>'
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
                st.caption("🌐 Answered from live web search · grounded on Tavily results")
            elif message.get("provider_used"):
                st.caption(f"Answered with {message['provider_used'].title()} · grounded context only")
            render_sources(message.get("sources", []))


def handle_legal_question(question, language):
    question = (question or "").strip()
    if not question:
        st.warning("Please enter a question first.")
        return

    add_message("user", question)
    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        '<div class="typing-indicator"><span></span><span></span><span></span></div>',
        unsafe_allow_html=True,
    )

    try:
        result = get_legal_answer(
            question,
            language=language,
            top_k=KB_TOP_K,
            enable_web_search=st.session_state.get("enable_web_search", False),
        )
    except Exception:
        typing_placeholder.empty()
        logger.exception("Unable to answer legal question")
        error_message = (
            "I couldn't complete that answer because the AI service is "
            "unavailable right now. Please try again in a moment."
        )
        st.error("I couldn't answer that right now. Please try again in a moment.")
        messages = get_active_messages()
        if (
            not messages
            or messages[-1].get("role") != "assistant"
            or messages[-1].get("content") != error_message
        ):
            add_message("assistant", error_message)
        return

    typing_placeholder.empty()
    answer_text = result.get("answer") or "I could not find enough relevant material to answer confidently."
    sources = result.get("sources", [])
    provider = result.get("provider_used")
    web_used = result.get("web_search_used", False)
    add_message("assistant", answer_text, sources=sources, provider_used=provider, web_search_used=web_used)


def set_question_from_example(question):
    st.session_state.legal_question_input = question


def get_document_id(file_bytes):
    """Return a stable identity for one uploaded file's contents."""

    return hashlib.sha256(file_bytes).hexdigest()


def get_document_options(documents):
    """Build readable, unique labels for content-addressed documents."""

    names = {}
    for document_id, document in documents.items():
        names.setdefault(document.name, []).append(document_id)

    options = {}
    for name, document_ids in names.items():
        for document_id in document_ids:
            label = name
            if len(document_ids) > 1:
                label = f"{name} ({document_id[:8]})"
            options[label] = document_id
    return options


init_state()

with st.container(key="main-header"):
    st.markdown(
        '<div class="hero"><h1>⚖️ RightsGuide AI</h1>'
        '<p>Clear legal guidance grounded in Pakistan-specific legal sources.</p></div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown(
        '<div class="brand-box"><div class="brand-name">⚖️ RightsGuide AI</div>'
        '<div class="brand-tag">Legal information, simplified and grounded in source material.</div></div>',
        unsafe_allow_html=True,
    )
    recent_conversations = get_real_conversations()
    if recent_conversations:
        st.markdown('<div class="sidebar-button">', unsafe_allow_html=True)
        if st.button("+ New Chat", key="new_chat_button", use_container_width=True):
            create_chat_session()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.caption("Recent conversations")
        for chat_id, conversation in recent_conversations:
            button_label = conversation["title"]
            with st.container(
                key=f"history-item-{chat_id}",
                horizontal=True,
                vertical_alignment="center",
                gap="small",
            ):
                if st.button(
                    button_label,
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    help="Open this conversation",
                ):
                    mark_chat_active(chat_id)
                    st.rerun()
                with st.popover(
                    "",
                    help="Delete this conversation",
                    width="content",
                    key=(
                        f"delete-popover-{chat_id}-"
                        f"{st.session_state.delete_popover_versions.get(chat_id, 0)}"
                    ),
                ):
                    st.caption("Delete this conversation?")
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button(
                            "Delete",
                            key=f"confirm_delete_{chat_id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            delete_chat_session(chat_id)
                            st.rerun()
                    with cancel_col:
                        if st.button(
                            "Cancel",
                            key=f"cancel_delete_{chat_id}",
                            use_container_width=True,
                        ):
                            st.session_state.delete_popover_versions[chat_id] = (
                                st.session_state.delete_popover_versions.get(chat_id, 0)
                                + 1
                            )
                            st.rerun()

    st.divider()
    with st.container(key="settings-panel"):
        st.subheader("Settings")
        st.session_state.language = st.selectbox(
            "Answer language",
            options=["roman_urdu", "english", "urdu"],
            index=["roman_urdu", "english", "urdu"].index(st.session_state.language),
            format_func=lambda item: {"roman_urdu": "Roman Urdu", "english": "English", "urdu": "Urdu script"}[item],
        )

        # ── Phase 8: Live Web Search toggle ──────────────────────────────
        st.markdown("---")
        st.session_state.enable_web_search = st.toggle(
            "🌐 Live Web Search",
            value=st.session_state.get("enable_web_search", False),
            help=(
                "When enabled, if local legal sources have low confidence "
                "the system will search the web (via Tavily) for a more "
                "current answer. Requires TAVILY_API_KEY in Streamlit Secrets."
            ),
        )
        if st.session_state.enable_web_search:
            st.info(
                "Live Web Search is ON. Web-sourced answers will carry the "
                "disclaimer: *⚠️ This information is retrieved from the web, please verify.*",
                icon="🌐",
            )

        st.markdown(
            '<div class="disclaimer-box">Qanoon Dost AI provides general legal information and guidance for informational purposes only. It is not a substitute for professional legal advice.</div>',
            unsafe_allow_html=True,
        )
    
with st.container(key="main-navigation"):
    tab_laws, tab_upload = st.tabs(["⚖️ Legal Chat", "📄 Ask Your PDF"])

with tab_laws:
    with st.container(key="legal-chat-workspace"):
        chat_messages = get_active_messages()
        chat_panel_key = "chat-panel" if chat_messages else "chat-panel-empty"
        with st.container(key=chat_panel_key):
            if chat_messages:
                for message_index, message in enumerate(chat_messages):
                    render_message(message, message_index)

            else:
                with st.container(key="empty-chat-content"):
                    st.markdown(
                        """
                        <div class="empty-state">
                            <div class="empty-state-icon">⚖️</div>
                            <h2>How can we help you today?</h2>
                            <p>Ask a legal question and get clear, Pakistan-specific guidance.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.container(key="empty-state-suggestions"):
                        suggestion_columns = st.columns(3)
                        suggestions = [
                            ("🏠 Tenant & landlord rights", "What are my rights as a tenant or landlord in Pakistan?"),
                            ("⚖️ Legal notices", "How should I respond to a legal notice in Pakistan?"),
                            ("👨‍⚖️ Criminal & civil rights", "What are my criminal and civil rights in Pakistan?"),
                        ]
                        for suggestion_index, (column, (label, question_text)) in enumerate(
                            zip(suggestion_columns, suggestions)
                        ):
                            with column:
                                st.button(
                                    label,
                                    key=f"example_question_{suggestion_index}",
                                    use_container_width=True,
                                    on_click=set_question_from_example,
                                    args=(question_text,),
                                )

        composer_key = "legal-chat-composer" if chat_messages else "legal-chat-flow"
        with st.container(key=composer_key):
            with st.form("law_questions_form", clear_on_submit=True):
                cols = st.columns([5, 1])

                with cols[0]:
                    question = st.text_input(
                        "Ask about Pakistani law",
                        placeholder="Example: My landlord is trying to evict me. What are my rights?",
                        label_visibility="collapsed",
                        key="legal_question_input",
                    )

                with cols[1]:
                    submitted = st.form_submit_button(
                        "Send",
                        use_container_width=True,
                        type="primary",
                    )

    if submitted:
        handle_legal_question(question, st.session_state.language)
        if question.strip():
            st.rerun()

    with st.container(key="knowledge-base-note"):
        st.markdown(
            '<p><strong>Knowledge base</strong> · Ask about Pakistan-specific legal rights '
            "and obligations using the built-in legal corpus.</p>",
            unsafe_allow_html=True,
        )

with tab_upload:
    st.subheader("Upload a private PDF")
    st.markdown(
        '<div class="pdf-tab-intro"><strong>Private document guidance</strong><br>'
        "Files are processed in memory for this session and are never added to the main legal database.</div>",
        unsafe_allow_html=True,
    )
    uploads = st.file_uploader(
        "Upload one or more text-based PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Maximum 15 MB per file. Scanned/image-only PDFs need OCR before upload.",
    )
    if uploads:
        if st.session_state.document_model is None:
            with st.spinner("Loading the document search model..."):
                st.session_state.document_model = load_embedding_model()

        for upload in uploads:
            file_bytes = upload.getvalue()
            document_id = get_document_id(file_bytes)
            if document_id not in st.session_state.uploaded_documents:
                try:
                    with st.spinner(f"Indexing {upload.name}..."):
                        st.session_state.uploaded_documents[document_id] = build_document(
                            upload.name,
                            file_bytes,
                            st.session_state.document_model,
                        )
                except ValueError as error:
                    st.error(f"{upload.name}: {error}")

        if st.session_state.uploaded_documents:
            document_options = get_document_options(
                st.session_state.uploaded_documents
            )
            selected_label = st.selectbox(
                "Document to ask about",
                list(document_options),
            )
            selected = document_options[selected_label]
            with st.container(key="pdf-selected-document"):
                st.markdown(
                    f"**Selected document**  \n`{st.session_state.uploaded_documents[selected].name}`"
                )
            with st.form("pdf_questions_form", clear_on_submit=True):
                cols = st.columns([5, 1])
                with cols[0]:
                    document_question = st.text_input(
                        "Question about this PDF",
                        value="",
                        placeholder="What does this document say about notice periods?",
                        label_visibility="collapsed",
                    )
                with cols[1]:
                    pdf_submit = st.form_submit_button("Ask", use_container_width=True, type="primary")

            if pdf_submit:
                question_text = (document_question or "").strip()
                if not question_text:
                    st.warning("Please enter a question first.")
                else:
                    try:
                        with st.spinner("Finding the most relevant passages..."):
                            result = get_document_answer(
                                st.session_state.uploaded_documents[selected],
                                question_text,
                                st.session_state.document_model,
                                language=st.session_state.language,
                                top_k=DOCUMENT_TOP_K,
                            )
                        with st.container(key="pdf-answer-card"):
                            st.markdown("**Answer from your private document**")
                            st.markdown(result["answer"])
                            if result.get("sources"):
                                render_sources(result["sources"])
                            st.caption(
                                f"Answered with {result['provider_used'].title()} · grounded context only"
                            )
                    except Exception as error:
                        st.error(f"Unable to answer right now: {error}")

            if st.button("Clear uploaded PDFs from this session", key="clear_documents"):
                st.session_state.uploaded_documents = {}
                st.rerun()
    else:
        st.info("Upload a PDF to create a temporary, private search index.")
