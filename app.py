import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma_db")
LLM_MODEL = "deepseek-r1:8b"
EMBEDDING_MODEL = "bge-m3"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


def inject_app_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f6f7f9;
            --app-panel: #ffffff;
            --app-text: #1e293b;
            --app-muted: #64748b;
            --app-border: #d7dee8;
            --app-accent: #2563eb;
            --app-accent-soft: #dbeafe;
        }

        footer {
            visibility: hidden;
            height: 0;
        }

        [data-testid="stHeader"] {
            visibility: visible;
            background: rgba(246, 247, 249, 0.94);
            pointer-events: auto;
        }

        .stApp {
            color: var(--app-text);
            background: var(--app-bg);
        }

        .main .block-container {
            max-width: 1080px;
            padding-top: 1.6rem;
            padding-bottom: 6rem;
        }

        h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {
            color: var(--app-text);
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] {
            background: #eef2f7;
            border-right: 1px solid var(--app-border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: var(--app-text);
        }

        [data-testid="stSidebar"] h2 {
            color: var(--app-text);
            font-size: 1rem;
        }

        .stButton button,
        [data-testid="stFileUploader"] button {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            color: var(--app-text);
            background: var(--app-panel);
            transition: border-color 160ms ease, background 160ms ease;
        }

        .stButton button:hover,
        [data-testid="stFileUploader"] button:hover {
            border-color: var(--app-accent);
            color: var(--app-accent);
            background: var(--app-accent-soft);
        }

        [data-testid="stFileUploader"],
        [data-testid="stExpander"],
        [data-testid="stAlert"] {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: var(--app-panel);
        }

        [data-testid="stFileUploader"] section {
            border-color: var(--app-border);
            background: #f8fafc;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: var(--app-panel);
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            border-color: #bfdbfe;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            border-color: #cbd5e1;
        }

        [data-testid="stChatInput"] {
            background: rgba(246, 247, 249, 0.94);
            border-top: 1px solid var(--app-border);
        }

        [data-testid="stChatInput"] textarea {
            color: var(--app-text);
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: #ffffff;
        }

        [data-testid="stChatInput"] textarea:focus {
            border-color: var(--app-accent);
        }

        .stMarkdown code {
            color: #1d4ed8;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_app_theme()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()
    if "ollama_server_started" not in st.session_state:
        st.session_state.ollama_server_started = False


def is_ollama_server_running(timeout: float = 1.0) -> bool:
    try:
        with urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout) as response:
            return 200 <= response.status < 500
    except (OSError, URLError):
        return False


def ensure_ollama_server() -> tuple[bool, str]:
    if is_ollama_server_running():
        return True, "Ollama 서버가 실행 중입니다."

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        st.session_state.ollama_server_started = True
    except FileNotFoundError:
        return False, "Ollama 명령을 찾을 수 없습니다. Ollama 설치 상태를 확인해 주세요."
    except OSError as exc:
        return False, f"Ollama 서버 자동 실행에 실패했습니다: {exc}"

    for _ in range(20):
        if is_ollama_server_running(timeout=0.5):
            return True, "Ollama 서버를 자동으로 실행했습니다."
        time.sleep(0.5)

    return False, "Ollama 서버 응답이 없습니다. `ollama serve`를 수동으로 실행해 주세요."


@st.cache_resource(show_spinner=False)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def get_llm(temperature: float) -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        temperature=temperature,
        base_url=OLLAMA_BASE_URL,
    )


def reset_vector_db() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(exist_ok=True)
    st.session_state.indexed_files = set()


def save_uploaded_pdf(uploaded_file) -> Path:
    file_path = DATA_DIR / uploaded_file.name
    with file_path.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def index_pdf(file_path: Path) -> int:
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = file_path.name

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def has_indexed_documents() -> bool:
    return get_indexed_chunk_count() > 0


def get_indexed_chunk_count() -> int:
    try:
        vectorstore = get_vectorstore()
        return int(vectorstore._collection.count())
    except Exception:
        return 0


def format_recent_history(messages: list[dict], limit: int = 6) -> str:
    recent_messages = messages[-limit:]
    if not recent_messages:
        return "이전 대화 없음"

    formatted = []
    for message in recent_messages:
        role = "사용자" if message["role"] == "user" else "assistant"
        formatted.append(f"{role}: {message['content']}")
    return "\n".join(formatted)


def format_context(docs) -> str:
    if not docs:
        return "검색된 문서 내용 없음"

    context_blocks = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        context_blocks.append(
            f"[문서 {index}] source={source}{page_text}\n{doc.page_content}"
        )
    return "\n\n".join(context_blocks)


def render_references(docs) -> None:
    if not docs:
        return

    with st.expander("참고 문서", expanded=True):
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            page_label = f" / p.{page + 1}" if isinstance(page, int) else ""
            preview = doc.page_content[:700].strip()
            if len(doc.page_content) > 700:
                preview += "..."

            st.markdown(f"**Chunk {index} - {source}{page_label}**")
            st.write(preview)


def answer_question(question: str, top_k: int, temperature: float) -> tuple[str, list]:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 PDF 문서 기반 RAG 챗봇입니다.
답변은 반드시 한국어로 작성하세요.
업로드된 문서 내용을 최우선 근거로 사용하세요.
문서에서 찾을 수 없는 내용은 추측하지 말고 "제공된 문서에서는 확인할 수 없습니다."라고 답하세요.
답변은 업무 문서처럼 명확하고 간결하게 작성하세요.""",
            ),
            (
                "human",
                """[최근 대화 이력]
{history}

[검색된 문서 context]
{context}

[사용자 질문]
{question}

위 문서 context와 최근 대화 이력을 참고하여 답변하세요.""",
            ),
        ]
    )

    chain = prompt | get_llm(temperature)
    response = chain.invoke(
        {
            "history": format_recent_history(st.session_state.messages),
            "context": format_context(docs),
            "question": question,
        }
    )

    return response.content, docs


init_session_state()

st.title("Local RAG Chatbot")
st.caption("PDF를 로컬에서 인덱싱하고, 업로드한 문서를 근거로 질문에 답합니다.")

status_cols = st.columns(3)
status_cols[0].metric("LLM", LLM_MODEL)
status_cols[1].metric("Embedding", EMBEDDING_MODEL)
status_cols[2].metric("저장된 chunk", get_indexed_chunk_count())

with st.sidebar:
    st.header("문서 설정")
    ollama_ready, ollama_status = ensure_ollama_server()
    if ollama_ready:
        st.success(ollama_status)
    else:
        st.warning(ollama_status)

    uploaded_files = st.file_uploader(
        "PDF 업로드",
        type=["pdf"],
        accept_multiple_files=True,
    )

    top_k = st.slider("검색 문서 수 top_k", min_value=1, max_value=10, value=4)
    temperature = st.slider(
        "temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )

    if st.button("벡터 DB 초기화", type="secondary"):
        reset_vector_db()
        st.success("벡터 DB를 초기화했습니다.")

    if uploaded_files and not ollama_ready:
        st.warning("문서 인덱싱 전에 Ollama 서버 연결이 필요합니다.")
    elif uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.indexed_files:
                continue

            try:
                with st.spinner(f"{uploaded_file.name} 인덱싱 중..."):
                    saved_path = save_uploaded_pdf(uploaded_file)
                    chunk_count = index_pdf(saved_path)
                    st.session_state.indexed_files.add(uploaded_file.name)
                st.success(f"문서 인덱싱 완료: {uploaded_file.name} ({chunk_count} chunks)")
            except Exception as exc:
                st.error(f"PDF 인덱싱 중 오류가 발생했습니다: {exc}")

    st.info(f"현재 저장된 chunk 수: {get_indexed_chunk_count()}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("docs"):
            render_references(message["docs"])


user_input = st.chat_input("업로드한 PDF에 대해 질문하세요")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if not any(DATA_DIR.glob("*.pdf")):
            answer = "먼저 사이드바에서 PDF 문서를 업로드해 주세요."
            st.warning(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        elif not has_indexed_documents():
            answer = "Chroma DB가 비어 있습니다. 먼저 PDF 문서를 인덱싱해 주세요."
            st.warning(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        elif not ollama_ready:
            answer = ollama_status
            st.warning(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            try:
                with st.spinner("문서를 검색하고 답변을 생성하는 중..."):
                    answer, docs = answer_question(user_input, top_k, temperature)
                st.markdown(answer)
                render_references(docs)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "docs": docs,
                    }
                )
            except Exception as exc:
                answer = (
                    "Ollama 연결 오류가 발생했을 수 있습니다. "
                    "`ollama serve`가 실행 중인지, "
                    "`deepseek-r1:8b`와 `bge-m3` 모델이 설치되어 있는지 확인해 주세요.\n\n"
                    f"오류 내용: {exc}"
                )
                st.error(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
