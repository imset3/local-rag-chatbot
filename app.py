import shutil
from pathlib import Path

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


DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


def inject_cyberpunk_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cyber-bg: #05070f;
            --cyber-panel: rgba(8, 13, 27, 0.84);
            --cyber-panel-strong: rgba(12, 18, 38, 0.94);
            --cyber-cyan: #22f7ff;
            --cyber-pink: #ff2bd6;
            --cyber-purple: #9b5cff;
            --cyber-text: #e8f7ff;
            --cyber-muted: #9fb4c7;
            --cyber-border: rgba(34, 247, 255, 0.36);
        }

        [data-testid="stHeader"], footer {
            visibility: hidden;
            height: 0;
        }

        .stApp {
            color: var(--cyber-text);
            background:
                radial-gradient(circle at 14% 16%, rgba(255, 43, 214, 0.18), transparent 30%),
                radial-gradient(circle at 88% 8%, rgba(34, 247, 255, 0.18), transparent 28%),
                radial-gradient(circle at 70% 86%, rgba(155, 92, 255, 0.16), transparent 32%),
                linear-gradient(135deg, #03040a 0%, #07111f 48%, #0c0820 100%);
        }

        .main .block-container {
            max-width: 1120px;
            padding-top: 2.2rem;
            padding-bottom: 6rem;
        }

        .cyber-hero {
            position: relative;
            overflow: hidden;
            padding: 1.35rem 1.45rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(34, 247, 255, 0.42);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(8, 13, 27, 0.92), rgba(18, 8, 33, 0.86)),
                repeating-linear-gradient(90deg, rgba(34, 247, 255, 0.08) 0 1px, transparent 1px 18px);
            box-shadow:
                0 0 24px rgba(34, 247, 255, 0.14),
                inset 0 0 24px rgba(255, 43, 214, 0.08);
        }

        .cyber-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            border-top: 2px solid rgba(255, 43, 214, 0.75);
            pointer-events: none;
            filter: drop-shadow(0 0 10px rgba(255, 43, 214, 0.7));
        }

        .cyber-kicker {
            margin: 0 0 0.45rem;
            color: var(--cyber-cyan);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            text-shadow: 0 0 12px rgba(34, 247, 255, 0.8);
        }

        .cyber-title {
            margin: 0;
            color: #ffffff;
            font-size: clamp(2rem, 5vw, 4.2rem);
            line-height: 0.98;
            font-weight: 900;
            letter-spacing: 0;
            text-shadow:
                0 0 12px rgba(34, 247, 255, 0.9),
                0 0 26px rgba(255, 43, 214, 0.42);
        }

        .cyber-subtitle {
            max-width: 780px;
            margin: 0.8rem 0 0;
            color: var(--cyber-muted);
            font-size: 1rem;
            line-height: 1.65;
        }

        .cyber-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 1.35rem;
        }

        .cyber-card {
            min-height: 96px;
            padding: 1rem;
            border: 1px solid rgba(155, 92, 255, 0.42);
            border-radius: 8px;
            background: rgba(8, 13, 27, 0.72);
            box-shadow: 0 0 18px rgba(155, 92, 255, 0.12);
        }

        .cyber-card strong {
            display: block;
            color: var(--cyber-pink);
            margin-bottom: 0.35rem;
            text-shadow: 0 0 12px rgba(255, 43, 214, 0.7);
        }

        .cyber-card span {
            color: var(--cyber-muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {
            color: var(--cyber-text);
            letter-spacing: 0;
            text-shadow: 0 0 14px rgba(34, 247, 255, 0.46);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(5, 7, 15, 0.98), rgba(14, 10, 31, 0.98));
            border-right: 1px solid rgba(34, 247, 255, 0.38);
            box-shadow: 10px 0 28px rgba(34, 247, 255, 0.08);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: var(--cyber-text);
        }

        [data-testid="stSidebar"] h2 {
            color: var(--cyber-cyan);
            text-transform: uppercase;
            font-size: 1rem;
            text-shadow: 0 0 12px rgba(34, 247, 255, 0.76);
        }

        .stButton button,
        [data-testid="stFileUploader"] button {
            border: 1px solid rgba(34, 247, 255, 0.65);
            border-radius: 8px;
            color: var(--cyber-text);
            background: linear-gradient(135deg, rgba(34, 247, 255, 0.12), rgba(255, 43, 214, 0.14));
            box-shadow: 0 0 12px rgba(34, 247, 255, 0.14);
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }

        .stButton button:hover,
        [data-testid="stFileUploader"] button:hover {
            border-color: var(--cyber-pink);
            color: #ffffff;
            box-shadow:
                0 0 18px rgba(255, 43, 214, 0.42),
                0 0 28px rgba(34, 247, 255, 0.18);
            transform: translateY(-1px);
        }

        [data-testid="stFileUploader"],
        [data-testid="stExpander"],
        [data-testid="stAlert"] {
            border: 1px solid rgba(34, 247, 255, 0.28);
            border-radius: 8px;
            background: rgba(8, 13, 27, 0.72);
            box-shadow: 0 0 18px rgba(34, 247, 255, 0.08);
        }

        [data-testid="stFileUploader"] section {
            border-color: rgba(255, 43, 214, 0.32);
            background: rgba(255, 255, 255, 0.025);
        }

        [data-testid="stSlider"] [role="slider"] {
            border-color: var(--cyber-cyan);
            box-shadow: 0 0 12px rgba(34, 247, 255, 0.78);
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--cyber-border);
            border-radius: 8px;
            background: rgba(8, 13, 27, 0.72);
            box-shadow:
                0 0 18px rgba(34, 247, 255, 0.10),
                inset 0 0 18px rgba(155, 92, 255, 0.05);
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            border-color: rgba(34, 247, 255, 0.45);
            box-shadow: 0 0 20px rgba(34, 247, 255, 0.14);
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            border-color: rgba(255, 43, 214, 0.42);
            box-shadow: 0 0 20px rgba(255, 43, 214, 0.12);
        }

        [data-testid="stChatInput"] {
            background: rgba(5, 7, 15, 0.74);
            border-top: 1px solid rgba(34, 247, 255, 0.25);
        }

        [data-testid="stChatInput"] textarea {
            color: var(--cyber-text);
            border: 1px solid rgba(34, 247, 255, 0.45);
            border-radius: 8px;
            background: rgba(7, 10, 21, 0.94);
            box-shadow: 0 0 16px rgba(34, 247, 255, 0.12);
        }

        [data-testid="stChatInput"] textarea:focus {
            border-color: var(--cyber-pink);
            box-shadow: 0 0 22px rgba(255, 43, 214, 0.22);
        }

        [data-testid="stChatInputSubmitButton"] {
            color: var(--cyber-cyan);
        }

        .stMarkdown code {
            color: var(--cyber-cyan);
            background: rgba(34, 247, 255, 0.08);
            border: 1px solid rgba(34, 247, 255, 0.22);
        }

        @media (max-width: 760px) {
            .main .block-container {
                padding-top: 1rem;
            }

            .cyber-card-grid {
                grid-template-columns: 1fr;
            }

            .cyber-title {
                font-size: 2.15rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Local RAG Chatbot", page_icon="📄", layout="wide")
inject_cyberpunk_theme()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()


@st.cache_resource(show_spinner=False)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def get_llm(temperature: float) -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        temperature=temperature,
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
    try:
        vectorstore = get_vectorstore()
        return vectorstore._collection.count() > 0
    except Exception:
        return False


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
답변은 과제 발표에 적합하게 명확하고 간결하게 작성하세요.""",
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

st.markdown(
    """
    <section class="cyber-hero">
        <p class="cyber-kicker">LOCAL DOCUMENT INTELLIGENCE NODE</p>
        <h1 class="cyber-title">LOCAL RAG // CYBERPUNK CHATBOT</h1>
        <p class="cyber-subtitle">
            Ollama DeepSeek-R1, bge-m3 embeddings, Chroma, and Streamlit로 구동되는
            로컬 PDF 기반 질의응답 콘솔입니다.
        </p>
    </section>
    <section class="cyber-card-grid">
        <div class="cyber-card">
            <strong>PDF INGEST</strong>
            <span>업로드한 문서를 chunk로 분할하고 로컬 벡터 DB에 저장합니다.</span>
        </div>
        <div class="cyber-card">
            <strong>LOCAL RAG</strong>
            <span>검색된 문서 context를 우선하여 한국어 답변을 생성합니다.</span>
        </div>
        <div class="cyber-card">
            <strong>MULTI-TURN</strong>
            <span>최근 대화 이력을 반영해 후속 질문까지 이어갑니다.</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("문서 설정")
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

    if uploaded_files:
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

    indexed_count = 0
    try:
        indexed_count = get_vectorstore()._collection.count()
    except Exception:
        indexed_count = 0
    st.info(f"현재 저장된 chunk 수: {indexed_count}")


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


# README
# ======
# 실행 방법:
# 1. ollama pull deepseek-r1:8b
# 2. ollama pull bge-m3
# 3. pip install -r requirements.txt
# 4. streamlit run app.py
#
# 사용 방법:
# - 사이드바에서 PDF 파일을 업로드하면 data/ 폴더에 저장되고 Chroma DB에 인덱싱됩니다.
# - 질문은 하단 채팅 입력창에 입력합니다.
# - 답변 아래의 "참고 문서"에서 검색에 사용된 chunk 일부를 확인할 수 있습니다.
# - 벡터 DB를 새로 만들고 싶으면 사이드바의 "벡터 DB 초기화" 버튼을 누릅니다.
