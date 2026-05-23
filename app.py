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


st.set_page_config(page_title="Local RAG Chatbot", page_icon="📄", layout="wide")


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

st.title("PDF 기반 Local RAG Chatbot")
st.caption("Ollama DeepSeek-R1 + bge-m3 + Chroma + Streamlit")

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
