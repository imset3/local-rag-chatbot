# Local RAG Chatbot MVP

## 한줄 소개

Ollama 기반 DeepSeek-R1 모델을 사용하여 LangChain, Chroma, Streamlit으로 구현한 로컬 RAG 챗봇 프로젝트입니다. PDF 문서를 업로드하면 bge-m3 임베딩으로 벡터화하여 Chroma DB에 저장하고, 문서 기반 질의응답과 멀티턴 대화를 지원합니다.

## 프로젝트 개요

이 챗봇은 OpenAI API 키 없이 로컬 환경에서 실행됩니다. 사용자가 PDF를 업로드하면 LangChain이 문서를 chunk 단위로 나누고, Ollama의 `bge-m3` 임베딩 모델로 벡터화한 뒤 Chroma DB에 저장합니다. 질문이 들어오면 Chroma에서 관련 문서를 검색하고, 검색된 내용을 context로 넣어 Ollama의 `deepseek-r1:8b` 모델이 한국어 답변을 생성합니다.

## 주요 기능

- PDF 업로드 및 문서 인덱싱
- Chroma 기반 로컬 벡터 DB 저장
- 업로드 문서 기반 질의응답
- 최근 6개 메시지를 활용한 멀티턴 대화
- 답변 아래 참고 문서 chunk 표시
- Streamlit 사이드바에서 `top_k`, `temperature`, 벡터 DB 초기화 제어
- 문서에 없는 내용은 추측하지 않고 안내

## 실행 방법

```bash
ollama pull deepseek-r1:8b
ollama pull bge-m3
pip install -r requirements.txt
streamlit run app.py
```

## 기술 스택

- Python
- Streamlit
- LangChain
- Ollama
- DeepSeek-R1 8B
- bge-m3 Embeddings
- Chroma DB
- PyPDF

## 사용 방법

- 사이드바에서 PDF 파일을 업로드합니다.
- 업로드된 PDF는 `data/` 폴더에 저장되고 `chroma_db/`에 인덱싱됩니다.
- 하단 채팅 입력창에서 문서 기반 질문을 입력합니다.
- 답변 아래의 "참고 문서"에서 검색에 사용된 chunk 일부를 확인할 수 있습니다.
- 사이드바의 "벡터 DB 초기화" 버튼으로 기존 인덱스를 삭제할 수 있습니다.

## 프로젝트 구조

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── .gitkeep
└── chroma_db/        # 실행 중 자동 생성, Git 제외
```

## 주의사항

- `data/` 안의 실제 PDF 파일은 GitHub에 올리지 않습니다.
- `chroma_db/`는 로컬 실행 중 생성되는 벡터 DB이므로 GitHub에 올리지 않습니다.
- `.venv`, `.env`, `__pycache__`, `.streamlit/secrets.toml`은 `.gitignore`로 제외합니다.
