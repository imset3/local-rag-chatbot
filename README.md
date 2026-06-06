# Local RAG Chatbot

Ollama, LangChain, Chroma, Streamlit으로 만든 로컬 PDF 질의응답 앱입니다. PDF를 업로드하면 문서를 chunk로 나누어 로컬 벡터 DB에 저장하고, 질문이 들어오면 관련 문서를 검색해 한국어 답변을 생성합니다.

## 주요 기능

- PDF 업로드 및 문서 인덱싱
- Chroma 기반 로컬 벡터 DB 저장
- 업로드 문서를 근거로 한 질의응답
- 최근 대화 이력을 반영한 후속 질문 처리
- 답변에 사용된 참고 문서 chunk 표시
- 사이드바에서 검색 개수, temperature, 벡터 DB 초기화 제어

## 실행 방법

```bash
ollama pull deepseek-r1:8b
ollama pull bge-m3
pip install -r requirements.txt
streamlit run app.py
```

Ollama 주소는 기본값으로 `http://127.0.0.1:11434`를 사용합니다. 다른 주소를 쓰려면 `OLLAMA_BASE_URL` 환경변수를 설정하세요.

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434
streamlit run app.py
```

## 사용 흐름

1. 사이드바에서 PDF 파일을 업로드합니다.
2. 앱이 PDF를 `data/`에 저장하고 `chroma_db/`에 인덱싱합니다.
3. 하단 채팅 입력창에서 문서에 대해 질문합니다.
4. 답변 아래의 참고 문서에서 검색에 사용된 일부 chunk를 확인합니다.
5. 새 문서 세트로 시작하려면 사이드바의 벡터 DB 초기화 버튼을 누릅니다.

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

## 로컬 데이터

- `data/` 안의 PDF 파일은 GitHub에 올리지 않습니다.
- `chroma_db/`는 실행 중 생성되는 벡터 DB라 Git에서 제외합니다.
- `.venv`, `.env`, `__pycache__`, `.streamlit/secrets.toml`도 `.gitignore`로 제외합니다.
