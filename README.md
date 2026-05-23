# Local RAG Chatbot MVP

이 챗봇은 Ollama로 DeepSeek-R1 모델을 로컬에서 실행합니다.
사용자가 PDF를 업로드하면 LangChain이 문서를 chunk 단위로 나누고,
bge-m3 임베딩 모델로 벡터화한 뒤 Chroma DB에 저장합니다.
질문이 들어오면 Chroma에서 관련 문서를 검색하고,
검색된 내용을 context로 넣어 DeepSeek-R1이 답변을 생성합니다.
또한 Streamlit session_state를 사용해 최근 대화 이력을 유지하여 멀티턴 질문도 처리합니다.

## 실행 방법

```bash
ollama pull deepseek-r1:8b
ollama pull bge-m3
pip install -r requirements.txt
streamlit run app.py
```

## 사용 방법

- 사이드바에서 PDF 파일을 업로드합니다.
- 업로드된 PDF는 `data/` 폴더에 저장되고 `chroma_db/`에 인덱싱됩니다.
- 하단 채팅 입력창에서 문서 기반 질문을 입력합니다.
- 답변 아래의 "참고 문서"에서 검색에 사용된 chunk 일부를 확인할 수 있습니다.
- 사이드바의 "벡터 DB 초기화" 버튼으로 기존 인덱스를 삭제할 수 있습니다.
