import json
from typing import Dict, Generator, List

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


SYSTEM_PROMPT = """You are a precise document assistant. Answer questions ONLY based on the provided context from the documents.

Rules:
1. Answer strictly from the retrieved document context below
2. If the answer is not in the context, respond with: "I could not find information about this in the provided documents."
3. Always be concise, accurate, and cite which document/section you found the answer in
4. Do not make up information or use knowledge outside the provided context
5. When referencing information, mention the source document name

Context from documents:
{context}
"""


class ChatService:
    def __init__(self, config):
        self.retrieval_k = config["RETRIEVAL_K"]
        self.memory_window = config["MEMORY_WINDOW"]
        self.sessions: Dict[str, List[Dict]] = {}

        key = (config.get("GEMINI_API_KEY") or "").strip()
        self.llm = None
        if key:
            self.llm = ChatGoogleGenerativeAI(
                model=config["LLM_MODEL"],
                temperature=config["LLM_TEMPERATURE"],
                google_api_key=key,
                convert_system_message_to_human=True,
                streaming=True,
            )

    def _get_session(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, [])

    def _save_to_session(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})
        max_messages = self.memory_window * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]

    def reset_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def get_history(self, session_id: str) -> List[Dict]:
        return self._get_session(session_id)

    def _build_messages(self, context: str, history: List[Dict], question: str) -> List:
        messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=question))
        return messages

    def _format_context(self, docs) -> tuple:
        context_parts = []
        sources = []
        for doc in docs:
            meta = doc.metadata
            filename = meta.get("source", "Unknown")
            page = meta.get("page", 0)
            chunk_idx = meta.get("chunk_index", 0)

            context_parts.append(
                f"[Source: {filename}, Page: {page + 1}, Chunk: {chunk_idx}]\n{doc.page_content}"
            )

            entry = {
                "filename": filename,
                "page": page + 1,
                "chunk_index": chunk_idx,
                "doc_id": meta.get("doc_id", ""),
            }
            if entry not in sources:
                sources.append(entry)

        return "\n\n---\n\n".join(context_parts), sources

    def chat_with_context(self, session_id: str, question: str, context: str, sources: List[Dict]) -> Dict:
        if self.llm is None:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        if not (context or "").strip():
            answer = "I could not find information about this in the provided documents."
            self._save_to_session(session_id, "user", question)
            self._save_to_session(session_id, "assistant", answer)
            return {"answer": answer, "sources": [], "session_id": session_id}

        history = self._get_session(session_id)
        messages = self._build_messages(context, history, question)

        response = self.llm.invoke(messages)
        answer = response.content

        self._save_to_session(session_id, "user", question)
        self._save_to_session(session_id, "assistant", answer)
        return {"answer": answer, "sources": sources, "session_id": session_id}

    def chat(self, session_id: str, question: str, vectorstore: FAISS) -> Dict:
        if self.llm is None:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.retrieval_k})
        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            answer = "I could not find information about this in the provided documents."
            self._save_to_session(session_id, "user", question)
            self._save_to_session(session_id, "assistant", answer)
            return {"answer": answer, "sources": [], "session_id": session_id}

        context, sources = self._format_context(relevant_docs)
        history = self._get_session(session_id)
        messages = self._build_messages(context, history, question)

        response = self.llm.invoke(messages)
        answer = response.content

        self._save_to_session(session_id, "user", question)
        self._save_to_session(session_id, "assistant", answer)
        return {"answer": answer, "sources": sources, "session_id": session_id}

    def chat_stream(self, session_id: str, question: str, vectorstore: FAISS) -> Generator:
        if self.llm is None:
            yield f"data: {json.dumps({'error': 'GEMINI_API_KEY is not configured', 'done': True})}\n\n"
            return

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.retrieval_k})
        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            answer = "I could not find information about this in the provided documents."
            self._save_to_session(session_id, "user", question)
            self._save_to_session(session_id, "assistant", answer)
            yield f"data: {json.dumps({'token': answer, 'done': False})}\n\n"
            yield f"data: {json.dumps({'sources': [], 'done': True})}\n\n"
            return

        context, sources = self._format_context(relevant_docs)
        history = self._get_session(session_id)
        messages = self._build_messages(context, history, question)

        full_answer = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content or ""
            full_answer += token
            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

        self._save_to_session(session_id, "user", question)
        self._save_to_session(session_id, "assistant", full_answer)
        yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"

    def chat_stream_with_context(
        self, session_id: str, question: str, context: str, sources: List[Dict]
    ) -> Generator:
        if self.llm is None:
            yield f"data: {json.dumps({'error': 'GEMINI_API_KEY is not configured', 'done': True})}\n\n"
            return

        if not (context or "").strip():
            answer = "I could not find information about this in the provided documents."
            self._save_to_session(session_id, "user", question)
            self._save_to_session(session_id, "assistant", answer)
            yield f"data: {json.dumps({'token': answer, 'done': False})}\n\n"
            yield f"data: {json.dumps({'sources': [], 'done': True})}\n\n"
            return

        history = self._get_session(session_id)
        messages = self._build_messages(context, history, question)

        full_answer = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content or ""
            full_answer += token
            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

        self._save_to_session(session_id, "user", question)
        self._save_to_session(session_id, "assistant", full_answer)
        yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"
