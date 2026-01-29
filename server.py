

from crewai import Crew, Agent, Task, LLM
from pathlib import Path
import importlib.util
import litserve as ls
from starlette.middleware.cors import CORSMiddleware

from rag_index import RAGIndex, RAGSettings, build_context

# LitServe enables MCP if the "mcp" package is installed, but it also needs "fastmcp".
# On some setups "mcp" exists without "fastmcp", which causes a NameError at startup.
if importlib.util.find_spec("fastmcp") is None:
    try:
        import litserve.server as ls_server
        ls_server._MCP_AVAILABLE = False
    except Exception:
        pass

# Local Ollama LLM (make sure Ollama is running and `ollama pull qwen3` is done).
# Default Ollama base URL is http://localhost:11434
llm = LLM(model="ollama/qwen3")


def _format_citations(results):
    citations = []
    for item in results:
        text = (item.get("text") or "").strip()
        snippet = text[:220] + ("..." if len(text) > 220 else "")
        citations.append(
            {
                "source": item.get("source_path", "unknown"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "score": item.get("score"),
                "rerank_score": item.get("rerank_score"),
                "snippet": snippet,
            }
        )
    return citations


class AgenticRAGAPI(ls.LitAPI):
    def setup(self, device):
        base_dir = Path(__file__).resolve().parent
        self.rag_settings = RAGSettings.from_env(base_dir)
        self.rag_index = RAGIndex.load(self.rag_settings)
        if not self.rag_index.available:
            print("RAG index not found or empty. Run: python ingest.py")

        assistant_agent = Agent(
            role="Assistant",
            goal="Answer the user's query using provided context and cite sources when possible.",
            backstory="You are a helpful assistant that answers questions grounded in local documents.",
            verbose=True,
            llm=llm,
        )

        answer_task = Task(
            description=(
                "Use the context to answer the question. "
                "If the context is empty or insufficient, say so.\n\n"
                "Context:\n{context}\n\nQuestion: {query}"
            ),
            expected_output="A concise and informative response grounded in the context",
            agent=assistant_agent,
        )

        self.crew = Crew(
            agents=[assistant_agent],
            tasks=[answer_task],
            verbose=True,
        )
        self.last_citations = []

    def decode_request(self, request):
        return request["query"]

    def predict(self, query):
        context = ""
        self.last_citations = []
        if self.rag_index and self.rag_index.available:
            results = self.rag_index.query(query)
            context = build_context(results, self.rag_settings)
            self.last_citations = _format_citations(results)
        return self.crew.kickoff(inputs={"query": query, "context": context})

    def encode_response(self, output):
        return {"output": output, "citations": self.last_citations}

if __name__ == "__main__":
    api = AgenticRAGAPI()
    server = ls.LitServer(
        api,
        workers_per_device=1,
        middlewares=[
            (
                CORSMiddleware,
                {
                    "allow_origins": ["*"],
                    "allow_methods": ["*"],
                    "allow_headers": ["*"],
                    "allow_credentials": False,
                },
            )
        ],
    )
    server.run(port=8000)
