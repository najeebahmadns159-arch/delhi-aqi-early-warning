import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from src.forecast import grap_stage

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

STAGE_ORDER = ["Stage I", "Stage II", "Stage III", "Stage IV"]
_cache = {}


def _db():
    # notebook 06 mein banaya hua chroma_grap_v2 hi load ho raha hai, dobara nahi banta
    if "db" not in _cache:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _cache["db"] = Chroma(persist_directory=str(ROOT / "chroma_grap_v2"), embedding_function=emb)
    return _cache["db"]


def _llm():
    if "llm" not in _cache:
        _cache["llm"] = ChatGroq(model=os.getenv("GROQ_MODEL"), temperature=0)
    return _cache["llm"]


def get_context(stage, question, k=10):
    # upar ke stage mein neeche ke stages ki actions bhi lagu hoti hain
    upto = STAGE_ORDER[: STAGE_ORDER.index(stage) + 1]
    db = _db()
    gen = db.similarity_search(question, k=2, filter={"stage": "General"})
    docs = db.similarity_search(question, k=k, filter={"stage": {"$in": upto}})
    return gen + docs


def answer(aqi_24h, question):
    stage = grap_stage(aqi_24h)
    if stage is None:
            return "Predicted AQI is below 200, so no GRAP stage is expected to apply."

    ctx = get_context(stage, question)
    context_text = "\n\n".join(
        f"[{d.metadata['stage']}, pdf page {d.metadata['page'] + 1}]\n{d.page_content}" for d in ctx
    )

    prompt = f"""you explain the GRAP document for delhi-ncr in simple english.
predicted 24-hour average AQI is {aqi_24h:.0f}, which means {stage} (already decided from the document thresholds).

rules:
- answer only from the context below. if something is not in the context, say it was not found.
- actions of lower stages continue along with the current stage, mention this when it matters.
- if the context lists exemptions or exceptions (projects or activities allowed to continue), always mention them together with the ban.
- the context comes from a pdf, so tables can look garbled. do not guess numbers or capacity limits from garbled table text. if unsure, say to check the table on that pdf page.
- keep the answer short, use bullet points.

context:
{context_text}

question: {question}"""

    out = _llm().invoke(prompt).content
    used = sorted({(d.metadata["stage"], d.metadata["page"] + 1) for d in ctx})
    return out + "\n\nsources used: " + ", ".join(f"{s} p{p}" for s, p in used)