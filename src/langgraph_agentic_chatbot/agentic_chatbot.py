import ast
import json
import math
import operator
import os
import shutil
import sqlite3
from pathlib import Path

import requests
import yfinance as yf
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pypdf import PdfReader

from langfuse import get_client

# ============================================================
# Configuration
# ============================================================

load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

CHAT_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gemma4:e2b",
)

EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text:latest",
)

PDF_PATH = Path(
    os.getenv(
        "RAG_PDF_PATH",
        "my_paper.pdf",
    )
)

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "./chroma_db",
)

CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "agentic_chatbot_documents",
)


# ============================================================
# Ollama
# ============================================================

llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    validate_model_on_init=True,
)

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)


# ============================================================
# Langfuse
# ============================================================


def _create_langfuse_handler() -> CallbackHandler | None:
    """Create tracing only when Langfuse credentials are configured."""

    required_settings = (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    )

    if not all(os.getenv(setting) for setting in required_settings):
        return None

    get_client()

    return CallbackHandler()


langfuse_handler = _create_langfuse_handler()


# ============================================================
# RAG
# ============================================================


def _load_pdf_documents(
    pdf_path: Path,
) -> list[Document]:
    """Load a PDF page-by-page using pypdf."""

    if not pdf_path.exists():
        return []

    reader = PdfReader(pdf_path)

    documents = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_index,
                    "page_number": page_index + 1,
                },
            )
        )

    return documents


def _build_rag_vector_store() -> Chroma | None:
    """
    Build the Chroma index from the configured PDF.

    The old collection is removed before indexing so restarting the
    application does not continuously insert duplicate chunks.
    """

    documents = _load_pdf_documents(PDF_PATH)

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Rebuild the persistent index.
    #
    # This prevents:
    #
    # application run #1 -> 20 chunks
    # application run #2 -> 40 chunks
    # application run #3 -> 60 chunks
    #
    # because the same PDF should not be inserted repeatedly.

    chroma_directory = Path(CHROMA_PATH)

    if chroma_directory.exists():
        shutil.rmtree(chroma_directory)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PATH,
    )


vector_store = _build_rag_vector_store()

retriever = (
    vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,
        },
    )
    if vector_store
    else None
)


@tool
def rag_tool(query: str) -> str:
    """
    Search the configured PDF for information relevant to the user's question.

    Use this tool whenever the user asks about the loaded PDF, paper,
    document, notes, or information that should be grounded in that document.

    Args:
        query: A clear semantic-search query for the PDF.
    """

    if retriever is None:
        return (
            "RAG is currently unavailable because no PDF document "
            f"was loaded from '{PDF_PATH}'."
        )

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the loaded PDF."

    formatted_documents = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            str(PDF_PATH),
        )

        page_number = document.metadata.get(
            "page_number",
            "Unknown",
        )

        formatted_documents.append(
            f"[Retrieved chunk {index}]\n"
            f"Source: {source}\n"
            f"PDF page: {page_number}\n"
            f"Content:\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# ============================================================
# Web Search
# ============================================================


@tool
def web_search(
    query: str,
    max_results: int = 5,
) -> str:
    """
    Search the web with DuckDuckGo.

    Use for recent information, current events,
    or factual information requiring web access.
    """

    try:
        results = DDGS(timeout=10).text(
            query,
            max_results=max_results,
        )

        if not results:
            return "No search results found."

        formatted = []

        for index, item in enumerate(
            results,
            start=1,
        ):
            title = item.get(
                "title",
                "Untitled",
            )

            body = item.get(
                "body",
                "",
            )

            url = item.get("href") or item.get("url", "")

            formatted.append(f"{index}. {title}\n{body}\nURL: {url}")

        return "\n\n".join(formatted)

    except (
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return f"DuckDuckGo search error: {exc}"


# ============================================================
# Calculator
# ============================================================

_ALLOWED_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
}


def _safe_eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPS:
        return _ALLOWED_BINARY_OPS[type(node.op)](
            _safe_eval(node.left),
            _safe_eval(node.right),
        )

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_safe_eval(node.operand))

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = _ALLOWED_FUNCTIONS.get(node.func.id)

        if func is None:
            raise ValueError(f"Function '{node.func.id}' is not allowed")

        return func(*(_safe_eval(arg) for arg in node.args))

    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression locally.

    Examples:
    - 2 + 2
    - sqrt(16)
    - 10 * 5
    """

    try:
        return str(
            _safe_eval(
                ast.parse(
                    expression,
                    mode="eval",
                )
            )
        )

    except (
        ArithmeticError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as exc:
        return f"Calculation error: {exc}"


# ============================================================
# Stock Price
# ============================================================


@tool
def get_stock_price(symbol: str) -> str:
    """
    Get the latest available market price for a stock ticker
    such as AAPL or TSLA.
    """

    try:
        normalized_symbol = symbol.upper().strip()

        ticker = yf.Ticker(normalized_symbol)

        history = ticker.history(
            period="5d",
            interval="1d",
            auto_adjust=False,
        )

        if history.empty:
            return f"No market data found for {symbol}."

        last_row = history.iloc[-1]

        return json.dumps(
            {
                "symbol": normalized_symbol,
                "latest_close": round(
                    float(last_row["Close"]),
                    4,
                ),
                "currency_note": (
                    "Usually the exchange's quoted currency; verify if needed."
                ),
                "market_date": (history.index[-1].strftime("%Y-%m-%d")),
                "source": ("Yahoo Finance via yfinance"),
            }
        )

    except (
        KeyError,
        TypeError,
        ValueError,
        requests.RequestException,
    ) as exc:
        return f"Stock lookup error: {exc}"


# ============================================================
# Weather
# ============================================================

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@tool
def get_current_weather(
    location: str,
) -> str:
    """
    Get current weather for a city or location
    using Open-Meteo.

    No API key is required.
    """

    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=10,
        )

        geo_response.raise_for_status()

        results = geo_response.json().get("results", [])

        if not results:
            return f"Could not find location: {location}"

        place = results[0]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,"
                    "apparent_temperature,"
                    "relative_humidity_2m,"
                    "surface_pressure,"
                    "weather_code,"
                    "wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )

        weather_response.raise_for_status()

        data = weather_response.json()

        current = data["current"]

        units = data.get(
            "current_units",
            {},
        )

        display_name = ", ".join(
            part
            for part in (
                place.get("name"),
                place.get("admin1"),
                place.get("country"),
            )
            if part
        )

        code = current.get("weather_code")

        condition = WEATHER_CODES.get(
            code,
            f"Weather code {code}",
        )

        return (
            f"Current weather in {display_name}:\n"
            f"- Condition: {condition}\n"
            f"- Temperature: "
            f"{current.get('temperature_2m')} "
            f"{units.get('temperature_2m', '°C')}\n"
            f"- Feels like: "
            f"{current.get('apparent_temperature')} "
            f"{units.get('apparent_temperature', '°C')}\n"
            f"- Humidity: "
            f"{current.get('relative_humidity_2m')} "
            f"{units.get('relative_humidity_2m', '%')}\n"
            f"- Surface pressure: "
            f"{current.get('surface_pressure')} "
            f"{units.get('surface_pressure', 'hPa')}\n"
            f"- Wind speed: "
            f"{current.get('wind_speed_10m')} "
            f"{units.get('wind_speed_10m', 'km/h')}\n"
            f"- Observation time: "
            f"{current.get('time')}"
        )

    except requests.Timeout:
        return "Weather request timed out."

    except requests.RequestException as exc:
        return f"Weather service connection error: {exc}"

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        return f"Unexpected weather response: {exc}"


# ============================================================
# Tools
# ============================================================

tools = [
    rag_tool,
    web_search,
    calculator,
    get_stock_price,
    get_current_weather,
]

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are a helpful AI assistant with access to specialized tools.

Tool usage rules:

1. RAG / PDF
   If the user asks about the loaded PDF, paper, document,
   notes, or asks for information specifically from the
   provided document, use `rag_tool`.

   Base document-grounded answers on the information returned
   by `rag_tool`.

   Do not substitute your own general knowledge for information
   that the user explicitly asked to retrieve from the document.

   If the document does not contain enough information, clearly
   say so.

2. Web search
   Use `web_search` when the user asks for recent, current,
   changing, or web-specific information that cannot be reliably
   answered from your existing knowledge.

3. Calculator
   Use `calculator` when an exact mathematical calculation is
   required.

4. Stocks
   Use `get_stock_price` for current or latest available stock
   market prices.

5. Weather
   Use `get_current_weather` for current weather conditions.

General rules:

- Prefer the most specialized appropriate tool.
- Do not call tools unnecessarily.
- Never invent tool results.
- Never invent document quotations, sources, or page numbers.
- If a tool fails, explain the limitation rather than inventing
  an answer.
- After receiving tool results, answer the user's original
  question clearly and concisely.
"""


# ============================================================
# LangGraph
# ============================================================


def chat_node(
    state: MessagesState,
):
    response = llm_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
    )

    return {"messages": [response]}


# ============================================================
# SQLite Checkpointing
# ============================================================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False,
)

checkpoint = SqliteSaver(conn=conn)


def get_threads():
    """Return saved checkpoint records, newest first."""

    return checkpoint.list(None)


# ============================================================
# Build Graph
# ============================================================

graph = StateGraph(MessagesState)

graph.add_node(
    "chat_node",
    chat_node,
)

graph.add_node(
    "tools",
    ToolNode(tools),
)

graph.add_edge(
    START,
    "chat_node",
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge(
    "tools",
    "chat_node",
)

chatbot = graph.compile(checkpointer=checkpoint)


# ============================================================
# Observability
# ============================================================

if langfuse_handler:
    chatbot = chatbot.with_config(
        {
            "callbacks": [langfuse_handler],
            "run_name": "agentic-chatbot",
        }
    )


def flush_observability() -> None:
    """
    Flush pending Langfuse events before
    a short-lived process exits.
    """

    if langfuse_handler:
        get_client().flush()
