import ast
import json
import math
import operator
import os
import sqlite3
from typing import Annotated, TypedDict

import requests
import yfinance as yf
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langfuse import get_client

load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)


def _create_langfuse_handler() -> CallbackHandler | None:
    """Create tracing only when credentials for a Langfuse instance are configured."""
    required_settings = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    if not all(os.getenv(setting) for setting in required_settings):
        return None
    get_client()
    return CallbackHandler()


langfuse_handler = _create_langfuse_handler()


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web with DuckDuckGo. Use for recent or factual web information."""
    try:
        results = DDGS(timeout=10).text(query, max_results=max_results)
        if not results:
            return "No search results found."
        formatted = []
        for index, item in enumerate(results, start=1):
            title = item.get("title", "Untitled")
            body = item.get("body", "")
            url = item.get("href") or item.get("url", "")
            formatted.append(f"{index}. {title}\n{body}\nURL: {url}")
        return "\n\n".join(formatted)
    except (RuntimeError, TypeError, ValueError) as exc:
        return f"DuckDuckGo search error: {exc}"


_ALLOWED_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
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
            _safe_eval(node.left), _safe_eval(node.right)
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
    """Calculate a mathematical expression locally. Examples: 2+2, sqrt(16), 10*5."""
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval")))
    except (ArithmeticError, SyntaxError, TypeError, ValueError) as exc:
        return f"Calculation error: {exc}"


@tool
def get_stock_price(symbol: str) -> str:
    """Get the latest available market price for a stock ticker such as AAPL or TSLA."""
    try:
        ticker = yf.Ticker(symbol.upper().strip())
        history = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if history.empty:
            return f"No market data found for {symbol}."
        last_row = history.iloc[-1]
        return json.dumps(
            {
                "symbol": symbol.upper().strip(),
                "latest_close": round(float(last_row["Close"]), 4),
                "currency_note": "Usually the exchange's quoted currency; verify if needed.",
                "market_date": history.index[-1].strftime("%Y-%m-%d"),
                "source": "Yahoo Finance via yfinance",
            }
        )
    except (KeyError, TypeError, ValueError, requests.RequestException) as exc:
        return f"Stock lookup error: {exc}"


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
def get_current_weather(location: str) -> str:
    """Get current weather for a city or location using Open-Meteo. No API key required."""
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
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
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "surface_pressure,weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        data = weather_response.json()
        current = data["current"]
        units = data.get("current_units", {})
        display_name = ", ".join(
            part
            for part in (place.get("name"), place.get("admin1"), place.get("country"))
            if part
        )
        code = current.get("weather_code")
        condition = WEATHER_CODES.get(code, f"Weather code {code}")
        return (
            f"Current weather in {display_name}:\n"
            f"- Condition: {condition}\n"
            f"- Temperature: {current.get('temperature_2m')} {units.get('temperature_2m', '°C')}\n"
            f"- Feels like: {current.get('apparent_temperature')} {units.get('apparent_temperature', '°C')}\n"
            f"- Humidity: {current.get('relative_humidity_2m')} {units.get('relative_humidity_2m', '%')}\n"
            f"- Surface pressure: {current.get('surface_pressure')} {units.get('surface_pressure', 'hPa')}\n"
            f"- Wind speed: {current.get('wind_speed_10m')} {units.get('wind_speed_10m', 'km/h')}\n"
            f"- Observation time: {current.get('time')}"
        )
    except requests.Timeout:
        return "Weather request timed out."
    except requests.RequestException as exc:
        return f"Weather service connection error: {exc}"
    except (KeyError, TypeError, ValueError) as exc:
        return f"Unexpected weather response: {exc}"


tools = [web_search, calculator, get_stock_price, get_current_weather]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn=conn)


def get_threads():
    """Return saved checkpoint records, newest first."""
    return checkpoint.list(None)


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)
if langfuse_handler:
    chatbot = chatbot.with_config(
        {"callbacks": [langfuse_handler], "run_name": "agentic-chatbot"}
    )


def flush_observability() -> None:
    """Flush pending Langfuse events before a short-lived process exits."""
    if langfuse_handler:
        get_client().flush()
