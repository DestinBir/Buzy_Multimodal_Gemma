import json
import re
from datetime import datetime, timedelta
from typing import List, Optional

from src.rag import KnowledgeBase


BUSINESS_TOOLS = {
    "search_facts": {
        "description": "Search the knowledge base for facts matching a keyword. Use this to find specific information across all uploaded documents.",
        "parameters": {"keyword": "string"},
    },
    "get_supplier_info": {
        "description": "Get all information about a specific supplier from the knowledge base.",
        "parameters": {"supplier_name": "string"},
    },
    "get_date": {
        "description": "Get the current date and time. Use this when you need to calculate deadlines, contract expiry, or payment due dates.",
        "parameters": {},
    },
    "calculate": {
        "description": "Perform a basic arithmetic calculation. Use this for computing totals, percentages, VAT, exchange rates, or any numerical computation.",
        "parameters": {"expression": "string — a mathematical expression like '15000 * 0.18' or '(500000 + 75000) / 12'"},
    },
    "currency_convert": {
        "description": "Convert an amount between currencies. Useful for converting USD to XAF, CDF, NGN, etc.",
        "parameters": {"amount": "number", "from_currency": "string", "to_currency": "string"},
    },
}

TOOL_DESCRIPTIONS = "\n".join(
    f"- {name}: {info['description']} Params: {json.dumps(info['parameters'])}"
    for name, info in BUSINESS_TOOLS.items()
)


class Agent:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def run_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "search_facts":
            keyword = args.get("keyword", "")
            results = self.kb.retrieve(keyword, top_k=5)
            if not results:
                return f"No results found for '{keyword}'."
            return "\n".join(results)

        if tool_name == "get_supplier_info":
            name = args.get("supplier_name", "").lower()
            results = self.kb.retrieve(name, top_k=8)
            if not results:
                return f"No information found for supplier '{name}'."
            return "\n".join(results)

        if tool_name == "get_date":
            now = datetime.now()
            return (
                f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Day of week: {now.strftime('%A')}\n"
                f"ISO week: {now.isocalendar()[1]}"
            )

        if tool_name == "calculate":
            expr = args.get("expression", "")
            if not expr:
                return "No expression provided."
            try:
                allowed = re.sub(r"[^0-9+\-*/.()% ]", "", expr)
                result = eval(allowed, {"__builtins__": {}}, {})
                return f"Result: {expr} = {result}"
            except Exception as e:
                return f"Calculation error: {e}"

        if tool_name == "currency_convert":
            amount = args.get("amount", 0)
            from_cur = args.get("from_currency", "").upper()
            to_cur = args.get("to_currency", "").upper()

            rates = {
                "USD": 1.0, "XAF": 615.0, "XOF": 615.0,
                "CDF": 2850.0, "NGN": 1550.0, "KES": 130.0,
                "GHS": 15.0, "ZAR": 18.5, "EUR": 0.92,
                "GBP": 0.79, "MAD": 10.0, "EGP": 48.0,
            }
            if from_cur not in rates or to_cur not in rates:
                return (
                    f"Currency not supported. Available: {', '.join(sorted(rates.keys()))}\n"
                    "Rates are approximate and for reference only."
                )
            usd_amount = amount / rates[from_cur]
            result = usd_amount * rates[to_cur]
            return (
                f"{amount:,.2f} {from_cur} = {result:,.2f} {to_cur}\n"
                f"(Rate: 1 USD = {rates[from_cur]:,.2f} {from_cur}, "
                f"1 USD = {rates[to_cur]:,.2f} {to_cur})"
            )

        return f"Tool '{tool_name}' not found. Available: {', '.join(BUSINESS_TOOLS)}"

    @staticmethod
    def parse_tool_call(text: str):
        tool_pattern = re.compile(
            r'<tool_call>\s*\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*(\{.*?\})\s*\}\s*</tool_call>',
            re.DOTALL,
        )
        match = tool_pattern.search(text)
        if match:
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                args = {}
            return tool_name, args

        json_pattern = re.compile(
            r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*(\{.*?\})\s*\}', re.DOTALL
        )
        match = json_pattern.search(text)
        if match:
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                args = {}
            return tool_name, args

        return None, None
