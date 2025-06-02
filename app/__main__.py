import pandas as pd
import json
import sys
from ollama import ChatResponse, chat
from app.settings import settings
from app.utils import (
    get_csv_schema,
    check_and_pull_model,
    wait_for_ollama_connection,
)
from app.tools import query_csv_data


def main():
    # Check for command line argument
    if len(sys.argv) < 2:
        print("Usage: uv run --frozen --no-dev -m app 'Your question here'")
        print(
            "Example: uv run --frozen --no-dev -m app 'На какой платформе наибольшее количество фрилансеров - эксперты?'"
        )
        sys.exit(1)

    # Get user content from command line argument
    user_content = sys.argv[1]

    # === 0) Wait for Ollama and check/pull model ===
    if not wait_for_ollama_connection():
        print("Cannot proceed without Ollama connection")
        sys.exit(1)

    if not check_and_pull_model(settings.model):
        print(f"Cannot proceed without model '{settings.model}'")
        sys.exit(1)

    # === 1) Read CSV schema and build system prompt dynamically ===
    schema = get_csv_schema()

    # We can get the total number of rows in the CSV via pandas as well
    total_rows = len(pd.read_csv(settings.csv_path, encoding="utf-8"))

    system_content = (
        "You are a helpful assistant that answers user questions "
        "based on CSV file content. Use query_csv_data to fetch and analyze CSV data "
        f"CSV schema: {schema} CSV total rows: {total_rows} "
    )

    messages = [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    query_csv_data_tool = {
        "type": "function",
        "function": {
            "name": "query_csv_data",
            "description": "Queries and analyzes CSV data with filtering, grouping, and aggregation capabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "select": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'List of columns to select, e.g. ["Freelancer_ID", "Job_Category"]',
                    },
                    "where": {
                        "type": "object",
                        "description": 'Filter conditions. Supports equality: {"Platform": "Fiverr"}, lists: {"Platform": ["Fiverr", "Upwork"]}, and operators: {"Job_Completed": {"$lt": 100}, "Earnings_USD": {"$gte": 5000}}. Operators: $lt, $lte, $gt, $gte, $ne, $eq, $in, $nin',
                    },
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'Columns to group by, e.g. ["Payment_Method", "Platform"]',
                    },
                    "agg": {
                        "type": "object",
                        "description": 'Aggregation functions, e.g. {"Marketing_Spent": ["mean", "count"], "Job_Completed": "sum"}',
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Column name to sort results by",
                    },
                    "sort_desc": {
                        "type": "boolean",
                        "description": "Sort in descending order (default: false)",
                    },
                },
            },
        },
    }

    print(f"=== Processing query: {user_content} ===")
    response: ChatResponse = chat(
        model=settings.model,
        messages=messages,
        tools=[query_csv_data_tool],
    )

    # === 2) Check if the model wants to call a tool ===
    if not response.message.tool_calls:
        print("No function calls were returned by the model.\n")
        print("Assistant:", response.message.content)
        return

    # === 3) Invoke the requested function calls ===
    print("=== Model requested the following function calls: ===")
    for tool_call in response.message.tool_calls:
        func_name = tool_call.function.name
        func_args = tool_call.function.arguments or {}

        print(f" → Function to call: {func_name}")
        print("   Arguments:", func_args)

        if func_name == "query_csv_data":
            tool_output = query_csv_data(**func_args)
        else:
            print(f"ERROR: Unknown function '{func_name}'")
            tool_output = None

        print("   Function output:", tool_output)

        # 4) Append the assistant's 'function_call' message to messages
        messages.append({
            "role": "assistant",
            "content": None,
            "name": func_name,
            "arguments": func_args,
        })

        # 5) Append the 'tool' response so the model sees the function result
        messages.append({
            "role": "tool",
            "name": func_name,
            "content": json.dumps(tool_output, ensure_ascii=False),
        })

    # === 6) Send a follow-up chat to get the model's final answer ===
    final_response: ChatResponse = chat(model=settings.model, messages=messages)

    print("\n=== Final response from model: ===")
    print(final_response.message.content)


if __name__ == "__main__":
    main()
