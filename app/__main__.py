import pandas as pd
import json
from ollama import ChatResponse, chat
from app.settings import settings


def get_csv_schema() -> dict:
    """
    Returns schema information about the CSV file including:
    - Column names and types
    - For string columns: unique values
    - For numeric columns: min/max values
    """
    df = pd.read_csv(settings.csv_path, encoding="utf-8")
    schema = {}

    for col in df.columns:
        col_info = {
            "type": str(df[col].dtype),
        }

        # For string/object columns - get unique values
        if df[col].dtype == "object" or df[col].dtype.name == "string":
            unique_vals = df[col].dropna().unique().tolist()
            # Limit to reasonable number of unique values
            if len(unique_vals) <= 50:
                col_info["unique_values"] = unique_vals
            else:
                col_info["unique_count"] = len(unique_vals)
                col_info["sample_values"] = unique_vals[:10]  # Show first 10 as sample

        # For numeric columns - get min/max/mean
        elif df[col].dtype in ["int64", "float64", "int32", "float32"]:
            col_info["min"] = df[col].min()
            col_info["max"] = df[col].max()
            col_info["mean"] = round(df[col].mean(), 2)

        schema[col] = col_info

    return schema


def query_csv_data(
    select: list[str] = None,
    where: dict = None,
    group_by: list[str] = None,
    agg: dict = None,
    sort_by: str = None,
    sort_desc: bool = False,
) -> list[dict]:
    """
    Advanced CSV querying with filtering, grouping, and aggregation capabilities.

    Args:
        select: List of columns to select
        where: Dictionary for filtering, supports:
               - Direct equality: {"Payment_Method": "Crypto"}
               - List membership: {"Client_Region": ["Asia", "Europe"]}
               - Comparison operators: {"Job_Completed": {"$lt": 100}, "Earnings_USD": {"$gte": 5000}}
        group_by: List of columns to group by
        agg: Dictionary of aggregations, e.g. {"Income": ["mean", "count"], "Projects_Completed": "sum"}
        sort_by: Column to sort by
        sort_desc: Sort in descending order

    Returns:
        List of dictionaries with query results
    """

    df = pd.read_csv(settings.csv_path, encoding="utf-8")

    # Apply WHERE filtering
    if where:
        for col, value in where.items():
            if col in df.columns:
                if isinstance(value, dict):
                    # Handle comparison operators
                    for operator, operand in value.items():
                        if operator == "$lt":
                            df = df[df[col] < operand]
                        elif operator == "$lte":
                            df = df[df[col] <= operand]
                        elif operator == "$gt":
                            df = df[df[col] > operand]
                        elif operator == "$gte":
                            df = df[df[col] >= operand]
                        elif operator == "$ne":
                            df = df[df[col] != operand]
                        elif operator == "$eq":
                            df = df[df[col] == operand]
                        elif operator == "$in":
                            df = df[df[col].isin(operand)]
                        elif operator == "$nin":
                            df = df[~df[col].isin(operand)]
                        else:
                            print(
                                f"Warning: Unsupported operator '{operator}' for column '{col}'"
                            )
                elif isinstance(value, list):
                    df = df[df[col].isin(value)]
                else:
                    df = df[df[col] == value]

    # Handle GROUP BY with aggregations
    if group_by and agg:
        # Group by specified columns
        grouped = df.groupby(group_by)

        # Apply aggregations
        agg_results = {}
        for col, funcs in agg.items():
            if col in df.columns:
                if isinstance(funcs, str):
                    funcs = [funcs]
                for func in funcs:
                    if func == "mean":
                        agg_results[f"{col}_mean"] = grouped[col].mean()
                    elif func == "count":
                        agg_results[f"{col}_count"] = grouped[col].count()
                    elif func == "sum":
                        agg_results[f"{col}_sum"] = grouped[col].sum()
                    elif func == "min":
                        agg_results[f"{col}_min"] = grouped[col].min()
                    elif func == "max":
                        agg_results[f"{col}_max"] = grouped[col].max()
                    elif func == "std":
                        agg_results[f"{col}_std"] = grouped[col].std()

        # Combine results
        result_df = pd.DataFrame(agg_results).reset_index()

    # Handle simple aggregations without grouping
    elif agg and not group_by:
        agg_results = {}
        for col, funcs in agg.items():
            if col in df.columns:
                if isinstance(funcs, str):
                    funcs = [funcs]
                for func in funcs:
                    if func == "mean":
                        agg_results[f"{col}_mean"] = [df[col].mean()]
                    elif func == "count":
                        agg_results[f"{col}_count"] = [df[col].count()]
                    elif func == "sum":
                        agg_results[f"{col}_sum"] = [df[col].sum()]
                    elif func == "min":
                        agg_results[f"{col}_min"] = [df[col].min()]
                    elif func == "max":
                        agg_results[f"{col}_max"] = [df[col].max()]
                    elif func == "std":
                        agg_results[f"{col}_std"] = [df[col].std()]

        result_df = pd.DataFrame(agg_results)

    else:
        # Regular selection and filtering
        result_df = df
        if select:
            missing_cols = [col for col in select if col not in df.columns]
            if missing_cols:
                return []
            result_df = result_df[select]

    # Apply sorting
    if sort_by and sort_by in result_df.columns:
        result_df = result_df.sort_values(sort_by, ascending=not sort_desc)

    return result_df.to_dict(orient="records")


def main():
    # === 1) Read CSV schema and build system prompt dynamically ===
    schema = get_csv_schema()

    # We can get the total number of rows in the CSV via pandas as well
    total_rows = len(pd.read_csv(settings.csv_path, encoding="utf-8"))

    system_content = (
        "You are a helpful assistant that answers user questions "
        "based on CSV file content. Use query_csv_data to fetch and analyze CSV data "
        f"CSV schema: {schema} CSV total rows: {total_rows} "
    )
    # user_content = "На какой платформе наибольшее количество фрилансеров - эксперты ?"
    # user_content = "Как платформа влияет на заработок фрилансеров ?"
    user_content = "Какой процент фрилансеров, считающих себя экспертами, выполнил менее 100 проектов?"

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

    print("=== Sending initial prompt to Ollama ===")
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
        print(response.message)
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

    # === 6) Send a follow-up chat to get the model’s final answer ===
    final_response: ChatResponse = chat(model=settings.model, messages=messages)

    print("\n=== Final response from model: ===")
    print(final_response.message.content)


if __name__ == "__main__":
    main()
