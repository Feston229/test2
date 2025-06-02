import pandas as pd
from app.settings import settings


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
