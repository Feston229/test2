from ollama import list as ollama_list, pull as ollama_pull
import pandas as pd
from app.settings import settings
import time


def check_and_pull_model(model_name: str) -> bool:
    """
    Check if the specified model exists locally, and pull it if it doesn't.

    Args:
        model_name: Name of the model to check/pull

    Returns:
        bool: True if model is available, False if failed to pull
    """
    try:
        # Get list of available models
        models = ollama_list()
        available_models = [model.model for model in models.models]

        # Check if our model is in the list
        model_exists = any(model_name in model for model in available_models)

        if model_exists:
            print(f"✓ Model '{model_name}' is already available locally")
            return True
        else:
            print(
                f"⚠ Model '{model_name}' not found locally. Pulling from repository..."
            )

            # Pull the model
            pull_response = ollama_pull(model_name)

            # The pull function returns a generator, so we need to consume it
            print("Downloading model (this may take a while)...")
            for chunk in pull_response:
                if "status" in chunk:
                    status = chunk["status"]
                    if "total" in chunk and "completed" in chunk:
                        total = chunk["total"]
                        completed = chunk["completed"]
                        percentage = (completed / total) * 100 if total > 0 else 0
                        print(f"\r{status}: {percentage:.1f}%", end="", flush=True)
                    else:
                        print(f"\r{status}", end="", flush=True)

            print(f"\n✓ Model '{model_name}' pulled successfully!")
            return True

    except Exception as e:
        print(f"✗ Error checking/pulling model '{model_name}': {str(e)}")
        print("Please make sure Ollama is running and the model name is correct.")
        return False


def wait_for_ollama_connection(max_retries: int = 30, delay: int = 2) -> bool:
    """
    Wait for Ollama to be available by trying to list models.

    Args:
        max_retries: Maximum number of connection attempts
        delay: Delay between attempts in seconds

    Returns:
        bool: True if connection successful, False if timed out
    """
    print("Checking Ollama connection...")

    for attempt in range(max_retries):
        try:
            ollama_list()
            print("✓ Connected to Ollama successfully")
            return True
        except Exception as exc:
            if attempt == 0:
                print(
                    f"⚠ Ollama not ready yet, waiting... (attempt {attempt + 1}/{max_retries})"
                )
            elif attempt < max_retries - 1:
                print(
                    f"⚠ Still waiting for Ollama... (attempt {attempt + 1}/{max_retries})"
                )
            time.sleep(delay)

    print("✗ Failed to connect to Ollama after maximum retries")
    print("Please ensure Ollama is running and accessible")
    return False


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
