import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
ESSAY_URL = (
    "https://raw.githubusercontent.com/run-llama/llama_index/main"
    "/docs/examples/data/paul_graham/paul_graham_essay.txt"
)
# No pre-built golden dataset exists in the repo; it will be generated locally.
GOLDEN_DATASET_URL = None


def download_file(url: str, dest: Path) -> bool:
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        dest.write_bytes(response.content)
        print(f"Downloaded: {dest}")
        return True
    print(f"Failed to download {url} — HTTP {response.status_code}")
    return False


def generate_golden_dataset(essay_path: Path, output_path: Path) -> None:
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.evaluation import DatasetGenerator
    from llama_index.llms.anthropic import Anthropic

    print("Generating golden dataset from essay using Claude (ANTHROPIC_API_KEY)...")
    documents = SimpleDirectoryReader(input_files=[str(essay_path)]).load_data()
    llm = Anthropic(model="claude-haiku-4-5-20251001", api_key=os.environ["ANTHROPIC_API_KEY"])
    generator = DatasetGenerator.from_documents(
        documents, llm=llm, num_questions_per_chunk=2, show_progress=True
    )
    qa_dataset = generator.generate_dataset_from_nodes()
    qa_dataset.save_json(str(output_path))
    print(f"Golden dataset saved: {output_path} ({len(qa_dataset.queries)} questions)")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    essay_path = DATA_DIR / "paul_graham_essay.txt"
    download_file(ESSAY_URL, essay_path)

    golden_path = DATA_DIR / "paul_graham_golden_dataset.json"
    if not essay_path.exists():
        print("Essay not downloaded; cannot generate golden dataset.")
        return

    print("\nGenerating golden dataset from essay...")
    try:
        generate_golden_dataset(essay_path, golden_path)
    except ImportError as exc:
        print(
            f"Import error: {exc}\n"
            "Install with: pip install llama-index llama-index-llms-openai"
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"Generation failed: {exc}")


if __name__ == "__main__":
    main()
