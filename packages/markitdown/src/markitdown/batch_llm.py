import os
import sys
import argparse
import time
from pathlib import Path

try:
    from ._markitdown import MarkItDown
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError:
    print("Please install required packages: pip install markitdown[llm]")
    sys.exit(1)

def get_clients():
    clients = []
    
    # 1. Google Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        clients.append({
            "name": "Gemini",
            "client": OpenAI(api_key=gemini_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
            "model": "gemini-2.0-flash" 
        })
        
    # 2. Groq
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        clients.append({
            "name": "Groq",
            "client": OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"),
            "model": "llama3-70b-8192"
        })
        
    # 3. OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        clients.append({
            "name": "OpenAI",
            "client": OpenAI(api_key=openai_key),
            "model": "gpt-4o-mini"
        })
        
    return clients

def main():
    parser = argparse.ArgumentParser(description="Batch convert PDFs to Markdown using MarkItDown with Multi-API LLM fallback.")
    parser.add_argument("directory", nargs="?", default=".", help="Directory containing PDF files (default: current directory)")
    parser.add_argument("--env", type=str, help="Path to a specific .env file to load (default: .env in current directory)")
    args = parser.parse_args()

    # Load environment variables
    if args.env:
        load_dotenv(args.env)
    else:
        load_dotenv()
    
    input_path = Path(args.directory).resolve()
    if not input_path.is_dir():
        print(f"Error: Directory {input_path} does not exist.")
        sys.exit(1)
        
    pdfs = list(input_path.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {input_path}")
        return
        
    print(f"Found {len(pdfs)} PDF files in {input_path}. Starting conversion...")
    
    clients = get_clients()
    
    if not clients:
        print("Warning: No API keys found (GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY).")
        print("Running MarkItDown WITHOUT LLM image/context analysis.")
        md = MarkItDown()
    else:
        print(f"Loaded {len(clients)} LLM providers for fallback: {', '.join(c['name'] for c in clients)}")

    for pdf in pdfs:
        output_file = pdf.with_suffix(".md")
        print(f"Converting {pdf.name}...", end=" ", flush=True)
        
        success = False
        
        if not clients:
            try:
                md = MarkItDown()
                result = md.convert(str(pdf))
                output_file.write_text(result.text_content, encoding="utf-8")
                print("Done (No LLM).")
            except Exception as e:
                print(f"Failed: {e}")
            continue

        for provider in clients:
            try:
                md_llm = MarkItDown(llm_client=provider["client"], llm_model=provider["model"])
                result = md_llm.convert(str(pdf))
                output_file.write_text(result.text_content, encoding="utf-8")
                print(f"Done (via {provider['name']}).")
                success = True
                break
            except Exception as e:
                print(f"\n  [!] {provider['name']} failed: {e}. Trying next provider...", end=" ", flush=True)
                time.sleep(2)
                
        if not success:
            print(f"\nFailed to convert {pdf.name} with all available providers.")

if __name__ == "__main__":
    main()
