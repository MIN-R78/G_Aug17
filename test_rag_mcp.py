### Min
import subprocess
import json
import os
import time
import signal

### load environment variables from .env if exists
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


def create_test_data():
    ### create test data directory and sample files for GitHub compatibility
    test_dir = "./test_data"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created test directory: {test_dir}")

    sample_pdf = os.path.join(test_dir, "sample.pdf")
    if not os.path.exists(sample_pdf):
        with open(sample_pdf, "w") as f:
            f.write("This is a sample PDF content for testing purposes.")
        print(f"Created sample PDF: {sample_pdf}")


def test_rag_mcp_server():
    ### start the server and test it
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = current_dir + os.pathsep + env.get("PYTHONPATH", "")

    if "TEST_PDF_PATH" not in env:
        env["TEST_PDF_PATH"] = "./test_data/sample.pdf"
    if "TEST_FOLDER_PATH" not in env:
        env["TEST_FOLDER_PATH"] = "./test_data"

    print("Starting RAG server...")
    print(f"Using test PDF: {env['TEST_PDF_PATH']}")
    print(f"Using test folder: {env['TEST_FOLDER_PATH']}")

    proc = subprocess.Popen(
        ["python", "rag_mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=current_dir,
        env=env
    )

    def send_request(request, timeout=60):
        ### send a request and get response back with timeout
        try:
            print(f"  Sending request to server...")
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            print(f"  Request sent, waiting for response...")

            start_time = time.time()
            while time.time() - start_time < timeout:
                if proc.stdout.readable():
                    response_line = proc.stdout.readline()
                    if response_line:
                        print(f"  Response received in {time.time() - start_time:.2f}s")
                        return json.loads(response_line)
                time.sleep(0.1)

            print(f"  Timeout after {timeout} seconds!")
            return {"error": f"Timeout after {timeout} seconds"}
        except Exception as e:
            print(f"  Error: {e}")
            return {"error": str(e)}

    try:
        ### test single PDF parsing
        print("\n=== Test 1: Single PDF Parsing ===")
        test_request = {
            "type": "execute_tool",
            "tool_name": "pdf_parser",
            "inputs": {
                "operation": "parse_pdf",
                "pdf_path": env["TEST_PDF_PATH"],
                "parser_type": "advanced",
                "chunk_size": 3
            }
        }
        print("Sending request:", json.dumps(test_request, indent=2))
        result = send_request(test_request)
        print("Result:", json.dumps(result, indent=2))

        ### test batch PDF parsing
        print("\n=== Test 2: Batch PDF Parsing ===")
        test_request = {
            "type": "execute_tool",
            "tool_name": "pdf_parser",
            "inputs": {
                "operation": "batch_parse_folder",
                "folder_path": env["TEST_FOLDER_PATH"],
                "parser_type": "advanced",
                "chunk_size": 3,
                "batch_size": 5
            }
        }
        print("Sending request:", json.dumps(test_request, indent=2))
        result = send_request(test_request)
        print("Result:", json.dumps(result, indent=2))

        ### test single text embedding
        print("\n=== Test 3: Single Text Embedding ===")
        test_texts = [
            "This is the first test text.",
            "This is the second test text.",
            "This is the third test text."
        ]
        test_request = {
            "type": "execute_tool",
            "tool_name": "embedding",
            "inputs": {
                "operation": "embed",
                "texts": test_texts,
                "model_name": "all-MiniLM-L6-v2"
            }
        }
        print("Sending request:", json.dumps(test_request, indent=2))
        result = send_request(test_request)
        print("Result:", json.dumps(result, indent=2))

        ### test batch text embedding
        print("\n=== Test 4: Batch Text Embedding ===")
        large_test_texts = [
            f"This is test text number {i} for batch processing."
            for i in range(1, 21)
        ]
        test_request = {
            "type": "execute_tool",
            "tool_name": "embedding",
            "inputs": {
                "operation": "batch_embed",
                "texts": large_test_texts,
                "model_name": "all-MiniLM-L6-v2",
                "batch_size": 5
            }
        }
        print("Sending request:", json.dumps(test_request, indent=2))
        result = send_request(test_request)
        print("Result:", json.dumps(result, indent=2))

        ### test single vector index creation
        print("\n=== Test 5: Single Vector Index Creation ===")
        test_request = {
            "type": "execute_tool",
            "tool_name": "embedding",
            "inputs": {
                "operation": "create_index",
                "texts": test_texts,
                "model_name": "all-MiniLM-L6-v2"
            }
        }
        print("Sending request:", json.dumps(test_request, indent=2))
        result = send_request(test_request)
        print("Result:", json.dumps(result, indent=2))

        ### test batch vector index creation
        print("\n=== Test 6: Batch Vector Index Creation ===")
        print("  This test may take longer due to model processing...")
        test_request = {
            "type": "execute_tool",
            "tool_name": "embedding",
            "inputs": {
                "operation": "batch_create_index",
                "texts": large_test_texts,
                "model_name": "all-MiniLM-L6-v2",
                "batch_size": 5
            }
        }
        print("Sending request:", json.dumps(test_request, indent=2))
        result = send_request(test_request)
        print("Result:", json.dumps(result, indent=2))

        ### test similarity search
        print("\n=== Test 7: Similarity Search ===")
        test_request = {
            "type": "execute_tool",
            "tool_name": "embedding",
            "inputs": {
                "operation": "search",
                "query": "test text",
                "top_k": 3
            }
        }
        print("Sending request:", json.dumps(test_request, indent=2))
        result = send_request(test_request)
        print("Result:", json.dumps(result, indent=2))

        ### test save and load index
        print("\n=== Test 8: Save and Load Index ===")
        test_request = {
            "type": "execute_tool",
            "tool_name": "embedding",
            "inputs": {
                "operation": "save_index",
                "save_path": "./test_index"
            }
        }
        print("Sending request:", json.dumps(test_request, indent=2))
        result = send_request(test_request)
        print("Result:", json.dumps(result, indent=2))

        ### test list tools
        print("\n=== Test 9: List Tools ===")
        test_request = {"type": "list_tools"}
        print("Sending request:", json.dumps(test_request, indent=2))
        result = send_request(test_request)
        print("Result:", json.dumps(result, indent=2))

        ### test server status
        print("\n=== Test 10: Server Status ===")
        test_request = {"type": "get_status"}
        print("Sending request:", json.dumps(test_request, indent=2))
        result = send_request(test_request)
        print("Result:", json.dumps(result, indent=2))

    finally:
        ### clean up
        print("\nStopping server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server stopped")


def test_tools_directly():
    ### test tools without starting server
    print("\n=== Testing Tools Directly ===")

    try:
        from pdf_parser_mcp import PDFParserTool
        from embedding_mcp import EmbeddingTool

        pdf_tool = PDFParserTool()
        embedding_tool = EmbeddingTool()

        ### check tool info
        print("PDF Tool Info:", pdf_tool.get_tool_info())
        print("Embedding Tool Info:", embedding_tool.get_tool_info())

        ### test embedding tool
        test_texts = ["Hello", "World"]
        result = embedding_tool.execute({
            "operation": "embed",
            "texts": test_texts
        })
        print("Embedding result:", result)

        ### test batch embedding
        large_texts = [f"Text {i}" for i in range(1, 21)]
        result = embedding_tool.execute({
            "operation": "batch_embed",
            "texts": large_texts,
            "batch_size": 5
        })
        print("Batch embedding result:", result)

        ### test PDF tool
        test_pdf_path = os.getenv("TEST_PDF_PATH", "./test_data/sample.pdf")
        result = pdf_tool.execute({
            "operation": "parse_pdf",
            "pdf_path": test_pdf_path,
            "parser_type": "advanced",
            "chunk_size": 3
        })
        print("PDF parsing result:", result)

    except ImportError as e:
        print(f"Import error: {e}")


if __name__ == "__main__":
    try:
        create_test_data()
        test_rag_mcp_server()
        test_tools_directly()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
### #%#