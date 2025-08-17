### Min
### Main server that combines PDF parser and embedding tools

from typing import Dict, Any, List
import json
import sys
from pdf_parser_mcp import PDFParserTool
from embedding_mcp import EmbeddingTool

class RAGMCPServer:
    def __init__(self):
        self.pdf_tool = PDFParserTool()
        self.embedding_tool = EmbeddingTool()
        self.tools = {
            self.pdf_tool.name: self.pdf_tool,
            self.embedding_tool.name: self.embedding_tool
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
### what tools we got
        return [
            tool.get_tool_info()
            for tool in self.tools.values()
        ]

    def get_tool_status(self) -> Dict[str, Any]:
### how are the tools doing
        return {
            tool_name: tool.get_status()
            for tool_name, tool in self.tools.items()
        }

    def execute_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
### run a tool with some inputs
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}

        tool = self.tools[tool_name]
        return tool.execute(inputs)

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
### figure out what they want and do it
        try:
            request_type = request.get("type")

            if request_type == "list_tools":
                return {
                    "type": "tools_list",
                    "tools": self.get_available_tools()
                }

            elif request_type == "get_tool_info":
                tool_name = request.get("tool_name")
                if tool_name in self.tools:
                    return {
                        "type": "tool_info",
                        "tool": self.tools[tool_name].get_tool_info()
                    }
                else:
                    return {"error": f"Tool '{tool_name}' not found"}

            elif request_type == "execute_tool":
                tool_name = request.get("tool_name")
                inputs = request.get("inputs", {})
                result = self.execute_tool(tool_name, inputs)
                return {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "result": result
                }

            elif request_type == "get_status":
                return {
                    "type": "status",
                    "server_status": "running",
                    "tools_status": self.get_tool_status()
                }

            else:
                return {"error": f"Unknown request type: {request_type}"}

        except Exception as e:
            return {"error": f"Request handling failed: {str(e)}"}

    def run(self):
### start the server and wait for requests
        print("Starting RAG MCP Server...")
        print("Available tools:", list(self.tools.keys()))
        print("Server ready. Send JSON requests to stdin.")

        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

### parse the request
                try:
                    request = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    response = {"error": f"Invalid JSON: {str(e)}"}
                    print(json.dumps(response))
                    continue

### do the work
                response = self.handle_request(request)
                print(json.dumps(response))

            except KeyboardInterrupt:
                print("Server interrupted by user")
                break
            except Exception as e:
                response = {"error": f"Unexpected error: {str(e)}"}
                print(json.dumps(response))

    def run_simple_mode(self):
### easier way to test stuff
        print("Starting RAG MCP Server in simple mode...")
        print("Available tools:", list(self.tools.keys()))

        while True:
            try:
                line = input("Enter request (or 'exit' to quit): ").strip()
                if line.lower() == "exit":
                    break

                if not line:
                    continue

                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON: {e}")
                    continue

                response = self.handle_request(request)
                print("Response:", json.dumps(response, indent=2))

            except KeyboardInterrupt:
                print("\nServer interrupted by user")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    server = RAGMCPServer()

### check if we want simple mode
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        server.run_simple_mode()
    else:
        server.run()
### #%#