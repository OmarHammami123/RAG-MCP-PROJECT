import asyncio
import json
from typing import Dict, Any, List
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class MCPClient:
    """MCP client for communicating with filesystem server"""
    
    def __init__(self):
        self.session = None
    
    async def connect(self):
        """Connect to MCP server"""
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "mcp_server.py"]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                
                # Initialize
                await session.initialize()
                
                # List available tools
                tools = await session.list_tools()
                print(f"[SUCCESS] Connected to MCP server. Available tools:")
                for tool in tools.tools:
                    print(f"   - {tool.name}: {tool.description}")
                
                return session
    
    async def list_directory(self, path: str = ".") -> str:
        """List directory via MCP"""
        result = await self.session.call_tool("list_directory", {"path": path})
        return result.content[0].text
    
    async def read_file(self, path: str) -> str:
        """Read file via MCP"""
        result = await self.session.call_tool("read_file", {"path": path})
        return result.content[0].text
    
    async def search_files(self, pattern: str, directory: str = ".") -> str:
        """Search files via MCP"""
        result = await self.session.call_tool("search_files", {
            "pattern": pattern,
            "directory": directory
        })
        return result.content[0].text
    
    async def get_file_info(self, path: str) -> str:
        """Get file info via MCP"""
        result = await self.session.call_tool("get_file_info", {"path": path})
        return result.content[0].text


async def test_mcp_client():
    """Test the MCP client"""
    client = MCPClient()
    
    async with stdio_client(
        StdioServerParameters(command="uv", args=["run", "mcp_server.py"])
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("\n[LIST] Testing: list_directory")
            result = await session.call_tool("list_directory", {"path": "."})
            print(result.content[0].text)
            
            print("\n[SEARCH] Testing: search_files")
            result = await session.call_tool("search_files", {"pattern": "*.py"})
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(test_mcp_client())