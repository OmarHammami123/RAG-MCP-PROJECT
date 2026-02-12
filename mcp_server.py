import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import os
import shutil
from exceptions import MCPSecurityError, MCPToolError, handle_exception
from logging_config import rag_logger


class FilesystemMCPServer:
    """ MCP server for filesystem operations"""
    
    def __init__(self, base_directory: str= "."):
        self.server = Server("filesystem-server")
        self.base_directory = Path(base_directory).resolve()
        self.logger = rag_logger
        self.setup_handlers()
        self.logger.info(f"MCP Server initialized with base directory: {self.base_directory}")
    
    
    def is_safe_path(self, user_path: str)-> bool:
        """
        validate that user_path doesnt escape base_directory
        
        security: prevents path traversal attacks like ../../etc/passwd
        """    
        try:
            #resolve the fulll path
            requested_path = (self.base_directory / user_path).resolve()
            return requested_path.is_relative_to(self.base_directory)
        except(ValueError, OSError):
            return False
    
    
    def validate_path_or_raise(self, user_path: str)->Path:
        """
        validate path and return resolved path object , or raise error
        raises:
        MCPSecurityError if path is unsafe or outside allowed directory
        """
        if not self.is_safe_path(user_path):
            raise MCPSecurityError(
                user_path,
                reason="path_traversal",
                base_dir=str(self.base_directory)
            )
        return (self.base_directory / user_path).resolve()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def setup_handlers(self):
        """Register MCP tools"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="list_directory",
                    description="List files and directories in a given path. Returns names, types, and sizes.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to list (default: current directory)"
                            }
                        }
                    }
                ),
                Tool(
                    name="read_file",
                    description="Read the contents of a text file. Returns the full file content.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="search_files",
                    description="Search for files matching a pattern (supports wildcards like *.py)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Search pattern (e.g., '*.py', 'test*', 'README*')"
                            },
                            "directory": {
                                "type": "string",
                                "description": "Directory to search in (default: current directory)"
                            }
                        },
                        "required": ["pattern"]
                    }
                ),
                Tool(
                    name="get_file_info",
                    description="Get detailed information about a file (size, modified time, permissions)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="create_directory",
                    description="Create a new directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the directory to create"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="write_file",
                    description="Create or overwrite a file with content",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["path", "content"]
                    }
                ),
                Tool(
                    name="move_file",
                    description="Move or rename a file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Source file path"
                            },
                            "destination": {
                                "type": "string",
                                "description": "Destination file path"
                            }
                        },
                        "required": ["source", "destination"]
                    }
                ),
                Tool(
                    name="delete_file",
                    description="Delete a file or directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file or directory to delete"
                            }
                        },
                        "required": ["path"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            """Execute tool calls"""
            
            if name == "list_directory":
                return await self.list_directory(arguments.get("path", "."))
            
            elif name == "read_file":
                return await self.read_file(arguments["path"])
            
            elif name == "search_files":
                return await self.search_files(
                    arguments["pattern"],
                    arguments.get("directory", ".")
                )
            
            elif name == "get_file_info":
                return await self.get_file_info(arguments["path"])
            
            elif name == "create_directory":
                return await self.create_directory(arguments["path"])
            
            elif name == "write_file":
                return await self.write_file(arguments["path"], arguments["content"])
            
            elif name == "move_file":
                return await self.move_file(arguments["source"], arguments["destination"])
            
            elif name == "delete_file":
                return await self.delete_file(arguments["path"])
            
            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]
    
    async def list_directory(self, path: str) -> List[TextContent]:
        """List directory contents"""
        try:
            target_path = self.validate_path_or_raise(path)
            
            if not target_path.exists():
                return [TextContent(
                    type="text",
                    text=f"Error: Directory '{path}' does not exist"
                )]
            
            if not target_path.is_dir():
                return [TextContent(
                    type="text",
                    text=f"Error: '{path}' is not a directory"
                )]
            
            entries = []
            for item in sorted(target_path.iterdir()):
                item_type = "directory" if item.is_dir() else "file"
                size = item.stat().st_size if item.is_file() else 0
                
                entries.append({
                    "name": item.name,
                    "type": item_type,
                    "size": size
                })
            
            result = f"Directory listing for {path}:\n"
            for entry in entries:
                size_str = f" ({entry['size']} bytes)" if entry['type'] == 'file' else ""
                result += f"- {entry['name']} [{entry['type']}]{size_str}\n"
            
            return [TextContent(type="text", text=result)]
        
        except MCPSecurityError as e:
            self.logger.error(f"Security violation in list_directory: {e}")
            return [TextContent(type="text", text=f"[SECURITY ERROR] {e.message}")]
        except Exception as e:
            handle_exception(e, logger=self.logger)
            return [TextContent(type="text", text=f"Error listing directory: {str(e)}")]
    
    async def read_file(self, path: str) -> List[TextContent]:
        """Read file contents"""
        try:
            file_path = self.validate_path_or_raise(path)
            
            if not file_path.exists():
                return [TextContent(type="text", text=f"Error: File '{path}' does not exist")]
            
            if not file_path.is_file():
                return [TextContent(type="text", text=f"Error: '{path}' is not a file")]
            
            # Read file
            content = file_path.read_text(encoding='utf-8')
            
            result = f"Content of {path}:\n{'='*50}\n{content}\n{'='*50}"
            return [TextContent(type="text", text=result)]
        
        except MCPSecurityError as e:
            self.logger.error(f"Security violation in read_file: {e}")
            return [TextContent(type="text", text=f"[SECURITY ERROR] {e.message}")]
        except UnicodeDecodeError:
            return [TextContent(type="text", text=f"Error: '{path}' is not a text file (binary file?)")]
        except Exception as e:
            handle_exception(e, logger=self.logger)
            return [TextContent(type="text", text=f"Error reading file: {str(e)}")]
    
    async def search_files(self, pattern: str, directory: str) -> List[TextContent]:
        """Search for files matching pattern"""
        try:
            search_path = self.validate_path_or_raise(directory)
            
            if not search_path.exists():
                return [TextContent(type="text", text=f"Error: Directory '{directory}' does not exist")]
            
            # Search for files
            matches = list(search_path.glob(pattern))
            
            if not matches:
                return [TextContent(type="text", text=f"No files found matching '{pattern}' in '{directory}'")]
            
            result = f"Found {len(matches)} file(s) matching '{pattern}':\n"
            for match in sorted(matches):
                rel_path = match.relative_to(search_path)
                result += f"- {rel_path}\n"
            
            return [TextContent(type="text", text=result)]
            
        except ValueError as ve:
            return [TextContent(type="text", text=f"[security error] {str(ve)}")]    
        except Exception as e:
            return [TextContent(type="text", text=f"Error searching files: {str(e)}")]
    
    async def get_file_info(self, path: str) -> List[TextContent]:
        """Get file information"""
        try:
            file_path = self.validate_path_or_raise(path)
            
            if not file_path.exists():
                return [TextContent(type="text", text=f"Error: '{path}' does not exist")]
            
            stat = file_path.stat()
            
            info = {
                "name": file_path.name,
                "path": str(file_path),
                "type": "directory" if file_path.is_dir() else "file",
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "readable": os.access(file_path, os.R_OK),
                "writable": os.access(file_path, os.W_OK)
            }
            
            result = f"File information for '{path}':\n"
            result += f"  Name: {info['name']}\n"
            result += f"  Type: {info['type']}\n"
            result += f"  Size: {info['size']} bytes\n"
            result += f"  Readable: {info['readable']}\n"
            result += f"  Writable: {info['writable']}\n"
            
            return [TextContent(type="text", text=result)]
            
        except ValueError as ve:
            return [TextContent(type="text", text=f"[security error] {str(ve)}")]    
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting file info: {str(e)}")]
    
    async def create_directory(self, path: str) -> List[TextContent]:
        """Create a new directory"""
        try:
            dir_path = self.validate_path_or_raise(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return [TextContent(type="text", text=f"[SUCCESS] Created directory: {path}")]
        except ValueError as ve:
            return [TextContent(type="text", text=f"[security error] {str(ve)}")]    
        except Exception as e:
            return [TextContent(type="text", text=f"Error creating directory: {str(e)}")]
    
    async def write_file(self, path: str, content: str) -> List[TextContent]:
        """Write content to a file"""
        try:
            file_path = self.validate_path_or_raise(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            return [TextContent(type="text", text=f"[SUCCESS] Wrote {len(content)} bytes to: {path}")]
        except ValueError as ve:
            return [TextContent(type="text", text=f"[security error] {str(ve)}")]    
        except Exception as e:
            return [TextContent(type="text", text=f"Error writing file: {str(e)}")]
    
    async def move_file(self, source: str, destination: str) -> List[TextContent]:
        """Move or rename a file"""
        try:
            source_path = self.validate_path_or_raise(source)
            dest_path = self.validate_path_or_raise(destination)
            
            if not source_path.exists():
                return [TextContent(type="text", text=f"Error: Source '{source}' does not exist")]
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.rename(dest_path)
            return [TextContent(type="text", text=f"[SUCCESS] Moved '{source}' to '{destination}'")]
        except ValueError as ve:
            return [TextContent(type="text", text=f"[security error] {str(ve)}")]    
        except Exception as e:
            return [TextContent(type="text", text=f"Error moving file: {str(e)}")]
    
    async def delete_file(self, path: str) -> List[TextContent]:
        """Delete a file or directory"""
        try:
            target_path = self.validate_path_or_raise(path)
            
            if not target_path.exists():
                return [TextContent(type="text", text=f"Error: '{path}' does not exist")]
            
            if target_path.is_dir():
                import shutil
                shutil.rmtree(target_path)
                return [TextContent(type="text", text=f"[SUCCESS] Deleted directory: {path}")]
            else:
                target_path.unlink()
                return [TextContent(type="text", text=f"[SUCCESS] Deleted file: {path}")]
        except ValueError as ve:
            return [TextContent(type="text", text=f"[security error] {str(ve)}")]    
        except Exception as e:
            return [TextContent(type="text", text=f"Error deleting: {str(e)}")]
    
    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    server = FilesystemMCPServer()
    await server.run()


if __name__ == "__main__":
    # Don't print anything - MCP uses stdio for communication
    asyncio.run(main())