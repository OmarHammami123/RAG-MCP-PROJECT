import os 
import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MCPToolResult:
    success: bool
    content: str
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    
    
    
class MCPTools:
    """mcp tools for filesystem and system operations"""
    def __init__(self, base_directory: str = None):
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()
        self.allowed_extensions = {'.txt', '.md', '.pdf', '.docx', '.py','.json','.csv'}
        
    def list_directory(self, path: str=".")-> MCPToolResult:
        """list contents of a directory"""
        try:
            target_path = self.base_directory / path
            if not target_path.exists():
                return MCPToolResult(False, "", f"Path {target_path} does not exist.")
            
            items = []
            for item in target_path.iterdir():
                item_info = {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                    "modified": item.stat().st_mtime
                }
                items.append(item_info)
            
            content = f"directory listing for {path}:\n"
            for item in sorted(items, key=lambda x: (x['type'], x['name'])):
                size_str = f" ({item['size']} bytes)" if item['size'] is not None else ""
                content += f"- {item['name']} [{item['type']}{size_str}]\n"
            
            return MCPToolResult(True, content, metadata={"items": items})
        except Exception as e:
            return MCPToolResult(False, "", f"error listing directory: {str(e)}")  
            
    
    def read_file(self, filepath: str) -> MCPToolResult:
        """read content of a file"""
        try:
            target_path = self.base_directory / filepath
            if not target_path.exists() or not target_path.is_file():
                return MCPToolResult(False, "", f"File {target_path} does not exist.")
            if target_path.suffix.lower() not in self.allowed_extensions:
                return MCPToolResult(False, "", f"File type {target_path.suffix} is not supported.")
            with open(target_path, "r", encoding="utf-8") as file:
                content = file.read()
                
            return MCPToolResult(True, content, metadata={
                "file_name": target_path.name,
                "file_path": str(target_path),
                "size": target_path.stat().st_size,
                "lines": content.count("\n") + 1
            })
        except UnicodeDecodeError:
            return MCPToolResult(False, "", f"Error decoding file {target_path}.")
        except Exception as e:
            return MCPToolResult(False, "", f"error reading file: {str(e)}")  
        
        
    def write_file(self,filepath: str, content:str, append: bool=False)-> MCPToolResult:
        """write content to a file"""
        try:
            target_path= self.base_directory / filepath
            #create directories if not exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if append else 'w'
            with open(target_path, mode, encoding="utf-8") as file:
                file.write(content)
            action= "appended to" if append else "written to "
            return MCPToolResult(True, f"Content successfully {action} {filepath}.", metadata={
                "file_name": target_path.name,
                "file_path": str(target_path),
                "size": target_path.stat().st_size,
                "action": action.strip()
            })
        except Exception as e:
            return MCPToolResult(False, "", f"error writing to file: {str(e)}")
        
        
    def search_files(self,pattern: str, directory: str=".",file_extension: str =None)-> MCPToolResult:
        """search for files by name pattern in a directory"""
        try:
            target_path = self.base_directory / directory
            if not target_path.exists():
                return MCPToolResult(False,"", f"Directory {target_path} does not exist.")
            
            matches = []
            search_pattern = f"*{pattern}*" if pattern else "*"
            
            for file_path in target_path.rglob(search_pattern):
                if file_path.is_file():
                    if file_extension and not file_path.suffix.lower() == file_extension.lower():
                        continue
                    matches.append({
                        "path": str(file_path.relative_to(self.base_directory)),
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                    })
            content = f"found {len(matches)} files matching '{pattern}' in '{directory}':\n"
            for match in matches:
                content += f" {match['path']} ({match['size']}) bytes\n"
                
            return MCPToolResult(True, content,metadata={"matches":matches})
        
        except Exception as e:
            return MCPToolResult(False, "", f"error searching files: {str(e)}")
        
        
    def get_file_info(self,filepath: str) -> MCPToolResult:
        """get detailed info about a file"""
        try:
            target_path = self.base_directory / filepath
            
            if not target_path.exists():
                return MCPToolResult(False, "", f"File {target_path} does not exist.")
            
            stat = target_path.stat()
            info = {
                "file_name": target_path.name,
                "file_path": str(target_path),
                "size": stat.st_size,
                "created": stat.st_birthtime,
                "modified": stat.st_mtime,
                "accessed": stat.st_atime,
                "is_directory": target_path.is_dir(),
                "is_file": target_path.is_file(),
                "extension": target_path.suffix
            }    
            
            content = f"File info for {filepath}:\n"
            content += f" Name: {info['file_name']}\n"
            content += f" size: {info['size']}\n"
            content += f"type: {'Directory' if info['is_directory'] else 'File'}\n"
            content += f"extension: {info['extension']}\n"
            
            return MCPToolResult(True, content, metadata=info)
        except Exception as e:
            return MCPToolResult(False, "", f"error getting file info: {str(e)}")
        
        
    def run_command(self, command: str)-> MCPToolResult:
        """run a safe system command (limited set)"""
        #safe commands only
        safe_commands = ["ls", "dir","echo", "pwd", "cd","time","date", "whoami"] 
        cmd_parts = command.split()
        if not cmd_parts or cmd_parts[0] not in safe_commands:
            return MCPToolResult(False, "", "Command not allowed or invalid.")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )  
            
            if result.returncode ==0:
                return MCPToolResult(True, result.stdout.strip(), metadata={
                    "command": command,
                    "output": result.stdout.strip(),
                    "error": result.stderr.strip() if result.stderr else None
                })
            else:
                return MCPToolResult(False, "", f"Command failed with error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            return MCPToolResult(False, "", "Command timed out.")
        except Exception as e:
            return MCPToolResult(False, "", f"error running command: {str(e)}")
        
        
        
        
        
#test mcp tools
if __name__ == "__main__":
    print("MCP Tools Test")
    
    tools = MCPTools(base_directory=".")
    #test directory listing
    print("\n-- Listing current directory --")
    result = tools.list_directory(".")
    print(result.content if result.success else result.error)
    
    #test file search
    print("\n-- Searching for .pdf files --")
    result = tools.search_files("","documents",file_extension=".pdf")
    print(result.content if result.success else result.error)
    
    #test file info
    print("\n-- Getting file info for config.py --")
    result = tools.get_file_info("config.py")
    print(result.content if result.success else result.error)
    
    #test write read file
    print("\n-- creating and reading test file --")    
    test_content = "This is a test file created by MCP tools.\nTimestamp: " + str(time.time())
    write_result = tools.write_file("test_mcp.txt", test_content)
    print(write_result.content if write_result.success else write_result.error)
    read_result = tools.read_file("test_mcp.txt")
    print(read_result.content if read_result.success else read_result.error)
    
                                 

  