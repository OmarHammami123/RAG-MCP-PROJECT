import unittest
import os
import shutil
from pathlib import Path
import sys

#add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_tools import MCPTools

class TestMCPTools(unittest.TestCase):
    def setUp(self):
        """set up a temporary test dir"""
        self.test_dir = Path("test_env")
        self.test_dir.mkdir(exist_ok=True)
        self.tools = MCPTools(base_directory=str(self.test_dir))
        
        #create some dummy files 
        (self.test_dir / "hello.txt").write_text("Hello, World!")
        (self.test_dir / "data.csv").write_text("name,age\nAlice,30\nBob,25")
        (self.test_dir / "subfolder").mkdir()
        (self.test_dir / "subfolder" / "nested.py").write_text("print('Hello from nested script!')")
        
    def tearDown(self):
        """clean up after tests"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    
    def test_list_directory(self):
        result = self.tools.list_directory(".")
        self.assertTrue(result.success)
        self.assertIn("hello.txt",result.content)
        self.assertIn("subfolder", result.content)
        
    def test_read_file(self):
        result = self.tools.read_file("hello.txt")
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Hello, World!")
        
    def test_read_nonexistent_file(self):
        result = self.tools.read_file("nonexistent.txt")
        self.assertFalse(result.success)        
                
    def test_search_files(self):
        result = self.tools.search_files("data")
        self.assertTrue(result.success)
        self.assertIn("data.csv", result.content)
    
    def test_write_file(self):
        result = self.tools.write_file("newfile.md", "# This is a test file")
        self.assertTrue(result.success)
        self.assertTrue((self.test_dir / "newfile.md").exists())
        self.assertEqual((self.test_dir / "newfile.md").read_text(), "# This is a test file")
        
if __name__ == "__main__":
    unittest.main()        