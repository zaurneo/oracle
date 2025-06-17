# test_main_context.py - Test exactly what main.py sees
"""
Test the exact context that main.py runs in to find the real issue
"""

import sys
import os
from pathlib import Path

def test_main_py_context():
    """Test the exact environment main.py runs in"""
    print("🔧 TESTING MAIN.PY EXECUTION CONTEXT")
    print("=" * 50)
    
    try:
        # Test the exact imports main.py does
        print("🔄 Testing main.py imports...")
        
        # Import agents (like main.py does)
        from agents import (
            project_owner, 
            data_engineer, 
            model_executer, 
            reporter,
            html_generator
        )
        
        print("✅ All agents imported successfully")
        
        # Test a simple tool call like main.py would do
        print("🔧 Testing tool execution in main.py context...")
        
        # Get a tool from data_engineer agent
        data_eng_tools = data_engineer.tools
        print(f"📋 Data engineer has {len(data_eng_tools)} tools")
        
        # Find get_stock_price tool
        stock_price_tool = None
        for tool in data_eng_tools:
            if hasattr(tool, 'name') and 'get_stock_price' in tool.name:
                stock_price_tool = tool
                break
        
        if stock_price_tool:
            print(f"✅ Found get_stock_price tool: {stock_price_tool}")
            
            # Test the tool exactly like the agents would use it
            try:
                result = stock_price_tool.invoke({"symbol": "AAPL"})
                
                if "import error" in result.lower() or "analysis package import error" in result.lower():
                    print("❌ FOUND THE ISSUE! Tool in agent context returns import error:")
                    print(f"Error result: {result}")
                    return False
                else:
                    print("✅ Tool in agent context works correctly!")
                    print(f"Result preview: {result[:100]}...")
                    return True
                    
            except Exception as e:
                print(f"❌ Tool execution error: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("❌ Could not find get_stock_price tool in data_engineer")
            print("Available tools:")
            for tool in data_eng_tools:
                if hasattr(tool, 'name'):
                    print(f"   - {tool.name}")
            return False
        
    except Exception as e:
        print(f"❌ Main.py context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversation_viewer():
    """Test if conversation_viewer.py has the issue"""
    print("\n🗣️ TESTING CONVERSATION_VIEWER CONTEXT")
    print("=" * 45)
    
    try:
        # Check if conversation_viewer imports correctly
        from conversation_viewer import GraphRunner
        print("✅ GraphRunner imported successfully")
        
        # Test creating a graph runner
        runner = GraphRunner()
        print("✅ GraphRunner instance created")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversation viewer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encoding_issue():
    """Test if there's an encoding issue with the analysis package"""
    print("\n📝 TESTING ENCODING ISSUE")
    print("=" * 30)
    
    try:
        # Try to read analysis/__init__.py with different encodings
        init_file = Path("analysis/__init__.py")
        
        # Try UTF-8 first
        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print("✅ UTF-8 encoding works")
        except UnicodeDecodeError as e:
            print(f"❌ UTF-8 encoding failed: {e}")
            
            # Try with errors='ignore'
            try:
                with open(init_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                print("✅ UTF-8 with errors='ignore' works")
                
                # Check for problematic characters
                if len(content) != len(content.encode('utf-8', errors='ignore').decode('utf-8')):
                    print("⚠️ File contains non-UTF-8 characters that could cause issues")
                    
                    # Fix the file
                    print("🔧 Fixing encoding...")
                    with open(init_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print("✅ Fixed encoding issues")
                
            except Exception as e2:
                print(f"❌ All encoding attempts failed: {e2}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Encoding test failed: {e}")
        return False

def run_actual_main_command():
    """Try to run the actual main.py command"""
    print("\n🚀 TESTING ACTUAL MAIN.PY COMMAND")
    print("=" * 40)
    
    try:
        # Import main.py modules
        import main
        print("✅ main.py imported successfully")
        
        # Try to create the query that's failing
        query = "analyze AAPL with clean HTML report"
        print(f"🔧 Testing query: {query}")
        
        # This might show us where the real error occurs
        # We won't actually run it, just test the import path
        
        return True
        
    except Exception as e:
        print(f"❌ Main.py import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all context tests"""
    print("🧪 TESTING MAIN.PY EXECUTION CONTEXT")
    print("=" * 60)
    
    # Test 1: Main.py agent context
    agents_ok = test_main_py_context()
    
    # Test 2: Conversation viewer context  
    conv_ok = test_conversation_viewer()
    
    # Test 3: Encoding issues
    encoding_ok = test_encoding_issue()
    
    # Test 4: Main.py import
    main_ok = run_actual_main_command()
    
    # Summary
    print(f"\n📊 CONTEXT TEST SUMMARY")
    print("=" * 30)
    print(f"Agents context: {'✅' if agents_ok else '❌'}")
    print(f"Conversation viewer: {'✅' if conv_ok else '❌'}")
    print(f"Encoding: {'✅' if encoding_ok else '❌'}")
    print(f"Main.py import: {'✅' if main_ok else '❌'}")
    
    if agents_ok and conv_ok and encoding_ok and main_ok:
        print("\n🎉 ALL TESTS PASSED! The issue might be elsewhere.")
        print("\n🚀 Try running main.py now:")
        print('python main.py --query "analyze AAPL with clean HTML report"')
    else:
        print("\n⚠️ FOUND SPECIFIC ISSUES!")
        if not agents_ok:
            print("🔧 Issue is in agent tool execution context")
        if not encoding_ok:
            print("🔧 Issue is encoding-related")
        if not main_ok:
            print("🔧 Issue is in main.py import")

if __name__ == "__main__":
    main()