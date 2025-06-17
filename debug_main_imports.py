# debug_main_imports.py - Debug exactly what main.py is seeing
"""
Debug script to find out why main.py still gets stub functions
"""

import sys
import os
from pathlib import Path

def debug_python_path():
    """Check Python path and import locations"""
    print("🔍 DEBUGGING PYTHON IMPORT PATH")
    print("=" * 50)
    
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📁 Python executable: {sys.executable}")
    
    print(f"\n📚 Python path ({len(sys.path)} entries):")
    for i, path in enumerate(sys.path):
        print(f"   {i}: {path}")

def debug_analysis_location():
    """Find where Python thinks analysis package is"""
    print("\n🔍 ANALYSIS PACKAGE LOCATION")
    print("=" * 40)
    
    # Clear any cached analysis imports
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('analysis')]
    print(f"🧹 Clearing {len(modules_to_remove)} cached analysis modules...")
    for module in modules_to_remove:
        del sys.modules[module]
    
    try:
        import analysis
        print(f"✅ Analysis imported from: {analysis.__file__}")
        print(f"📦 Analysis package path: {Path(analysis.__file__).parent}")
        
        # Check if it has our tools
        tools = ['get_stock_price', 'collect_multi_stock_data', 'create_comparative_html_report']
        for tool in tools:
            if hasattr(analysis, tool):
                tool_obj = getattr(analysis, tool)
                print(f"✅ {tool}: {type(tool_obj)}")
                
                # Test if it's a stub
                if hasattr(tool_obj, 'func'):
                    # It's a LangChain tool, test it
                    try:
                        if tool == 'get_stock_price':
                            result = tool_obj.invoke({"symbol": "AAPL"})
                            if "import error" in result.lower():
                                print(f"   ❌ {tool} is STUB FUNCTION: {result[:50]}...")
                            else:
                                print(f"   ✅ {tool} is REAL FUNCTION: {result[:50]}...")
                    except Exception as e:
                        print(f"   ⚠️ {tool} test error: {e}")
            else:
                print(f"❌ {tool}: MISSING")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Analysis import failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_agents_import():
    """Test exactly what agents.py imports"""
    print("\n🤖 DEBUGGING AGENTS.PY IMPORTS")
    print("=" * 40)
    
    # Clear agents cache
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('agents')]
    for module in modules_to_remove:
        del sys.modules[module]
    
    try:
        # Simulate exactly what agents.py does
        print("🔄 Simulating agents.py import...")
        
        # This is the exact import from agents.py
        from analysis import (
            get_stock_price, get_stock_news, compare_stocks, get_technical_indicators,
            fetch_historical_data, create_technical_features, train_decision_tree_model,
            predict_stock_price, backtest_model, quick_model_test,
            save_trained_model, load_trained_model, list_saved_models,
            smart_predict_stock_price, delete_saved_model, model_performance_summary,
            create_model_visualization, model_summary_report,
            collect_analysis_data, gather_visualization_files, create_html_report,
            collect_multi_stock_data, gather_multi_stock_visualizations, create_comparative_html_report
        )
        
        print("✅ All agents.py imports successful")
        
        # Test the key function
        print("🧪 Testing get_stock_price from agents import...")
        result = get_stock_price("AAPL")
        
        if "import error" in result.lower():
            print("❌ AGENTS IMPORT IS GETTING STUB FUNCTIONS!")
            print(f"Stub result: {result}")
            return False
        else:
            print("✅ AGENTS IMPORT IS GETTING REAL FUNCTIONS!")
            print(f"Real result preview: {result[:100]}...")
            return True
            
    except ImportError as e:
        print(f"❌ Agents import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_multiple_analysis_packages():
    """Check if there are multiple analysis packages"""
    print("\n🔍 CHECKING FOR MULTIPLE ANALYSIS PACKAGES")
    print("=" * 50)
    
    # Search for analysis directories
    analysis_dirs = []
    
    # Check current directory
    if Path("analysis").exists():
        analysis_dirs.append(Path("analysis").absolute())
    
    # Check Python path
    for path_str in sys.path:
        path = Path(path_str)
        if path.exists():
            analysis_path = path / "analysis"
            if analysis_path.exists() and analysis_path not in analysis_dirs:
                analysis_dirs.append(analysis_path)
    
    print(f"📦 Found {len(analysis_dirs)} analysis packages:")
    for i, dir_path in enumerate(analysis_dirs):
        print(f"   {i+1}. {dir_path}")
        
        # Check what's in each one
        init_file = dir_path / "__init__.py"
        if init_file.exists():
            try:
                with open(init_file, 'r') as f:
                    content = f.read()
                
                has_stubs = "stub function" in content.lower()
                has_real = "from .html_generator import" in content
                
                print(f"      - __init__.py: {'❌ HAS STUBS' if has_stubs else '✅ NO STUBS'}")
                print(f"      - Real imports: {'✅ YES' if has_real else '❌ NO'}")
                
            except Exception as e:
                print(f"      - Error reading __init__.py: {e}")
        else:
            print(f"      - ❌ No __init__.py")

def test_direct_tool_execution():
    """Test direct tool execution bypassing imports"""
    print("\n🔧 TESTING DIRECT TOOL EXECUTION")
    print("=" * 40)
    
    try:
        # Try to import and run directly from module
        sys.path.insert(0, str(Path.cwd()))
        
        from analysis.basic_tools import get_stock_price as direct_get_stock_price
        
        print("✅ Direct import from analysis.basic_tools successful")
        
        # Test direct execution
        result = direct_get_stock_price("AAPL")
        
        if "import error" in result.lower():
            print("❌ Even direct import gives stub!")
            print(f"Result: {result}")
        else:
            print("✅ Direct import works!")
            print(f"Result preview: {result[:100]}...")
        
    except Exception as e:
        print(f"❌ Direct import failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run complete debugging"""
    print("🐛 DEBUGGING MAIN.PY IMPORT ISSUE")
    print("=" * 60)
    
    # Step 1: Check Python environment
    debug_python_path()
    
    # Step 2: Find analysis package location
    analysis_pkg = debug_analysis_location()
    
    # Step 3: Test agents.py style import
    agents_ok = debug_agents_import()
    
    # Step 4: Check for multiple packages
    check_multiple_analysis_packages()
    
    # Step 5: Test direct import
    test_direct_tool_execution()
    
    # Summary and recommendations
    print("\n📊 DEBUGGING SUMMARY")
    print("=" * 30)
    
    if agents_ok:
        print("✅ Agents import is working - the issue might be elsewhere")
        print("\n🔧 NEXT STEPS:")
        print("1. Try running main.py again")
        print("2. Check if main.py is using a different Python environment")
        print("3. Try: python -c \"from agents import project_owner; print('Success!')\"")
    else:
        print("❌ Agents import is still getting stubs")
        print("\n🔧 SOLUTIONS TO TRY:")
        print("1. Restart your terminal/command prompt")
        print("2. Check if you're in the right directory")
        print("3. Try: python -m pip list | grep -i scikit")
        print("4. Check if there are multiple analysis folders")

if __name__ == "__main__":
    main()