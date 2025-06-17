# diagnose_analysis.py - Diagnostic script to check analysis package
"""
Script to diagnose what's wrong with the analysis package imports.
"""

import os
import sys
from pathlib import Path

def check_file_structure():
    """Check if all required analysis files exist"""
    print("🔍 Checking Analysis Package File Structure...")
    
    required_files = {
        "analysis/__init__.py": "Package initialization",
        "analysis/shared_state.py": "Centralized state manager",
        "analysis/basic_tools.py": "Basic stock analysis tools", 
        "analysis/ml_models.py": "ML and forecasting tools",
        "analysis/persistence.py": "Model persistence tools",
        "analysis/visualization.py": "Visualization and reporting",
        "analysis/html_generator.py": "HTML report generator (NEW)"
    }
    
    missing = []
    present = []
    
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            size_kb = Path(file_path).stat().st_size / 1024
            present.append(f"✅ {file_path} ({size_kb:.1f} KB) - {description}")
        else:
            missing.append(f"❌ {file_path} - {description}")
    
    for item in present:
        print(item)
    
    if missing:
        print("\n⚠️ Missing files:")
        for item in missing:
            print(item)
        return False
    
    print("✅ All required files present!")
    return True

def test_basic_imports():
    """Test importing individual modules"""
    print("\n🧪 Testing Individual Module Imports...")
    
    modules_to_test = [
        "analysis.shared_state",
        "analysis.basic_tools", 
        "analysis.ml_models",
        "analysis.persistence",
        "analysis.visualization",
        "analysis.html_generator"
    ]
    
    import_results = {}
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module} - Import successful")
            import_results[module] = True
        except ImportError as e:
            print(f"❌ {module} - Import failed: {e}")
            import_results[module] = False
        except Exception as e:
            print(f"⚠️ {module} - Import error: {e}")
            import_results[module] = False
    
    return import_results

def test_analysis_package():
    """Test importing the main analysis package"""
    print("\n📦 Testing Analysis Package Import...")
    
    try:
        import analysis
        print("✅ Analysis package imported successfully")
        
        # Check if tools are available
        tools_to_check = [
            "get_stock_price",
            "create_technical_features", 
            "train_decision_tree_model",
            "collect_analysis_data",
            "collect_multi_stock_data",  # NEW
            "create_comparative_html_report"  # NEW
        ]
        
        missing_tools = []
        present_tools = []
        
        for tool in tools_to_check:
            if hasattr(analysis, tool):
                present_tools.append(tool)
            else:
                missing_tools.append(tool)
        
        print(f"✅ Available tools: {len(present_tools)}")
        for tool in present_tools:
            print(f"   - {tool}")
        
        if missing_tools:
            print(f"❌ Missing tools: {len(missing_tools)}")
            for tool in missing_tools:
                print(f"   - {tool}")
        
        return len(missing_tools) == 0
        
    except ImportError as e:
        print(f"❌ Analysis package import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Analysis package error: {e}")
        return False

def test_tool_execution():
    """Test actually running a simple tool"""
    print("\n🔧 Testing Tool Execution...")
    
    try:
        from analysis import get_stock_price
        
        # Test the tool
        result = get_stock_price.invoke({"symbol": "AAPL"})
        
        if "Analysis package import error" in result:
            print("❌ Tool execution failed - using stub functions")
            print(f"Result: {result}")
            return False
        else:
            print("✅ Tool execution successful")
            print(f"Result preview: {result[:100]}...")
            return True
            
    except Exception as e:
        print(f"❌ Tool execution error: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📚 Checking Dependencies...")
    
    dependencies = [
        "yfinance",
        "pandas", 
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "langchain_core"
    ]
    
    missing_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - not installed")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {missing_deps}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def main():
    """Run complete diagnostic"""
    print("🔬 ANALYSIS PACKAGE DIAGNOSTIC")
    print("=" * 50)
    
    # Step 1: Check file structure
    files_ok = check_file_structure()
    
    # Step 2: Check dependencies
    deps_ok = check_dependencies()
    
    # Step 3: Test individual imports
    import_results = test_basic_imports()
    
    # Step 4: Test package import
    package_ok = test_analysis_package()
    
    # Step 5: Test tool execution
    tools_ok = test_tool_execution()
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY:")
    print("=" * 30)
    print(f"Files present: {'✅' if files_ok else '❌'}")
    print(f"Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"Module imports: {'✅' if all(import_results.values()) else '❌'}")
    print(f"Package import: {'✅' if package_ok else '❌'}")
    print(f"Tool execution: {'✅' if tools_ok else '❌'}")
    
    if all([files_ok, deps_ok, package_ok, tools_ok]):
        print("\n🎉 ALL DIAGNOSTICS PASSED!")
        print("The analysis package should work correctly.")
    else:
        print("\n⚠️ ISSUES FOUND!")
        
        # Specific recommendations
        if not files_ok:
            print("🔧 SOLUTION: Missing files detected")
            print("   - Save the updated analysis files I provided")
            print("   - Make sure analysis/html_generator.py is created")
        
        if not all(import_results.values()):
            failed_modules = [k for k, v in import_results.items() if not v]
            print(f"🔧 SOLUTION: Module import failures: {failed_modules}")
            print("   - Check syntax errors in the failed modules")
            print("   - Make sure all dependencies are installed")
        
        if not tools_ok:
            print("🔧 SOLUTION: Tool execution failed")
            print("   - The analysis package is using stub functions")
            print("   - Fix the import errors above")
            print("   - Make sure analysis/html_generator.py exists and has proper content")

if __name__ == "__main__":
    main()