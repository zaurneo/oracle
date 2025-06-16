# test_simple_import.py - Simple test to verify basic imports work
"""
Simple test script to check if the shared_state.py file is working correctly.
Run this first before running the full test suite.
"""

import os
import sys
from pathlib import Path

def test_file_exists():
    """Test if shared_state.py file exists and has content"""
    print("📁 Checking if shared_state.py exists...")
    
    shared_state_path = Path("analysis/shared_state.py")
    
    if not shared_state_path.exists():
        print("❌ analysis/shared_state.py does not exist!")
        print("Please create this file with the corrected content.")
        return False
    
    # Check file size
    file_size = shared_state_path.stat().st_size
    if file_size < 100:
        print(f"❌ analysis/shared_state.py is too small ({file_size} bytes)")
        print("The file might be empty or incomplete.")
        return False
    
    print(f"✅ analysis/shared_state.py exists ({file_size} bytes)")
    return True

def test_file_syntax():
    """Test if shared_state.py has valid Python syntax"""
    print("\n🔍 Checking Python syntax...")
    
    try:
        with open("analysis/shared_state.py", "r") as f:
            content = f.read()
        
        # Try to compile the code
        compile(content, "analysis/shared_state.py", "exec")
        print("✅ Python syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in shared_state.py: {e}")
        print(f"Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error reading shared_state.py: {e}")
        return False

def test_direct_import():
    """Test direct import of shared_state module"""
    print("\n🔧 Testing direct import...")
    
    try:
        # Add current directory to path to ensure we can import
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Try to import the module directly
        import analysis.shared_state as shared_state_module
        print("✅ Direct import successful")
        
        # Check if the required classes/objects exist
        if hasattr(shared_state_module, 'ModelStateManager'):
            print("✅ ModelStateManager class found")
        else:
            print("❌ ModelStateManager class not found")
            return False
        
        if hasattr(shared_state_module, 'state_manager'):
            print("✅ state_manager instance found")
        else:
            print("❌ state_manager instance not found")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_manager_basic():
    """Test basic state manager functionality"""
    print("\n🌐 Testing state manager basic functionality...")
    
    try:
        from analysis.shared_state import ModelStateManager, state_manager
        
        # Test state manager creation
        if state_manager is not None:
            print("✅ state_manager instance created")
        else:
            print("❌ state_manager is None")
            return False
        
        # Test basic operations
        test_symbol = "TEST"
        test_value = "test_data"
        
        # Test set
        state_manager.set_model_data(test_symbol, "test_key", test_value)
        print("✅ set_model_data works")
        
        # Test get
        retrieved = state_manager.get_model_data(test_symbol, "test_key")
        if retrieved == test_value:
            print("✅ get_model_data works")
        else:
            print(f"❌ get_model_data failed. Expected: {test_value}, Got: {retrieved}")
            return False
        
        # Test has
        if state_manager.has_model_data(test_symbol, "test_key"):
            print("✅ has_model_data works")
        else:
            print("❌ has_model_data failed")
            return False
        
        # Test clear
        state_manager.clear_model_data(test_symbol)
        if not state_manager.has_model_data(test_symbol, "test_key"):
            print("✅ clear_model_data works")
        else:
            print("❌ clear_model_data failed")
            return False
        
        print("✅ State manager basic functionality working!")
        return True
        
    except Exception as e:
        print(f"❌ State manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_package_import():
    """Test importing from analysis package"""
    print("\n📦 Testing analysis package import...")
    
    try:
        # Test package-level import
        import analysis
        print("✅ analysis package imported")
        
        # Test specific imports
        from analysis import state_manager, ModelStateManager
        print("✅ state_manager and ModelStateManager imported from package")
        
        if state_manager is not None:
            print("✅ state_manager is available at package level")
        else:
            print("❌ state_manager is None at package level")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Package import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple import tests"""
    print("🧪 SIMPLE IMPORT TEST")
    print("=" * 50)
    
    tests = [
        ("File Exists", test_file_exists),
        ("File Syntax", test_file_syntax),
        ("Direct Import", test_direct_import),
        ("State Manager Basic", test_state_manager_basic),
        ("Package Import", test_analysis_package_import)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*10} {test_name} {'='*10}")
        
        if not test_func():
            print(f"\n❌ {test_name} FAILED - stopping here")
            print("\n🔧 SOLUTION:")
            
            if test_name == "File Exists":
                print("1. Create analysis/shared_state.py file")
                print("2. Copy the corrected content from the artifact")
            elif test_name == "File Syntax":
                print("1. Check the shared_state.py file for syntax errors")
                print("2. Make sure indentation is correct")
                print("3. Check for missing quotes or brackets")
            elif test_name == "Direct Import":
                print("1. Verify the file has correct Python syntax")
                print("2. Check that __all__ is properly defined")
                print("3. Make sure ModelStateManager and state_manager are defined")
            
            return False
        
        print(f"✅ {test_name} PASSED")
    
    print(f"\n🎉 ALL SIMPLE TESTS PASSED!")
    print("You can now run the full test suite:")
    print("python test_fixes_updated.py")

if __name__ == "__main__":
    main()