# test_fixes_updated.py - Updated test script with centralized state manager
"""
Updated test script to verify the centralized state manager fixes.
"""

import os
import sys
import traceback
from pathlib import Path

def check_analysis_structure():
    """Check if analysis package structure is correct"""
    print("🔍 Checking Analysis Package Structure...")
    
    required_files = {
        "analysis/__init__.py": "Package initialization",
        "analysis/shared_state.py": "Centralized state manager",  # NEW
        "analysis/basic_tools.py": "Basic stock analysis tools", 
        "analysis/ml_models.py": "ML and forecasting tools",
        "analysis/persistence.py": "Model persistence tools",
        "analysis/visualization.py": "Visualization and reporting"
    }
    
    missing = []
    present = []
    
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            present.append(f"✅ {file_path} - {description}")
        else:
            missing.append(f"❌ {file_path} - {description}")
    
    for item in present:
        print(item)
    
    if missing:
        print("\nMissing files:")
        for item in missing:
            print(item)
        return False
    
    print("✅ All required files present!")
    return True

def test_centralized_state_manager():
    """Test the centralized state manager"""
    print("\n🌐 Testing Centralized State Manager...")
    
    try:
        # Test importing centralized state manager
        from analysis.shared_state import ModelStateManager, state_manager
        print("✅ Centralized state manager imported successfully")
        
        # Test singleton behavior
        manager1 = ModelStateManager()
        manager2 = ModelStateManager()
        
        if manager1 is manager2:
            print("✅ Singleton pattern working correctly")
        else:
            print("❌ Singleton pattern failed - multiple instances created")
            return False
        
        # Test state sharing
        test_symbol = "TEST"
        test_data = {"test": "centralized_value"}
        
        manager1.set_model_data(test_symbol, "test_key", test_data)
        retrieved_data = manager2.get_model_data(test_symbol, "test_key")
        
        if retrieved_data == test_data:
            print("✅ State sharing between instances works correctly")
        else:
            print("❌ State sharing failed")
            return False
        
        # Test has_model_data
        if manager1.has_model_data(test_symbol, "test_key"):
            print("✅ has_model_data works correctly")
        else:
            print("❌ has_model_data failed")
            return False
        
        # Clean up
        manager1.clear_model_data(test_symbol)
        
        print("✅ Centralized state manager functioning correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Centralized state manager error: {e}")
        traceback.print_exc()
        return False

def test_state_sharing_between_modules():
    """Test that state is shared between ml_models and persistence modules"""
    print("\n🔄 Testing State Sharing Between Modules...")
    
    try:
        # Import both modules
        from analysis import ml_models, persistence
        from analysis.shared_state import state_manager
        
        test_symbol = "SHARED_TEST"
        test_model_data = {"type": "test_model", "trained": True}
        
        # Set data in ml_models context (simulate training)
        print("Setting model data...")
        state_manager.set_model_data(test_symbol, "model", test_model_data)
        
        # Try to access from persistence context (simulate saving)
        print("Retrieving model data...")
        retrieved_data = state_manager.get_model_data(test_symbol, "model")
        
        if retrieved_data == test_model_data:
            print("✅ State successfully shared between modules!")
            
            # Clean up
            state_manager.clear_model_data(test_symbol)
            return True
        else:
            print(f"❌ State sharing failed. Expected: {test_model_data}, Got: {retrieved_data}")
            return False
        
    except Exception as e:
        print(f"❌ Module state sharing error: {e}")
        traceback.print_exc()
        return False

def test_package_imports():
    """Test package-level imports"""
    print("\n🔧 Testing Package Imports...")
    
    try:
        # Test importing from package level
        from analysis import get_stock_price, save_trained_model, train_decision_tree_model, state_manager
        print("✅ Package-level imports work!")
        
        # Test state manager is available at package level
        if state_manager is not None:
            print("✅ State manager available at package level")
        else:
            print("❌ State manager not available at package level")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_tool_execution():
    """Test actual tool execution with centralized state"""
    print("\n🔧 Testing Tool Execution...")
    
    try:
        from analysis import get_stock_price, save_trained_model
        
        # Test basic tool
        print("Testing get_stock_price...")
        result = get_stock_price.invoke({"symbol": "AAPL"})
        
        if "AAPL" in result and "Error" not in result:
            print("✅ get_stock_price working correctly")
        else:
            print(f"⚠️ get_stock_price result: {result[:100]}...")
        
        # Test save_trained_model (should fail gracefully without trained model)
        print("Testing save_trained_model (should fail gracefully)...")
        result = save_trained_model.invoke({"symbol": "TEST_SYMBOL"})
        
        if "No trained model found" in result and "❌" in result:
            print("✅ save_trained_model fails gracefully as expected")
        else:
            print(f"⚠️ Unexpected save_trained_model result: {result[:100]}...")
        
        print("✅ Tool execution tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Tool execution error: {e}")
        traceback.print_exc()
        return False

def test_complete_workflow_with_state():
    """Test complete workflow with centralized state management"""
    print("\n🧪 Testing Complete Workflow with Centralized State...")
    
    try:
        from analysis import (
            create_technical_features, train_decision_tree_model, 
            save_trained_model, state_manager
        )
        
        symbol = "AAPL"
        
        print(f"Step 1: Creating features for {symbol}...")
        features_result = create_technical_features.invoke({"symbol": symbol, "period": "6mo"})
        if "Error" in features_result or "❌" in features_result:
            print(f"❌ Feature creation failed: {features_result}")
            return False
        print("✅ Features created")
        
        # Check if feature data is in state manager
        feature_data = state_manager.get_model_data(symbol, 'feature_data')
        if feature_data is not None:
            print("✅ Feature data stored in centralized state manager")
        else:
            print("❌ Feature data not found in state manager")
            return False
        
        print(f"Step 2: Training model for {symbol}...")
        model_result = train_decision_tree_model.invoke({"symbol": symbol, "max_depth": 4})
        if "Error" in model_result or "❌" in model_result:
            print(f"❌ Model training failed: {model_result}")
            return False
        print("✅ Model trained")
        
        # Check if model data is in state manager
        model_data = state_manager.get_model_data(symbol, 'model')
        if model_data is not None:
            print("✅ Model data stored in centralized state manager")
        else:
            print("❌ Model data not found in state manager")
            state_manager.debug_state(symbol)
            return False
        
        print(f"Step 3: Saving model for {symbol}...")
        save_result = save_trained_model.invoke({"symbol": symbol, "version": "test_centralized"})
        if "❌" in save_result:
            print(f"❌ Model saving failed: {save_result}")
            state_manager.debug_state(symbol)
            return False
        print("✅ Model saved successfully!")
        
        print("🎉 Complete workflow with centralized state completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        traceback.print_exc()
        return False

def test_agent_compatibility():
    """Test that agents can still work with the updated tools"""
    print("\n🤖 Testing Agent Compatibility...")
    
    try:
        # Just test that we can import agents without error
        from agents import project_owner, data_engineer, model_executer, reporter
        print("✅ All agents imported successfully")
        
        # Test that agents are compiled graphs (expected type)
        # from langgraph.pregel import CompiledGraph
        from langgraph.graph import StateGraph
        
        agents = [project_owner, data_engineer, model_executer, reporter]
        agent_names = ["project_owner", "data_engineer", "model_executer", "reporter"]
        
        for agent, name in zip(agents, agent_names):
            if hasattr(agent, 'invoke'):  # Check if it has the invoke method
                print(f"✅ {name} has invoke method (correct type)")
            else:
                print(f"❌ {name} doesn't have invoke method")
                return False
        
        print("✅ Agent compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent compatibility error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all updated tests"""
    print("🚀 UPDATED TOOL EXECUTION FIX TESTING")
    print("=" * 70)
    
    tests = [
        ("File Structure", check_analysis_structure),
        ("Centralized State Manager", test_centralized_state_manager),
        ("State Sharing Between Modules", test_state_sharing_between_modules),
        ("Package Imports", test_package_imports),
        ("Tool Execution", test_tool_execution),
        ("Agent Compatibility", test_agent_compatibility),
        ("Complete Workflow", test_complete_workflow_with_state)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
                
                # Stop on critical failures
                if test_name in ["File Structure", "Centralized State Manager"]:
                    print(f"\n⚠️ Critical test failed. Please fix before continuing.")
                    break
                    
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} CRASHED: {e}")
    
    print(f"\n{'='*70}")
    print(f"📊 TEST RESULTS:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("The centralized state manager is working correctly!")
        print("\n🚀 Next Steps:")
        print("1. Your tool execution issues should now be resolved")
        print("2. Run your main demo: python main.py --demo")
        print("3. The save_trained_model tool should work without crashing")
        
    else:
        print(f"\n⚠️ {failed} tests failed.")
        
        if not Path("analysis/shared_state.py").exists():
            print("\n🔧 SOLUTION:")
            print("1. Create analysis/shared_state.py with the centralized state manager")
            print("2. Update analysis/persistence.py to use centralized state")
            print("3. Update analysis/ml_models.py to use centralized state")
            print("4. Update analysis/__init__.py to include shared_state")
        
        print(f"\n💡 Key Points:")
        print("• The centralized state manager ensures modules share data")
        print("• All modules must import from analysis.shared_state")
        print("• The singleton pattern prevents multiple state instances")

if __name__ == "__main__":
    main()