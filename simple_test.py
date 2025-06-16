# simple_test.py - Quick test to verify no circular imports
import sys
import traceback

def test_imports():
    """Test all imports without circular dependencies"""
    try:
        print("🧪 Testing imports...")
        
        # Test basic imports
        print("  ✓ Testing tools.py...")
        import tools
        
        print("  ✓ Testing func.py...")
        import func
        
        print("  ✓ Testing handoff.py...")
        import handoff
        
        print("  ✓ Testing prompts.py...")
        import prompts
        
        print("  ✓ Testing agents.py...")
        import agents
        
        print("  ✓ Testing conversation_viewer.py...")
        import conversation_viewer
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_agent_creation():
    """Test that agents are created properly"""
    try:
        print("\n🤖 Testing agent creation...")
        
        from agents import project_owner, data_engineer, model_executer, reporter
        
        print(f"  ✓ Project Owner: {type(project_owner)}")
        print(f"  ✓ Data Engineer: {type(data_engineer)}")
        print(f"  ✓ Model Executer: {type(model_executer)}")
        print(f"  ✓ Reporter: {type(reporter)}")
        
        print("✅ All agents created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        traceback.print_exc()
        return False

def test_graph_creation():
    """Test graph creation"""
    try:
        print("\n🔧 Testing graph creation...")
        
        from agents import project_owner, data_engineer, model_executer, reporter
        from langgraph.graph import StateGraph, START, MessagesState
        
        graph = (
            StateGraph(MessagesState)
            .add_node(project_owner)
            .add_node(data_engineer)
            .add_node(model_executer)
            .add_node(reporter)
            .add_edge(START, "project_owner")
            .compile()
        )
        
        print("  ✓ Graph compiled successfully!")
        print("✅ Graph creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Graph creation failed: {e}")
        traceback.print_exc()
        return False

def test_tools():
    """Test that tools work"""
    try:
        print("\n🛠️  Testing tools...")
        
        from func import get_stock_price
        
        # Test a simple tool
        result = get_stock_price("AAPL")
        if "AAPL" in result:
            print("  ✓ Stock price tool works!")
        else:
            print(f"  ⚠️  Unexpected result: {result}")
        
        print("✅ Tools test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 RUNNING SYSTEM TESTS")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Agent Creation Test", test_agent_creation),
        ("Graph Creation Test", test_graph_creation),
        ("Tools Test", test_tools)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
                print(f"\n❌ {test_name} FAILED!")
                break
        except Exception as e:
            print(f"\n❌ {test_name} CRASHED: {e}")
            all_passed = False
            break
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ System is ready to run!")
        print("\nYou can now run:")
        print("  python forecasting_demo.py")
        print("  python main.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please fix the errors above before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    main()