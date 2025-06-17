# test_suite.py - CONSOLIDATED TEST SUITE
# Replaces: debug.py, quick_verify.py, simple_test.py, test_fixes.py

import os
import sys
import traceback
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class TestSuite:
    """Comprehensive test suite for the stock forecasting system"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.results = []
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and track results"""
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name} PASSED")
                self.tests_passed += 1
                self.results.append((test_name, True, None))
                return True
            else:
                print(f"❌ {test_name} FAILED")
                self.tests_failed += 1
                self.results.append((test_name, False, "Test returned False"))
                return False
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            self.tests_failed += 1
            self.results.append((test_name, False, str(e)))
            return False
    
    def test_imports(self) -> bool:
        """Test all critical imports"""
        try:
            import tools
            import analysis
            import prompts
            import agents
            import conversation_viewer
            print("✅ All critical modules imported successfully")
            return True
        except Exception as e:
            print(f"❌ Import failed: {e}")
            return False
    
    def test_api_keys(self) -> bool:
        """Test API key configuration"""
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_openai import ChatOpenAI
            
            claude_key = os.environ.get("claude_api_key", "")
            gpt_key = os.environ.get("gpt_api_key", "")
            
            if not claude_key or not gpt_key:
                print("❌ Missing API keys in environment")
                return False
            
            # Quick API test
            claude_model = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=claude_key)
            gpt_model = ChatOpenAI(model="gpt-4o-2024-08-06", api_key=gpt_key)
            
            print("✅ API keys configured and models initialized")
            return True
        except Exception as e:
            print(f"❌ API test failed: {e}")
            return False
    
    def test_agents(self) -> bool:
        """Test agent creation"""
        try:
            from agents import project_owner, data_engineer, model_executer, reporter
            from langgraph.graph import StateGraph, START, MessagesState
            
            # Test graph creation
            graph = (
                StateGraph(MessagesState)
                .add_node(project_owner)
                .add_node(data_engineer)
                .add_node(model_executer)
                .add_node(reporter)
                .add_edge(START, "project_owner")
                .compile()
            )
            
            print("✅ All agents created and graph compiled successfully")
            return True
        except Exception as e:
            print(f"❌ Agent test failed: {e}")
            return False
    
    def test_basic_tools(self) -> bool:
        """Test basic stock analysis tools"""
        try:
            from analysis import get_stock_price, get_technical_indicators
            
            # Test stock price tool
            result = get_stock_price("AAPL")
            if "AAPL" in result and "$" in result:
                print("✅ Stock price tool working")
            else:
                print(f"⚠️ Unexpected stock price result: {result[:100]}")
                return False
            
            # Test technical indicators
            result2 = get_technical_indicators("AAPL")
            if "Technical Analysis" in result2:
                print("✅ Technical indicators tool working")
            else:
                print(f"⚠️ Unexpected technical result: {result2[:100]}")
                return False
            
            return True
        except Exception as e:
            print(f"❌ Basic tools test failed: {e}")
            return False
    
    def test_ml_model(self) -> bool:
        """Test the complete ML model workflow"""
        try:
            from analysis import (
                create_technical_features, 
                train_decision_tree_model, 
                predict_stock_price,
                quick_model_test
            )
            
            symbol = "AAPL"
            
            print("  Creating features...")
            feature_result = create_technical_features.invoke({"symbol": symbol, "period": "1y"})
            if "Data ready for model training!" not in feature_result:
                print(f"❌ Feature creation failed: {feature_result}")
                return False
            
            print("  Training model...")
            train_result = train_decision_tree_model.invoke({"symbol": symbol, "max_depth": 6})
            if "RandomForestRegressor" not in train_result:
                print(f"❌ Model training failed: {train_result}")
                return False
            
            print("  Testing predictions...")
            pred_result = predict_stock_price.invoke({"symbol": symbol})
            if "Predicted Price" not in pred_result:
                print(f"❌ Prediction failed: {pred_result}")
                return False
            
            print("  Running diagnostics...")
            test_result = quick_model_test.invoke({"symbol": symbol})
            if "✅ WORKING" in test_result:
                print("✅ Complete ML workflow working correctly")
                return True
            else:
                print(f"⚠️ Model diagnostic issues: {test_result}")
                return False
                
        except Exception as e:
            print(f"❌ ML model test failed: {e}")
            return False
    
    def test_model_persistence(self) -> bool:
        """Test model saving and loading"""
        try:
            from analysis import (
                save_trained_model, 
                load_trained_model, 
                list_saved_models,
                smart_predict_stock_price
            )
            
            symbol = "AAPL"
            
            # Ensure we have a trained model first
            if f'{symbol}_model' not in globals():
                self.test_ml_model()  # This should train a model
            
            print("  Testing model save...")
            save_result = save_trained_model.invoke({"symbol": symbol})
            if "✅ Model saved permanently" not in save_result:
                print(f"❌ Model save failed: {save_result}")
                return False
            
            print("  Testing model listing...")
            list_result = list_saved_models.invoke({})
            if symbol not in list_result:
                print(f"❌ Model not found in list: {list_result}")
                return False
            
            print("  Testing smart prediction...")
            smart_result = smart_predict_stock_price.invoke({"symbol": symbol})
            if "Predicted Price" not in smart_result:
                print(f"❌ Smart prediction failed: {smart_result}")
                return False
            
            print("✅ Model persistence working correctly")
            return True
            
        except Exception as e:
            print(f"❌ Model persistence test failed: {e}")
            return False
    
    def test_single_agent_execution(self) -> bool:
        """Test execution of a single agent"""
        try:
            from agents import project_owner
            from langchain_core.messages import HumanMessage
            from langgraph.graph import StateGraph, START, MessagesState
            
            # Create simple graph with just project_owner
            simple_graph = (
                StateGraph(MessagesState)
                .add_node(project_owner)
                .add_edge(START, "project_owner")
                .compile()
            )
            
            # Test with invoke
            result = simple_graph.invoke({
                "messages": [HumanMessage(content="Hello, analyze AAPL stock")]
            })
            
            if len(result.get('messages', [])) > 1:
                print("✅ Single agent execution working")
                return True
            else:
                print("❌ Agent didn't respond properly")
                return False
                
        except Exception as e:
            print(f"❌ Single agent test failed: {e}")
            return False
    
    def test_visualization(self) -> bool:
        """Test visualization creation"""
        try:
            from analysis import create_model_visualization
            import os
            
            symbol = "AAPL"
            
            # Ensure we have model data
            if f'{symbol}_model' not in globals():
                self.test_ml_model()
            
            # Test visualization creation
            viz_result = create_model_visualization.invoke({
                "symbol": symbol, 
                "chart_type": "performance"
            })
            
            if "visualization saved as" in viz_result:
                print("✅ Visualization creation working")
                return True
            else:
                print(f"❌ Visualization failed: {viz_result}")
                return False
                
        except Exception as e:
            print(f"❌ Visualization test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run complete test suite"""
        print("🚀 COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        tests = [
            ("Import Test", self.test_imports),
            ("API Keys Test", self.test_api_keys),
            ("Agent Creation Test", self.test_agents),
            ("Basic Tools Test", self.test_basic_tools),
            ("ML Model Test", self.test_ml_model),
            ("Model Persistence Test", self.test_model_persistence),
            ("Single Agent Execution", self.test_single_agent_execution),
            ("Visualization Test", self.test_visualization)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        self.print_summary()
        return self.tests_failed == 0
    
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUITE RESULTS")
        print("=" * 60)
        print(f"✅ Tests Passed: {self.tests_passed}")
        print(f"❌ Tests Failed: {self.tests_failed}")
        print(f"📈 Success Rate: {(self.tests_passed / (self.tests_passed + self.tests_failed) * 100):.1f}%")
        
        if self.tests_failed > 0:
            print("\n❌ FAILED TESTS:")
            for test_name, passed, error in self.results:
                if not passed:
                    print(f"  - {test_name}: {error}")
        
        if self.tests_failed == 0:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ System is fully operational and ready for use!")
        else:
            print(f"\n⚠️ {self.tests_failed} tests failed. Please fix before proceeding.")

def main():
    """Main entry point for test suite"""
    import sys
    
    suite = TestSuite()
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "imports":
            suite.run_test("Import Test", suite.test_imports)
        elif test_name == "api":
            suite.run_test("API Keys Test", suite.test_api_keys)
        elif test_name == "agents":
            suite.run_test("Agent Test", suite.test_agents)
        elif test_name == "tools":
            suite.run_test("Basic Tools Test", suite.test_basic_tools)
        elif test_name == "ml":
            suite.run_test("ML Model Test", suite.test_ml_model)
        elif test_name == "persistence":
            suite.run_test("Model Persistence Test", suite.test_model_persistence)
        elif test_name == "viz":
            suite.run_test("Visualization Test", suite.test_visualization)
        elif test_name == "quick":
            # Quick essential tests only
            essential_tests = [
                ("Import Test", suite.test_imports),
                ("Basic Tools Test", suite.test_basic_tools),
                ("ML Model Test", suite.test_ml_model)
            ]
            for test_name, test_func in essential_tests:
                suite.run_test(test_name, test_func)
            suite.print_summary()
        else:
            print("Available test options:")
            print("  imports, api, agents, tools, ml, persistence, viz, quick")
            print("  (no args) - run all tests")
    else:
        # Run all tests
        suite.run_all_tests()

if __name__ == "__main__":
    main()