# main.py - OPTIMIZED MAIN ENTRY POINT
# Consolidates: main.py + forecasting_demo.py functionality

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, MessagesState

# Import core modules
from conversation_viewer import ConversationViewer
from agents import project_owner, data_engineer, model_executer, reporter

load_dotenv()

class StockForecastingSystem:
    """Main class for the stock forecasting system"""
    
    def __init__(self):
        self.graph = None
        self.viewer = ConversationViewer()
        self.setup_graph()
    
    def setup_graph(self):
        """Initialize the multi-agent graph"""
        print("🔧 Building enhanced agent graph...")
        self.graph = (
            StateGraph(MessagesState)
            .add_node(project_owner)
            .add_node(data_engineer)
            .add_node(model_executer)
            .add_node(reporter)
            .add_edge(START, "project_owner")
            .compile()
        )
        print("✅ Agent graph created successfully!")
    
    def get_demo_queries(self):
        """Get predefined demo queries"""
        return [
            """Create a comprehensive RandomForest forecasting model for AAPL stock - collect advanced data, 
            engineer ML features, train RandomForest model to predict price CHANGES, save model permanently 
            to disk, backtest performance, and create dynamic visualizations""",
            
            """Analyze TSLA stock with advanced machine learning forecasting - prepare technical features, 
            train 100-tree RandomForest model to predict daily price movements, save trained model for reuse, 
            generate varying predictions, backtest trading strategy, and create performance charts""",
            
            """Build a professional predictive trading system for GOOGL - fetch historical data, engineer 
            advanced ML features, train persistent RandomForest to predict price changes, forecast future 
            movements with confidence intervals, backtest strategy performance, and provide investment 
            recommendations with comprehensive visualizations""",
            
            """Develop a complete algorithmic trading system for MSFT - create comprehensive technical features, 
            train RandomForest model to predict daily price changes, implement model persistence for production, 
            backtest trading strategy with realistic predictions, save all models permanently, and generate 
            executive-level investment reports with dynamic charts""",
            
            """Advanced multi-stock comparative analysis - build RandomForest models for AAPL vs GOOGL vs TSLA, 
            predict price changes for each stock, save all trained models permanently, create comparative 
            visualizations showing varying predictions, backtest strategies, and recommend the best investment 
            opportunity based on ML predictions and risk analysis""",
            
            """Professional portfolio optimization system - build and save RandomForest models for major tech 
            stocks (AAPL, GOOGL, MSFT, TSLA), predict price movements for each, implement model persistence 
            across stocks, create portfolio allocation recommendations, backtest combined strategies, and 
            generate comprehensive executive investment reports with risk analysis and dynamic visualizations"""
        ]
    
    def run_interactive_demo(self):
        """Run the interactive forecasting demonstration"""
        print("🚀 ADVANCED STOCK FORECASTING MULTI-AGENT SYSTEM")
        print("=" * 70)
        print("🎯 POWERED BY: RandomForest ML + Price Change Prediction + Model Persistence")
        print("=" * 70)
        print("This system will perform:")
        print("1. 📊 Advanced Data Collection & Feature Engineering")
        print("2. 🤖 RandomForest Model Training (100 trees)")
        print("3. 📈 Price Change Prediction & Forecasting")
        print("4. 💾 Permanent Model Persistence & Auto-Loading")
        print("5. 🧪 Comprehensive Backtesting & Validation")
        print("6. 📋 Professional Reporting & Dynamic Visualization")
        print("7. 🎯 Multi-Stock Portfolio Analysis")
        print("=" * 70)
        
        demo_queries = self.get_demo_queries()
        
        print("\n🎯 Available Demo Queries:")
        for i, query in enumerate(demo_queries, 1):
            clean_query = ' '.join(query.split())
            if len(clean_query) > 120:
                clean_query = clean_query[:120] + "..."
            print(f"{i}. {clean_query}")
        
        # Get user choice
        try:
            choice = input(f"\nSelect a demo (1-{len(demo_queries)}) or press Enter for default [1]: ").strip()
            if not choice:
                choice = "1"
            query_index = int(choice) - 1
            
            if query_index < 0 or query_index >= len(demo_queries):
                print("Invalid choice, using default...")
                query_index = 0
                
            selected_query = demo_queries[query_index]
            
        except (ValueError, KeyboardInterrupt):
            print("Using default query...")
            selected_query = demo_queries[0]
        
        print(f"\n🎮 Running Demo Query #{query_index + 1}:")
        print("=" * 80)
        print(f"📋 {' '.join(selected_query.split())}")
        print("=" * 80)
        
        # Run the conversation
        self.viewer.run(self.graph, selected_query)
        
        # Save the log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.viewer.save_log(f"forecasting_demo_{timestamp}.txt")
        
        print("\n" + "=" * 80)
        print("🎉 DEMO COMPLETED!")
        print("=" * 80)
        print("The system has demonstrated:")
        print("✅ Multi-agent coordination with RandomForest ML")
        print("✅ Advanced data collection & feature engineering")
        print("✅ RandomForest machine learning model training")
        print("✅ Price change prediction & forecasting")
        print("✅ Permanent model persistence & auto-loading")
        print("✅ Comprehensive backtesting")
        print("✅ Professional visualization & reporting")
    
    def run_custom_analysis(self, query: str):
        """Run custom stock analysis"""
        print(f"🎯 Running Custom Analysis:")
        print(f"Query: {query}")
        print("=" * 60)
        
        self.viewer.run(self.graph, query)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.viewer.save_log(f"custom_analysis_{timestamp}.txt")
    
    def run_simple_test(self):
        """Run a simple system test"""
        print("🧪 SIMPLE SYSTEM TEST")
        print("-" * 40)
        
        try:
            test_query = "Test AAPL forecasting - create features, train RandomForest to predict price changes, save model"
            
            print(f"Testing with: {test_query}")
            
            result = self.graph.invoke({"messages": [HumanMessage(content=test_query)]})
            
            print(f"✅ System test completed successfully!")
            print(f"📊 Total messages generated: {len(result.get('messages', []))}")
            
            # Check agent participation
            agent_responses = {}
            for msg in result.get('messages', []):
                if hasattr(msg, 'name') and msg.name:
                    agent_responses[msg.name] = agent_responses.get(msg.name, 0) + 1
            
            if agent_responses:
                print(f"📈 Agent participation:")
                for agent, count in agent_responses.items():
                    print(f"  - {agent}: {count} messages")
            
            return True
            
        except Exception as e:
            print(f"❌ System test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def show_help(self):
        """Show help information"""
        print("🤖 STOCK FORECASTING SYSTEM - HELP")
        print("=" * 50)
        print("\nAvailable commands:")
        print("  --demo         : Run interactive demo (default)")
        print("  --test         : Run simple system test")
        print("  --help         : Show this help")
        print("  --query 'text' : Run custom analysis")
        print("  --tools        : Show available tools")
        print("  --models       : Show model management info")
        print("\nExamples:")
        print("  python main.py --demo")
        print("  python main.py --query 'analyze AAPL stock with ML forecasting'")
        print("  python main.py --test")
    
    def show_tools(self):
        """Show available tools summary"""
        print("🛠️ AVAILABLE TOOLS SUMMARY")
        print("=" * 50)
        
        tools = {
            "Data Engineer Tools": [
                "get_stock_price - Current stock data",
                "get_stock_news - Recent news & sentiment",
                "compare_stocks - Multi-stock analysis",
                "fetch_historical_data - Historical data for ML",
                "create_technical_features - ML feature engineering"
            ],
            "Model Executor Tools": [
                "get_technical_indicators - Technical analysis",
                "train_decision_tree_model - RandomForest training",
                "predict_stock_price - Price forecasting",
                "backtest_model - Performance validation",
                "save_trained_model - Model persistence",
                "load_trained_model - Model loading",
                "smart_predict_stock_price - Auto-loading predictions"
            ],
            "Reporter Tools": [
                "create_model_visualization - Charts & graphs",
                "model_summary_report - Investment reports",
                "list_saved_models - Model inventory"
            ]
        }
        
        for category, tool_list in tools.items():
            print(f"\n📊 {category}:")
            for tool in tool_list:
                print(f"  • {tool}")
    
    def show_models(self):
        """Show model management information"""
        print("💾 MODEL MANAGEMENT")
        print("=" * 50)
        
        try:
            from func import list_saved_models
            result = list_saved_models.invoke({})
            print(result)
        except Exception as e:
            print(f"❌ Error accessing model info: {e}")
            print("\nModel management features:")
            print("• Permanent model persistence")
            print("• Auto-loading capabilities")
            print("• Version control")
            print("• Production deployment ready")

def main():
    """Main entry point"""
    system = StockForecastingSystem()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "--test":
            if system.run_simple_test():
                print("\n✅ All systems operational!")
            else:
                print("\n❌ System test failed!")
            return
        
        elif command == "--help":
            system.show_help()
            return
        
        elif command == "--tools":
            system.show_tools()
            return
        
        elif command == "--models":
            system.show_models()
            return
        
        elif command == "--query":
            if len(sys.argv) > 2:
                query = ' '.join(sys.argv[2:])
                system.run_custom_analysis(query)
            else:
                print("❌ Please provide a query after --query")
                print("Example: python main.py --query 'analyze AAPL stock'")
            return
        
        elif command == "--demo":
            system.run_interactive_demo()
            return
        
        else:
            print(f"❌ Unknown command: {command}")
            system.show_help()
            return
    
    # Default action - run interactive demo
    try:
        system.run_interactive_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()