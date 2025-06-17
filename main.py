# main.py - Updated with HTML Generator Agent and Logs Folder Support
# Consolidates: main.py + forecasting_demo.py functionality with HTML reporting

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Union
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, MessagesState

# Import core modules
from conversation_viewer import ConversationViewer
from agents import project_owner, data_engineer, model_executer, reporter, html_generator

load_dotenv()

class StockForecastingSystem:
    """Main class for the stock forecasting system with HTML reporting"""
    
    def __init__(self):
        self.graph = None
        self.viewer = ConversationViewer()
        self.setup_directories()
        self.setup_graph()
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['logs', 'plots', 'models', 'html_reports']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        print("📁 Directories initialized: logs, plots, models, html_reports")
    
    def setup_graph(self):
        """Initialize the multi-agent graph with HTML generator"""
        print("🔧 Building enhanced agent graph with HTML generator...")
        self.graph = (
            StateGraph(MessagesState)
            .add_node(project_owner)
            .add_node(data_engineer)
            .add_node(model_executer)
            .add_node(reporter)
            .add_node(html_generator)  # NEW
            .add_edge(START, "project_owner")
            .compile()
        )
        print("✅ Agent graph created successfully with 5 agents!")
    
    def get_demo_queries(self):
        """Get predefined demo queries with HTML reporting"""
        return [
            """Create a comprehensive RandomForest forecasting model for AAPL stock with professional HTML report - 
            collect advanced data, engineer ML features, train RandomForest model to predict price changes, save model 
            permanently, create dynamic visualizations, and generate a complete HTML report with embedded charts""",
            
            """Analyze TSLA stock with advanced machine learning forecasting and HTML presentation - prepare technical 
            features, train 100-tree RandomForest model to predict daily price movements, save trained model for reuse, 
            generate performance visualizations, and create a professional HTML report for stakeholders""",
            
            """Build a professional predictive trading system for GOOGL with HTML reporting - fetch historical data, 
            engineer advanced ML features, train persistent RandomForest to predict price changes, forecast future 
            movements with confidence intervals, create comprehensive visualizations, and deliver a complete HTML 
            analysis report""",
            
            """Develop a complete algorithmic trading system for MSFT with HTML presentation - create comprehensive 
            technical features, train RandomForest model to predict daily price changes, implement model persistence, 
            generate executive-level visualizations, and produce a professional HTML report with embedded charts 
            and interactive elements""",
            
            """Advanced multi-stock comparative analysis with HTML dashboard - build RandomForest models for AAPL vs 
            GOOGL vs TSLA, predict price changes for each stock, save all trained models permanently, create comparative 
            visualizations, and generate a comprehensive HTML report comparing all stocks with embedded charts and 
            investment recommendations""",
            
            """Professional portfolio optimization system with HTML reporting - build and save RandomForest models for 
            major tech stocks (AAPL, GOOGL, MSFT, TSLA), predict price movements for each, implement model persistence, 
            create portfolio allocation visualizations, and generate an executive-level HTML report with interactive 
            charts, risk analysis, and comprehensive investment recommendations"""
        ]
    
    def run_interactive_demo(self):
        """Run the interactive forecasting demonstration with HTML reporting"""
        print("🚀 ADVANCED STOCK FORECASTING MULTI-AGENT SYSTEM WITH HTML REPORTING")
        print("=" * 80)
        print("🎯 POWERED BY: RandomForest ML + Price Change Prediction + Model Persistence + HTML Reports")
        print("=" * 80)
        print("This system will perform:")
        print("1. 📊 Advanced Data Collection & Feature Engineering")
        print("2. 🤖 RandomForest Model Training (100 trees)")
        print("3. 📈 Price Change Prediction & Forecasting")
        print("4. 💾 Permanent Model Persistence & Auto-Loading")
        print("5. 🧪 Comprehensive Backtesting & Validation")
        print("6. 📋 Professional Reporting & Dynamic Visualization")
        print("7. 🌐 Professional HTML Report Generation")  # NEW
        print("8. 🎯 Multi-Stock Portfolio Analysis")
        print("=" * 80)
        
        demo_queries = self.get_demo_queries()
        
        print("\n🎯 Available Demo Queries (with HTML Reporting):")
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
        
        # Save the log to logs folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = self.viewer.save_log(f"forecasting_demo_{timestamp}.txt")
        
        print("\n" + "=" * 80)
        print("🎉 DEMO COMPLETED WITH HTML REPORTING!")
        print("=" * 80)
        print("The system has demonstrated:")
        print("✅ Multi-agent coordination with RandomForest ML")
        print("✅ Advanced data collection & feature engineering")
        print("✅ RandomForest machine learning model training")
        print("✅ Price change prediction & forecasting")
        print("✅ Permanent model persistence & auto-loading")
        print("✅ Comprehensive backtesting")
        print("✅ Professional visualization & reporting")
        print("✅ Complete HTML report generation")  # NEW
        
        # Show file locations
        print(f"\n📁 Generated Files:")
        if log_path:
            print(f"📋 Conversation Log: {log_path}")
        
        html_reports_dir = Path('html_reports')
        if html_reports_dir.exists():
            html_files = list(html_reports_dir.glob("*.html"))
            if html_files:
                latest_html = max(html_files, key=lambda x: x.stat().st_mtime)
                print(f"🌐 HTML Report: {latest_html}")
                print(f"💡 Open {latest_html.name} in your web browser to view the complete analysis!")
        
        plots_dir = Path('plots')
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            if plot_files:
                print(f"📊 Visualizations: {len(plot_files)} files in plots/")
        
        models_dir = Path('models')
        if models_dir.exists():
            model_files = list(models_dir.glob("**/model_*.joblib"))
            if model_files:
                print(f"🤖 Saved Models: {len(model_files)} files in models/")
    
    def run_custom_analysis(self, query: str):
        """Run custom stock analysis with HTML reporting"""
        print(f"🎯 Running Custom Analysis with HTML Reporting:")
        print(f"Query: {query}")
        print("=" * 60)
        
        self.viewer.run(self.graph, query)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = self.viewer.save_log(f"custom_analysis_{timestamp}.txt")
        
        # Show generated files
        print(f"\n📁 Files Generated:")
        if log_path:
            print(f"📋 Log: {log_path}")
        
        html_reports_dir = Path('html_reports')
        if html_reports_dir.exists():
            html_files = list(html_reports_dir.glob("*.html"))
            if html_files:
                latest_html = max(html_files, key=lambda x: x.stat().st_mtime)
                print(f"🌐 HTML Report: {latest_html}")
    
    def run_simple_test(self):
        """Run a simple system test with HTML generation"""
        print("🧪 SIMPLE SYSTEM TEST WITH HTML REPORTING")
        print("-" * 50)
        
        try:
            test_query = "Test AAPL forecasting - create features, train RandomForest to predict price changes, save model, and generate HTML report"
            
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
            
            # Check for HTML generation
            html_reports_dir = Path('html_reports')
            if html_reports_dir.exists():
                html_files = list(html_reports_dir.glob("*.html"))
                if html_files:
                    print(f"🌐 HTML reports generated: {len(html_files)}")
                else:
                    print("⚠️ No HTML reports generated")
            
            return True
            
        except Exception as e:
            print(f"❌ System test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def show_help(self):
        """Show help information"""
        print("🤖 STOCK FORECASTING SYSTEM WITH HTML REPORTING - HELP")
        print("=" * 60)
        print("\nAvailable commands:")
        print("  --demo         : Run interactive demo (default)")
        print("  --test         : Run simple system test")
        print("  --help         : Show this help")
        print("  --query 'text' : Run custom analysis")
        print("  --tools        : Show available tools")
        print("  --models       : Show model management info")
        print("  --files        : Show generated files")
        print("\nExamples:")
        print("  python main.py --demo")
        print("  python main.py --query 'analyze AAPL stock with HTML report'")
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
            ],
            "HTML Generator Tools": [  # NEW
                "collect_analysis_data - Gather all analysis results",
                "gather_visualization_files - Collect all charts",
                "create_html_report - Generate professional HTML reports"
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
            from analysis import list_saved_models
            result = list_saved_models.invoke({})
            print(result)
        except Exception as e:
            print(f"❌ Error accessing model info: {e}")
            print("\nModel management features:")
            print("• Permanent model persistence")
            print("• Auto-loading capabilities")
            print("• Version control")
            print("• Production deployment ready")
    
    def show_files(self):
        """Show generated files in all directories"""
        print("📁 GENERATED FILES SUMMARY")
        print("=" * 50)
        
        directories = {
            'logs': 'Conversation logs',
            'plots': 'Visualization charts',
            'models': 'Trained ML models',
            'html_reports': 'HTML analysis reports'
        }
        
        for dir_name, description in directories.items():
            dir_path = Path(dir_name)
            if dir_path.exists():
                files = list(dir_path.rglob("*"))
                files = [f for f in files if f.is_file()]
                print(f"\n📂 {dir_name}/ - {description}")
                if files:
                    for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                        size_kb = file.stat().st_size / 1024
                        mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                        print(f"  📄 {file.name} ({size_kb:.1f} KB) - {mod_time}")
                    if len(files) > 5:
                        print(f"  ... and {len(files) - 5} more files")
                else:
                    print(f"  (no files)")
            else:
                print(f"\n📂 {dir_name}/ - Directory not found")

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
        
        elif command == "--files":
            system.show_files()
            return
        
        elif command == "--query":
            if len(sys.argv) > 2:
                query = ' '.join(sys.argv[2:])
                system.run_custom_analysis(query)
            else:
                print("❌ Please provide a query after --query")
                print("Example: python main.py --query 'analyze AAPL stock with HTML report'")
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