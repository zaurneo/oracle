# debug_agent_flow.py - Find out why agents aren't responding
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, MessagesState

load_dotenv()

# Import your modules
try:
    from agents import project_owner, data_engineer, model_executer, reporter
    print("✅ Agents imported successfully")
except Exception as e:
    print(f"❌ Agent import error: {e}")
    exit(1)

def debug_single_agent():
    """Test just one agent first"""
    print("\n🔍 TESTING SINGLE AGENT (project_owner)...")
    
    try:
        # Create simple graph with just project_owner
        simple_graph = (
            StateGraph(MessagesState)
            .add_node(project_owner)
            .add_edge(START, "project_owner")
            .compile()
        )
        
        print("✅ Simple graph compiled")
        
        # Test with invoke (simpler than stream)
        result = simple_graph.invoke({
            "messages": [HumanMessage(content="Hello, can you help me analyze TSLA stock?")]
        })
        
        print(f"✅ Graph executed successfully")
        print(f"📝 Total messages in result: {len(result.get('messages', []))}")
        
        # Print all messages
        for i, msg in enumerate(result.get('messages', [])):
            print(f"Message {i+1}: {type(msg).__name__} - {getattr(msg, 'role', 'no role')}")
            if hasattr(msg, 'content'):
                content = str(msg.content)
                preview = content[:150] + "..." if len(content) > 150 else content
                print(f"  Content: {preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Single agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_api_keys():
    """Check if API keys work"""
    print("\n🔍 TESTING API KEYS...")
    
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import ChatOpenAI
        
        claude_key = os.environ.get("claude_api_key", "")
        gpt_key = os.environ.get("gpt_api_key", "")
        
        if not claude_key:
            print("❌ No Claude API key found")
            return False
        
        if not gpt_key:
            print("❌ No GPT API key found") 
            return False
        
        # Test Claude
        try:
            claude_model = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=claude_key
            )
            claude_response = claude_model.invoke([HumanMessage(content="Say 'Claude works'")])
            print(f"✅ Claude API works: {claude_response.content}")
        except Exception as e:
            print(f"❌ Claude API failed: {e}")
            return False
        
        # Test GPT
        try:
            gpt_model = ChatOpenAI(
                model="gpt-4o-2024-08-06",
                api_key=gpt_key
            )
            gpt_response = gpt_model.invoke([HumanMessage(content="Say 'GPT works'")])
            print(f"✅ GPT API works: {gpt_response.content}")
        except Exception as e:
            print(f"❌ GPT API failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

def debug_tools():
    """Test if tools work"""
    print("\n🔍 TESTING TOOLS...")
    
    try:
        from func import get_stock_price, get_technical_indicators
        
        # Test stock price tool
        result = get_stock_price("AAPL")
        print(f"✅ get_stock_price works: {result[:100]}...")
        
        # Test technical indicators tool  
        result2 = get_technical_indicators("AAPL")
        print(f"✅ get_technical_indicators works: {result2[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_full_graph_verbose():
    """Test full graph with verbose output"""
    print("\n🔍 TESTING FULL GRAPH WITH VERBOSE OUTPUT...")
    
    try:
        # Create the full graph
        full_graph = (
            StateGraph(MessagesState)
            .add_node(project_owner)
            .add_node(data_engineer) 
            .add_node(model_executer)
            .add_node(reporter)
            .add_edge(START, "project_owner")
            .compile()
        )
        
        print("✅ Full graph compiled")
        
        # Use invoke instead of stream for debugging
        print("📤 Sending message to graph...")
        
        result = full_graph.invoke({
            "messages": [HumanMessage(content="analyze AAPL stock")]
        })
        
        print(f"📥 Received result with {len(result.get('messages', []))} messages")
        
        # Print each message in detail
        for i, msg in enumerate(result.get('messages', [])):
            print(f"\n--- Message {i+1} ---")
            print(f"Type: {type(msg).__name__}")
            print(f"Role: {getattr(msg, 'role', 'no role')}")
            print(f"Content: {getattr(msg, 'content', 'no content')}")
            if hasattr(msg, 'name'):
                print(f"Name: {msg.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_streaming():
    """Test streaming specifically"""
    print("\n🔍 TESTING STREAMING...")
    
    try:
        # Create graph
        graph = (
            StateGraph(MessagesState)
            .add_node(project_owner)
            .add_node(data_engineer)
            .add_node(model_executer) 
            .add_node(reporter)
            .add_edge(START, "project_owner")
            .compile()
        )
        
        chunk_count = 0
        
        for chunk in graph.stream({
            "messages": [HumanMessage(content="analyze AAPL stock")]
        }):
            chunk_count += 1
            print(f"📦 Chunk {chunk_count}: {list(chunk.keys())}")
            
            for agent_name, data in chunk.items():
                messages = data.get('messages', [])
                print(f"  {agent_name}: {len(messages)} messages")
                
                if messages:
                    latest = messages[-1]
                    print(f"    Latest: {type(latest).__name__}")
                    if hasattr(latest, 'content'):
                        content_preview = str(latest.content)[:100]
                        print(f"    Content: {content_preview}...")
            
            # Stop after 5 chunks to avoid infinite loop
            if chunk_count >= 5:
                print("⏹️  Stopping after 5 chunks")
                break
        
        print(f"✅ Streaming test completed. Total chunks: {chunk_count}")
        return chunk_count > 1
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("🚀 COMPREHENSIVE AGENT DEBUG")
    print("="*60)
    
    tests = [
        ("API Keys", debug_api_keys),
        ("Tools", debug_tools), 
        ("Single Agent", debug_single_agent),
        ("Full Graph Invoke", debug_full_graph_verbose),
        ("Streaming", debug_streaming)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        if test_func():
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED - This is likely your issue!")
            print("\nFix this before continuing.")
            return
    
    print(f"\n🎉 ALL TESTS PASSED!")
    print("If all tests pass but live conversation doesn't work,")
    print("the issue might be in the LiveConversationViewer itself.")

if __name__ == "__main__":
    main()


# QUICK TEST: Minimal working example
def quick_test():
    """Absolute minimal test"""
    print("\n🚀 QUICK TEST - MINIMAL EXAMPLE")
    
    try:
        # Just test project_owner alone
        result = project_owner.invoke({
            "messages": [HumanMessage(content="Say hello")]
        })
        
        print("✅ Project owner responded!")
        print(f"Messages: {len(result.get('messages', []))}")
        
        if result.get('messages'):
            latest = result['messages'][-1]
            print(f"Response: {getattr(latest, 'content', 'no content')}")
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")

# Uncomment to run quick test
# quick_test()