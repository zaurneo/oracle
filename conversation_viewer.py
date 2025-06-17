# conversation_viewer.py - Updated with logs folder support
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Union
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, MessagesState

load_dotenv()

class ConversationViewer:
    def __init__(self):
        self.colors = {
            'project_owner': '\033[94m',     # Blue
            'data_engineer': '\033[92m',     # Green  
            'model_executer': '\033[93m',    # Yellow
            'reporter': '\033[95m',          # Magenta
            'html_generator': '\033[96m',    # Cyan - NEW
            'user': '\033[97m',              # White
            'tool': '\033[91m',              # Red
            'system': '\033[90m',            # Gray
            'reset': '\033[0m'
        }
        self.message_history = []
        self.seen_ids = set()
        self.last_handoff = None
        
        # Ensure logs directory exists
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)
        
    def print_header(self):
        """Print the conversation header"""
        print("\n" + "="*90)
        print("🚀 LIVE MULTI-AGENT CONVERSATION WITH HTML REPORTING")
        print("="*90)
        print("👤 USER: White | 🤖 PROJECT_OWNER: Blue | 🔧 DATA_ENGINEER: Green")  
        print("⚙️  MODEL_EXECUTER: Yellow | 📊 REPORTER: Magenta | 🌐 HTML_GENERATOR: Cyan")
        print("🛠️  TOOLS: Red | 📋 SYSTEM: Gray")
        print("="*90 + "\n")
    
    def extract_text_content(self, content):
        """Extract readable text from various content formats"""
        if content is None:
            return ""
            
        if isinstance(content, str):
            return content.strip()
        
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                else:
                    text_parts.append(str(item))
            return ' '.join(text_parts).strip()
        
        return str(content).strip()
    
    def format_and_print(self, agent_name: str, content: str, icon: str = "💬"):
        """Format and print a message with colors"""
        if not content or content in ["[]", "", "None"]:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = self.colors.get(agent_name, self.colors['system'])
        
        # Split into lines and print each
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                if i == 0:
                    print(f"{color}[{timestamp}] {icon} {agent_name.upper()}: {line}{self.colors['reset']}")
                else:
                    # Indent continuation lines
                    print(f"{color}{''.ljust(13)}{line}{self.colors['reset']}")
        
        # Record in history with timestamp
        self.message_history.append((timestamp, agent_name, content))
    
    def process_message(self, msg, current_agent):
        """Process a single message and display it"""
        # Get message ID
        msg_id = None
        if hasattr(msg, 'id'):
            msg_id = msg.id
        elif isinstance(msg, dict) and 'id' in msg:
            msg_id = msg['id']
            
        # Skip if already seen
        if msg_id and msg_id in self.seen_ids:
            return
        if msg_id:
            self.seen_ids.add(msg_id)
        
        # Handle AIMessage objects
        if isinstance(msg, AIMessage):
            content = msg.content
            name = getattr(msg, 'name', current_agent)
            tool_calls = getattr(msg, 'tool_calls', [])
            
            # Display agent message if there's content
            if content:
                text = self.extract_text_content(content)
                if text:
                    self.format_and_print(name, text, "💬")
            
            # Display tool calls
            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc.get('name', 'unknown')
                    if 'transfer_to_' not in tool_name:
                        self.format_and_print(name, f"Using tool: {tool_name}", "🔧")
            return
        
        # Handle ToolMessage objects
        if isinstance(msg, ToolMessage):
            content = msg.content
            tool_name = msg.name
            
            # Handle transfers
            if 'transfer_to_' in tool_name and 'Successfully transferred' in content:
                target = tool_name.replace('transfer_to_', '')
                handoff_key = f"{current_agent}->{target}"
                if handoff_key != self.last_handoff:
                    self.last_handoff = handoff_key
                    print(f"\n{self.colors['system']}{'─'*60}")
                    print(f"🔄 Handoff: {current_agent} → {target}")
                    print(f"{'─'*60}{self.colors['reset']}\n")
                return
            
            # Show tool results (truncated for readability)
            if content and any(keyword in tool_name for keyword in ['get_', 'create_', 'train_', 'save_']):
                lines = content.strip().split('\n')
                if len(lines) > 8:
                    result_text = '\n'.join(lines[:8]) + f"\n... ({len(lines)-8} more lines)"
                else:
                    result_text = content.strip()
                    
                self.format_and_print('tool', f"{tool_name}:\n{result_text}", "📊")
            return
        
        # Handle dictionary messages
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
            name = msg.get('name', '')
            tool_calls = msg.get('tool_calls', [])
            
            # Human messages
            if role in ['human', 'user']:
                text = self.extract_text_content(content)
                if text:
                    self.format_and_print('user', text, "👤")
                return
            
            # Tool messages
            if role == 'tool':
                tool_name = name or 'unknown_tool'
                
                # Handle transfers
                if 'transfer_to_' in tool_name and 'Successfully transferred' in content:
                    target = tool_name.replace('transfer_to_', '')
                    handoff_key = f"{current_agent}->{target}"
                    if handoff_key != self.last_handoff:
                        self.last_handoff = handoff_key
                        print(f"\n{self.colors['system']}{'─'*60}")
                        print(f"🔄 Handoff: {current_agent} → {target}")
                        print(f"{'─'*60}{self.colors['reset']}\n")
                    return
                
                # Show other tool results
                if content and any(keyword in tool_name for keyword in ['get_', 'create_', 'train_', 'save_']):
                    lines = content.strip().split('\n')
                    if len(lines) > 8:
                        result_text = '\n'.join(lines[:8]) + f"\n... ({len(lines)-8} more lines)"
                    else:
                        result_text = content.strip()
                        
                    self.format_and_print('tool', f"{tool_name}:\n{result_text}", "📊")
                return
            
            # Agent messages with name
            if name in ['project_owner', 'data_engineer', 'model_executer', 'reporter', 'html_generator']:
                # Display content if any
                if content:
                    text = self.extract_text_content(content)
                    if text:
                        self.format_and_print(name, text, "💬")
                
                # Display tool calls
                if tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        if 'transfer_to_' not in tool_name:
                            self.format_and_print(name, f"Using tool: {tool_name}", "🔧")
                return
    
    def run(self, graph, initial_message: str):
        """Run the conversation and display it live"""
        self.print_header()
        
        # Show initial message
        self.format_and_print('user', initial_message, "👤")
        print()
        
        chunk_count = 0
        
        try:
            # Stream the conversation
            for chunk in graph.stream({"messages": [HumanMessage(content=initial_message)]}):
                chunk_count += 1
                
                # Process each agent's messages in the chunk
                for agent_name, data in chunk.items():
                    messages = data.get('messages', [])
                    
                    # Process ALL messages
                    for msg in messages:
                        self.process_message(msg, agent_name)
                
                # Small delay for readability
                time.sleep(0.03)
                
        except KeyboardInterrupt:
            print(f"\n{self.colors['system']}⏹️  Conversation interrupted by user{self.colors['reset']}")
        except Exception as e:
            print(f"\n{self.colors['tool']}❌ Error: {e}{self.colors['reset']}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{self.colors['system']}✅ Conversation completed!{self.colors['reset']}")
        self.print_summary()
    
    def print_summary(self):
        """Print conversation summary"""
        print(f"\n{self.colors['system']}📊 CONVERSATION SUMMARY:")
        
        # Count messages by agent
        agent_counts = {}
        for _, agent, _ in self.message_history:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        print(f"Total exchanges: {len(self.message_history)}")
        if agent_counts:
            for agent, count in sorted(agent_counts.items()):
                color = self.colors.get(agent, self.colors['system'])
                print(f"{color}{agent.upper()}: {count} messages{self.colors['reset']}")
        
        print(f"{self.colors['reset']}")
    
    def save_log(self, filename=None):
        """Save conversation to file in logs directory"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"conversation_{timestamp}.txt"
        
        # Ensure filename goes to logs directory
        log_file = self.logs_dir / filename
        
        if not self.message_history:
            print("No messages to save.")
            return str(log_file)
            
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("MULTI-AGENT STOCK FORECASTING CONVERSATION LOG\n")
                f.write("=" * 70 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                
                for timestamp, agent, message in self.message_history:
                    f.write(f"[{timestamp}] {agent.upper()}:\n")
                    f.write(f"{message}\n")
                    f.write("-" * 50 + "\n\n")
                
                # Add summary
                f.write("\nCONVERSATION SUMMARY:\n")
                f.write("=" * 30 + "\n")
                
                agent_counts = {}
                for _, agent, _ in self.message_history:
                    agent_counts[agent] = agent_counts.get(agent, 0) + 1
                
                f.write(f"Total messages: {len(self.message_history)}\n")
                for agent, count in sorted(agent_counts.items()):
                    f.write(f"{agent.upper()}: {count} messages\n")
            
            print(f"💾 Conversation log saved to: {log_file}")
            return str(log_file)
            
        except Exception as e:
            print(f"❌ Error saving log: {e}")
            return None
    
    def get_conversation_data(self):
        """Get structured conversation data for HTML reports"""
        return {
            'messages': self.message_history,
            'agent_counts': self._get_agent_counts(),
            'total_messages': len(self.message_history),
            'start_time': self.message_history[0][0] if self.message_history else None,
            'end_time': self.message_history[-1][0] if self.message_history else None
        }
    
    def _get_agent_counts(self):
        """Get message counts by agent"""
        agent_counts = {}
        for _, agent, _ in self.message_history:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        return agent_counts


def full_diagnostic(graph):
    """Full diagnostic showing all messages in detail"""
    print("🔬 FULL DIAGNOSTIC - All Messages")
    print("="*60)
    
    initial_msg = HumanMessage(content="analyze AAPL stock with HTML report")
    all_messages = []
    
    for chunk_idx, chunk in enumerate(graph.stream({"messages": [initial_msg]})):
        print(f"\n📦 CHUNK {chunk_idx + 1}: {list(chunk.keys())}")
        
        for agent_name, data in chunk.items():
            messages = data.get('messages', [])
            print(f"\n🤖 {agent_name}: {len(messages)} messages total")
            
            # Show new messages only
            for i, msg in enumerate(messages):
                msg_id = getattr(msg, 'id', None) or (msg.get('id') if isinstance(msg, dict) else None)
                if msg_id in [m.get('id') for m in all_messages if 'id' in m]:
                    continue
                    
                print(f"\n  📧 Message {i+1}:")
                
                msg_info = {'agent': agent_name, 'index': i}
                
                if isinstance(msg, AIMessage):
                    print(f"    Type: AIMessage")
                    print(f"    Name: {msg.name}")
                    print(f"    Content: {repr(msg.content)[:200]}")
                    if msg.tool_calls:
                        print(f"    Tool calls: {[tc.get('name') for tc in msg.tool_calls]}")
                    msg_info.update({'type': 'ai', 'content': msg.content, 'id': msg.id})
                    
                elif isinstance(msg, ToolMessage):
                    print(f"    Type: ToolMessage")
                    print(f"    Tool: {msg.name}")
                    content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    print(f"    Content: {repr(content_preview)}")
                    msg_info.update({'type': 'tool', 'content': msg.content, 'id': msg.id})
                    
                elif isinstance(msg, dict):
                    print(f"    Type: Dict")
                    print(f"    Role: {msg.get('role')}")
                    print(f"    Name: {msg.get('name')}")
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        content_preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"    Content: {repr(content_preview)}")
                    msg_info.update({'type': 'dict', 'content': content, 'id': msg.get('id')})
                
                all_messages.append(msg_info)
        
        if chunk_idx >= 3:
            break
    
    print(f"\n✅ Total unique messages: {len(all_messages)}")
    print("\nMessage flow:")
    for msg in all_messages:
        if msg.get('content'):
            content_preview = str(msg['content'])[:50] + "..." if len(str(msg['content'])) > 50 else str(msg['content'])
            print(f"  {msg['agent']}: {content_preview}")