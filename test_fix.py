# test_fix.py - Test if the import fix worked
"""
Quick test to verify the analysis package import fix
"""

import sys

def test_import_fix():
    """Test if the import fix worked"""
    print("🧪 Testing Import Fix...")
    
    # Clear any cached imports
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('analysis')]
    for module in modules_to_remove:
        del sys.modules[module]
        print(f"   Cleared cached module: {module}")
    
    try:
        # Fresh import
        print("\n🔄 Attempting fresh import...")
        import analysis
        
        # Test basic tool
        print("🔧 Testing get_stock_price...")
        result = analysis.get_stock_price("AAPL")
        
        if "import error" in result.lower():
            print("❌ STILL GETTING STUB FUNCTIONS!")
            print(f"Result: {result}")
            return False
        else:
            print("✅ SUCCESS! Getting real stock data!")
            print(f"Preview: {result[:150]}...")
            
            # Test multi-stock tool
            print("\n🔧 Testing multi-stock tools...")
            if hasattr(analysis, 'collect_multi_stock_data'):
                print("✅ collect_multi_stock_data available")
                
                if hasattr(analysis, 'create_comparative_html_report'):
                    print("✅ create_comparative_html_report available")
                    print("\n🎉 ALL TESTS PASSED!")
                    print("\n🚀 Ready to run:")
                    print('   python main.py --query "analyze AAPL with clean HTML report"')
                    print('   python main.py --query "compare AAPL, GOOGL, TSLA"')
                    return True
                else:
                    print("❌ create_comparative_html_report missing")
                    return False
            else:
                print("❌ collect_multi_stock_data missing")
                return False
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_import_fix()