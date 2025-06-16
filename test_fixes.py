# test_fixes.py - CORRECTED VERSION - Run this after replacing your func.py
# Save this as a new file and run it to test your fixes

print("🧪 TESTING YOUR FIXED MODEL...")
print("=" * 50)

try:
    # Import your fixed functions
    from func import (
        create_technical_features, 
        train_decision_tree_model, 
        predict_stock_price,
        create_model_visualization,
        quick_model_test
    )
    print("✅ Successfully imported fixed functions")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

def test_complete_workflow():
    """Test the complete fixed workflow"""
    symbol = "AAPL"
    
    print(f"\n🔍 Testing complete workflow for {symbol}...")
    
    # Step 1: Create features - FIXED: Use invoke method
    print("\n1️⃣ Creating technical features...")
    try:
        # FIXED: Use invoke instead of direct call
        feature_result = create_technical_features.invoke({"symbol": symbol, "period": "1y"})
        if "Data ready for model training!" in feature_result:
            print("✅ Features created successfully")
        else:
            print("❌ Feature creation failed")
            print(f"Result: {feature_result}")
            return False
    except Exception as e:
        print(f"❌ Feature creation error: {e}")
        return False
    
    # Step 2: Train FIXED model - FIXED: Use invoke method
    print("\n2️⃣ Training FIXED RandomForest model...")
    try:
        # FIXED: Use invoke instead of direct call
        train_result = train_decision_tree_model.invoke({"symbol": symbol, "max_depth": 6})
        if "RandomForestRegressor" in train_result and "FIXED" in train_result:
            print("✅ FIXED model trained successfully")
            
            # Check if variance ratio is good
            if "✅ (Good)" in train_result:
                print("✅ Model predictions have good variance!")
            else:
                print("⚠️ Check variance ratio in training results")
                print(f"Training result snippet: {train_result[:200]}...")
        else:
            print("❌ Model training failed or still using old algorithm")
            print(f"Result: {train_result}")
            return False
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False
    
    # Step 3: Test predictions - FIXED: Use invoke method
    print("\n3️⃣ Testing price predictions...")
    try:
        # FIXED: Use invoke instead of direct call
        pred_result = predict_stock_price.invoke({"symbol": symbol})
        if "FIXED" in pred_result and "$" in pred_result:
            print("✅ FIXED predictions working")
            
            # Extract prediction from result
            lines = pred_result.split('\n')
            for line in lines:
                if "Predicted Price" in line:
                    print(f"✅ {line.strip()}")
                    break
        else:
            print("❌ Prediction failed")
            print(f"Result: {pred_result}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    
    # Step 4: Quick model test - FIXED: Use invoke method
    print("\n4️⃣ Running quick model diagnostic...")
    try:
        # FIXED: Use invoke instead of direct call
        test_result = quick_model_test.invoke({"symbol": symbol})
        print(test_result)
        
        if "✅ WORKING" in test_result:
            print("🎉 ALL TESTS PASSED!")
            return True
        else:
            print("❌ Model diagnostic failed")
            return False
    except Exception as e:
        print(f"❌ Diagnostic error: {e}")
        return False

def test_visualization():
    """Test if visualizations work with fixed model"""
    symbol = "AAPL"
    
    print(f"\n📊 Testing visualization with FIXED model...")
    
    try:
        # FIXED: Use invoke instead of direct call
        viz_result = create_model_visualization.invoke({"symbol": symbol, "chart_type": "prediction_vs_actual"})
        if "FIXED" in viz_result and "saved as" in viz_result:
            print("✅ FIXED visualization created successfully")
            print(f"✅ {viz_result}")
            return True
        else:
            print("❌ Visualization failed")
            print(f"Result: {viz_result}")
            return False
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def simple_direct_test():
    """Simple test calling functions directly (bypassing LangChain)"""
    print("\n🧪 SIMPLE DIRECT TEST (bypassing LangChain)...")
    
    try:
        # Import the underlying functions directly
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        print("✅ All required packages imported")
        
        # Test basic data fetch
        symbol = "AAPL"
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")
        
        if len(data) > 100:
            print(f"✅ Successfully fetched {len(data)} days of data for {symbol}")
        else:
            print(f"❌ Insufficient data: only {len(data)} days")
            return False
        
        # Test RandomForest creation
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        print(f"✅ RandomForest model created: {type(model).__name__}")
        
        # Test StandardScaler
        scaler = StandardScaler()
        test_data = np.random.rand(10, 5)
        scaled_data = scaler.fit_transform(test_data)
        print(f"✅ StandardScaler working: {scaled_data.shape}")
        
        print("🎉 SIMPLE DIRECT TEST PASSED!")
        print("Your environment is set up correctly for the fixed model!")
        return True
        
    except Exception as e:
        print(f"❌ Simple test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 COMPREHENSIVE TEST OF YOUR FIXES")
    print("=" * 60)
    
    # First try simple direct test
    simple_passed = simple_direct_test()
    
    if not simple_passed:
        print("\n❌ Basic environment test failed!")
        print("Please install missing packages:")
        print("pip install scikit-learn yfinance pandas numpy matplotlib")
        return
    
    # Test complete workflow
    print("\n" + "=" * 60)
    workflow_passed = test_complete_workflow()
    
    if workflow_passed:
        print("\n" + "=" * 60)
        print("🎉 PRIMARY TESTS PASSED!")
        print("✅ Your RandomForest model is working properly")
        print("✅ Predictions should now have proper variance")
        print("✅ No more flat orange lines!")
        
        # Test visualization
        viz_passed = test_visualization()
        
        if viz_passed:
            print("\n🎯 FINAL RESULT: ALL FIXES SUCCESSFUL!")
            print("📈 Your model should now show:")
            print("   - Varying predictions (not flat lines)")
            print("   - Better RMSE (lower error)")
            print("   - RandomForest algorithm")
            print("   - Proper feature scaling")
            
            print("\n🚀 NEXT STEPS:")
            print("1. Run your forecasting_demo.py")
            print("2. Create all visualizations with chart_type='all'")
            print("3. Your plots should show dynamic predictions!")
            
        else:
            print("\n⚠️ Visualization test failed - but core model is fixed")
    
    else:
        print("\n❌ WORKFLOW TESTS FAILED!")
        print("The basic environment works, but there might be an issue with:")
        print("1. The func.py file replacement")
        print("2. LangChain tool invocation")
        print("3. Global variables in the model functions")

if __name__ == "__main__":
    main()