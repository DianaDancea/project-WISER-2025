# Author: Diana Dancea
# Date: 08/09/2025
# Enhanced file to read and prepare bond data for quantum optimization

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_vanguard_excel_data(repo_path="./WISER_Optimization_VG"):
    """Load the actual Vanguard Excel data file."""
    # Path to the Excel file
    excel_path = Path(__file__).resolve().parent / "data_assets_dump_partial.xlsx"
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    print(f"📊 Loading Vanguard data from: {excel_path}")
    
    # Load the Excel file
    try:
        # Try reading the first sheet
        df = pd.read_excel(excel_path, sheet_name=0)
        print(f"✅ Loaded {len(df)} assets from Excel file")
        
        # Display basic info about the data
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📐 Shape: {df.shape}")
        
        # Show first few rows
        print("\n🔍 First 3 rows:")
        print(df.head(3))
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        raise

def analyze_vanguard_data(df):
    """Analyze the Vanguard data structure and identify portfolio characteristics."""
    print("\n🔬 DATA ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📈 Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"📝 Text columns: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\n⚠️ Missing values:")
        for col, count in missing_cols.items():
            print(f"   {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Identify potential characteristics for optimization
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Common portfolio characteristics to look for
    portfolio_characteristics = {
        'duration': ['duration', 'effective_duration', 'mod_duration', 'dur', 'modified_duration'],
        'yield': ['yield', 'ytm', 'yield_to_maturity', 'current_yield', 'sec_yield'],
        'spread': ['spread', 'credit_spread', 'oas', 'option_adjusted_spread', 'z_spread'],
        'price': ['price', 'market_price', 'clean_price', 'dirty_price', 'unit_price'],
        'market_value': ['market_value', 'mv', 'notional', 'par_value', 'face_value'],
        'rating': ['rating', 'credit_rating', 'moody', 'sp', 'fitch'],
        'sector': ['sector', 'industry', 'asset_class', 'category', 'issuer_type'],
        'maturity': ['maturity', 'time_to_maturity', 'years_to_maturity', 'ttm'],
        'coupon': ['coupon', 'coupon_rate', 'stated_coupon', 'nominal_coupon']
    }
    
    found_characteristics = {}
    for standard_name, variations in portfolio_characteristics.items():
        for col in df.columns:
            for variation in variations:
                if variation.lower() in col.lower():
                    found_characteristics[standard_name] = col
                    break
            if standard_name in found_characteristics:
                break
    
    print(f"\n🎯 Identified portfolio characteristics:")
    for standard, actual in found_characteristics.items():
        if actual in numeric_cols:
            data_preview = df[actual].dropna()
            print(f"   {standard}: {actual} (range: {data_preview.min():.2f} - {data_preview.max():.2f})")
        else:
            print(f"   {standard}: {actual} (categorical)")
    
    return found_characteristics

def clean_for_quantum(df, characteristics):
    """Clean and prepare data specifically for quantum optimization."""
    print("\n🧹 CLEANING DATA FOR QUANTUM OPTIMIZATION")
    print("-" * 50)
    
    cleaned_df = df.copy()
    cleaning_report = {}
    
    # Handle missing values for quantum-relevant characteristics
    for char_name, col_name in characteristics.items():
        if col_name in cleaned_df.columns:
            initial_missing = cleaned_df[col_name].isnull().sum()
            
            if cleaned_df[col_name].dtype in ['float64', 'int64']:
                # Use median for numeric columns
                median_val = cleaned_df[col_name].median()
                cleaned_df[col_name].fillna(median_val, inplace=True)
                cleaning_report[char_name] = f"filled {initial_missing} missing with median {median_val:.3f}"
            else:
                # Use mode for categorical columns
                mode_val = cleaned_df[col_name].mode()
                if len(mode_val) > 0:
                    cleaned_df[col_name].fillna(mode_val.iloc[0], inplace=True)
                    cleaning_report[char_name] = f"filled {initial_missing} missing with mode '{mode_val.iloc[0]}'"
    
    # Remove rows with too many missing values overall
    initial_rows = len(cleaned_df)
    missing_threshold = 0.5  # Remove rows with >50% missing data
    cleaned_df = cleaned_df.dropna(thresh=int(len(cleaned_df.columns) * missing_threshold))
    rows_removed = initial_rows - len(cleaned_df)
    
    if rows_removed > 0:
        print(f"   🗑️ Removed {rows_removed} rows with >{missing_threshold*100}% missing data")
    
    # Report cleaning actions
    for char, action in cleaning_report.items():
        print(f"   🔧 {char}: {action}")
    
    print(f"   ✅ Clean dataset: {len(cleaned_df)} assets ready for quantum optimization")
    return cleaned_df

def normalize_for_quantum(df, characteristics):
    """Normalize characteristics to [0,1] range for better quantum performance."""
    print("\n📐 NORMALIZING DATA FOR QUANTUM CIRCUITS")
    print("-" * 50)
    
    normalized_df = df.copy()
    normalization_info = {}
    
    for char_name, col_name in characteristics.items():
        if col_name in normalized_df.columns and normalized_df[col_name].dtype in ['float64', 'int64']:
            original_data = normalized_df[col_name]
            min_val = original_data.min()
            max_val = original_data.max()
            
            if max_val > min_val:  # Avoid division by zero
                normalized_df[col_name] = (original_data - min_val) / (max_val - min_val)
                normalization_info[char_name] = {
                    'original_range': (min_val, max_val),
                    'column': col_name
                }
                print(f"   📏 {char_name}: [{min_val:.3f}, {max_val:.3f}] → [0, 1]")
            else:
                print(f"   ⚠️ {char_name}: constant value {min_val}, skipping normalization")
    
    return normalized_df, normalization_info

def filter_for_quantum_size(df, max_assets=20, selection_strategy='diverse'):
    """Filter dataset to optimal size for quantum computing."""
    print(f"\n🎯 FILTERING FOR QUANTUM OPTIMIZATION (max {max_assets} assets)")
    print("-" * 50)
    
    if len(df) <= max_assets:
        print(f"   ✅ Dataset size ({len(df)}) already optimal for quantum")
        return df
    
    print(f"   📊 Reducing from {len(df)} to {max_assets} assets using '{selection_strategy}' strategy")
    
    if selection_strategy == 'random':
        # Simple random sampling
        filtered_df = df.sample(n=max_assets, random_state=42)
        
    elif selection_strategy == 'diverse':
        # Try to select diverse assets based on available characteristics
        try:
            # If we have sector/category info, ensure diversity
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                # Group by first categorical column and sample from each group
                first_cat_col = categorical_cols[0]
                groups = df.groupby(first_cat_col)
                samples_per_group = max(1, max_assets // len(groups))
                
                filtered_dfs = []
                for name, group in groups:
                    sample_size = min(len(group), samples_per_group)
                    filtered_dfs.append(group.sample(n=sample_size, random_state=42))
                
                filtered_df = pd.concat(filtered_dfs)
                
                # If we still have too many, randomly sample down
                if len(filtered_df) > max_assets:
                    filtered_df = filtered_df.sample(n=max_assets, random_state=42)
                    
                print(f"   🎨 Selected diverse assets across {len(groups)} categories")
            else:
                # Fall back to random if no categorical data
                filtered_df = df.sample(n=max_assets, random_state=42)
                
        except Exception as e:
            print(f"   ⚠️ Diverse selection failed ({e}), using random sampling")
            filtered_df = df.sample(n=max_assets, random_state=42)
    
    elif selection_strategy == 'top_liquid':
        # Select most liquid assets (highest market value if available)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_cols = [col for col in numeric_cols if any(term in col.lower() 
                     for term in ['market_value', 'mv', 'notional', 'outstanding'])]
        
        if value_cols:
            # Sort by first value column and take top assets
            filtered_df = df.nlargest(max_assets, value_cols[0])
            print(f"   💰 Selected top {max_assets} assets by {value_cols[0]}")
        else:
            # Fall back to random if no value columns
            filtered_df = df.sample(n=max_assets, random_state=42)
            print(f"   ⚠️ No liquidity data found, using random sampling")
    
    else:
        # Default to random sampling
        filtered_df = df.sample(n=max_assets, random_state=42)
    
    filtered_df = filtered_df.reset_index(drop=True)
    print(f"   ✅ Final quantum-optimized dataset: {len(filtered_df)} assets")
    
    return filtered_df

def prepare_quantum_dataset(df, max_assets=20, selection_strategy='diverse'):
    """Complete pipeline to prepare data for quantum optimization."""
    print("\n🚀 PREPARING DATASET FOR QUANTUM OPTIMIZATION")
    print("=" * 60)
    
    # Step 1: Analyze the data structure
    characteristics = analyze_vanguard_data(df)
    
    # Step 2: Clean the data
    cleaned_df = clean_for_quantum(df, characteristics)
    
    # Step 3: Filter to quantum-appropriate size
    filtered_df = filter_for_quantum_size(cleaned_df, max_assets, selection_strategy)
    
    # Step 4: Normalize for quantum circuits
    normalized_df, norm_info = normalize_for_quantum(filtered_df, characteristics)
    
    print(f"\n🎊 QUANTUM DATASET READY!")
    print(f"   📊 Assets: {len(normalized_df)}")
    print(f"   🎯 Characteristics: {len(characteristics)}")
    print(f"   📐 Normalized columns: {len(norm_info)}")
    
    return normalized_df, characteristics, norm_info

if __name__ == "__main__":
    # Test the enhanced data loading and preparation
    print("🧪 TESTING ENHANCED VANGUARD DATA LOADER")
    print("=" * 60)
    
    try:
        # Load raw data
        df = load_vanguard_excel_data()
        
        # Prepare for quantum optimization
        quantum_df, characteristics, norm_info = prepare_quantum_dataset(
            df, 
            max_assets=15, 
            selection_strategy='diverse'
        )
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Original dataset: {len(df)} assets")
        print(f"   Quantum-ready dataset: {len(quantum_df)} assets")
        print(f"   Identified characteristics: {list(characteristics.keys())}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()