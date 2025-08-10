"""
Bond Trading Cost Optimization - QUBO Matrix for Actual Trading
===============================================================
This version optimizes actual trading costs and expected profits
"""

import numpy as np
import pandas as pd

def build_trading_qubo_matrix(df, trading_strategy='cost_minimization'):
    """
    Build QUBO matrix for actual bond trading optimization
    
    Variables:
    - x_i = 1 if we BUY bond i, 0 otherwise
    
    Objectives:
    1. Minimize purchase costs
    2. Maximize expected returns  
    3. Minimize transaction costs
    4. Consider liquidity constraints
    """
    
    print(f"\n💰 BUILDING TRADING OPTIMIZATION QUBO")
    print(f"Strategy: {trading_strategy}")
    print("-" * 50)
    
    n_bonds = len(df)
    Q = np.zeros((n_bonds, n_bonds))
    
    # Required columns for trading optimization
    required_cols = {
        'price': 'Current market price',
        'oas': 'Credit spread (basis points)', 
        'spreadDur': 'Duration (interest rate sensitivity)',
        'fund_enriched.notionalMktValue': 'Position size/liquidity',
    }
    
    # Check if we have trading-relevant data
    available_cols = {}
    for col, description in required_cols.items():
        if col in df.columns:
            available_cols[col] = description
            print(f"   ✅ {col}: {description}")
        else:
            print(f"   ❌ Missing: {col} ({description})")
    
    if len(available_cols) < 2:
        print(f"   ⚠️ Insufficient trading data, falling back to simple cost optimization")
        return simple_cost_optimization(df)
    
    # === TRADING OBJECTIVE COMPONENTS ===
    
    if trading_strategy == 'cost_minimization':
        # Objective: Minimize total purchase cost
        return build_cost_minimization_qubo(df, available_cols)
    
    elif trading_strategy == 'yield_maximization':
        # Objective: Maximize expected yield/return
        return build_yield_maximization_qubo(df, available_cols)
    
    elif trading_strategy == 'risk_adjusted_return':
        # Objective: Maximize return per unit of risk
        return build_risk_adjusted_qubo(df, available_cols)
    
    else:
        return build_balanced_trading_qubo(df, available_cols)

def build_cost_minimization_qubo(df, available_cols):
    """Minimize the cost of purchasing bonds"""
    n_bonds = len(df)
    Q = np.zeros((n_bonds, n_bonds))
    
    print(f"   🎯 Objective: MINIMIZE PURCHASE COSTS")
    
    if 'price' in available_cols:
        prices = df['price'].fillna(df['price'].median()).values
        
        # Normalize prices to [0,1] for better optimization
        min_price, max_price = prices.min(), prices.max()
        if max_price > min_price:
            normalized_prices = (prices - min_price) / (max_price - min_price)
        else:
            normalized_prices = np.ones(n_bonds)
        
        # Diagonal terms: cost of buying each bond individually
        for i in range(n_bonds):
            Q[i, i] += normalized_prices[i]  # Higher price = higher cost
        
        print(f"   💵 Price optimization: ${min_price:.2f} - ${max_price:.2f}")
    
    # Add transaction cost penalties for illiquid bonds
    if 'fund_enriched.notionalMktValue' in available_cols:
        market_values = df['fund_enriched.notionalMktValue'].fillna(0).values
        
        # Penalty for low liquidity (small market values)
        liquidity_penalty = 0.1
        for i in range(n_bonds):
            if market_values[i] < np.percentile(market_values, 25):  # Bottom quartile
                Q[i, i] += liquidity_penalty
        
        print(f"   🏦 Added liquidity penalties for small positions")
    
    # Portfolio size constraint (don't buy everything)
    target_bonds = max(3, min(8, n_bonds // 3))
    constraint_penalty = 1.0
    
    for i in range(n_bonds):
        for j in range(n_bonds):
            if i == j:
                Q[i, j] += constraint_penalty * (1 - 2 * target_bonds)
            else:
                Q[i, j] += constraint_penalty
    
    print(f"   📊 Target portfolio size: ~{target_bonds} bonds")
    
    return Q, {'strategy': 'cost_minimization', 'target_bonds': target_bonds}

def build_yield_maximization_qubo(df, available_cols):
    """Maximize expected yield/return from bonds"""
    n_bonds = len(df)
    Q = np.zeros((n_bonds, n_bonds))
    
    print(f"   🎯 Objective: MAXIMIZE EXPECTED YIELD")
    
    # Use credit spread (OAS) as proxy for yield
    if 'oas' in available_cols:
        spreads = df['oas'].fillna(df['oas'].median()).values
        
        # Higher spread = higher yield = better (but also higher risk)
        # Normalize spreads
        min_spread, max_spread = spreads.min(), spreads.max()
        if max_spread > min_spread:
            normalized_spreads = (spreads - min_spread) / (max_spread - min_spread)
        else:
            normalized_spreads = np.ones(n_bonds)
        
        # NEGATIVE on diagonal because we want to MAXIMIZE yield (minimize negative yield)
        for i in range(n_bonds):
            Q[i, i] -= normalized_spreads[i]  # Higher spread = lower cost (better)
        
        print(f"   📈 Spread range: {min_spread:.0f} - {max_spread:.0f} bps")
    
    # Risk adjustment using duration
    if 'spreadDur' in available_cols:
        durations = df['spreadDur'].fillna(df['spreadDur'].median()).values
        
        # Penalty for very high duration (interest rate risk)
        duration_penalty = 0.05
        high_duration_threshold = np.percentile(durations, 75)
        
        for i in range(n_bonds):
            if durations[i] > high_duration_threshold:
                Q[i, i] += duration_penalty * (durations[i] - high_duration_threshold)
        
        print(f"   ⏰ Duration risk adjustment applied")
    
    # Portfolio constraints
    target_bonds = max(4, min(10, n_bonds // 2))
    constraint_penalty = 0.5
    
    for i in range(n_bonds):
        for j in range(n_bonds):
            if i == j:
                Q[i, j] += constraint_penalty * (1 - 2 * target_bonds)
            else:
                Q[i, j] += constraint_penalty
    
    return Q, {'strategy': 'yield_maximization', 'target_bonds': target_bonds}

def build_risk_adjusted_qubo(df, available_cols):
    """Maximize return per unit of risk (Sharpe ratio concept)"""
    n_bonds = len(df)
    Q = np.zeros((n_bonds, n_bonds))
    
    print(f"   🎯 Objective: MAXIMIZE RISK-ADJUSTED RETURN")
    
    # Expected return proxy (credit spread)
    expected_returns = np.zeros(n_bonds)
    if 'oas' in available_cols:
        spreads = df['oas'].fillna(df['oas'].median()).values
        expected_returns = spreads / 100  # Convert bps to percentage
    
    # Risk proxy (duration + credit quality)
    risk_scores = np.ones(n_bonds)
    if 'spreadDur' in available_cols:
        durations = df['spreadDur'].fillna(df['spreadDur'].median()).values
        risk_scores *= (durations / durations.max())  # Normalize
    
    # Risk-adjusted scores
    risk_adjusted_scores = np.where(risk_scores > 0, expected_returns / risk_scores, 0)
    
    # Normalize for optimization
    if risk_adjusted_scores.max() > risk_adjusted_scores.min():
        normalized_scores = (risk_adjusted_scores - risk_adjusted_scores.min()) / \
                          (risk_adjusted_scores.max() - risk_adjusted_scores.min())
    else:
        normalized_scores = np.ones(n_bonds)
    
    # Negative because we want to maximize (minimize negative)
    for i in range(n_bonds):
        Q[i, i] -= normalized_scores[i]
    
    print(f"   ⚖️ Risk-adjusted scores: {risk_adjusted_scores.min():.3f} - {risk_adjusted_scores.max():.3f}")
    
    # Diversification bonus (encourage selecting bonds with different characteristics)
    diversification_bonus = 0.05
    if 'oas' in available_cols and 'spreadDur' in available_cols:
        spreads = df['oas'].values
        durations = df['spreadDur'].values
        
        for i in range(n_bonds):
            for j in range(i+1, n_bonds):
                # Bonus for selecting bonds with different risk profiles
                spread_diff = abs(spreads[i] - spreads[j]) / (spreads.max() - spreads.min() + 1e-6)
                duration_diff = abs(durations[i] - durations[j]) / (durations.max() - durations.min() + 1e-6)
                
                diversity_score = (spread_diff + duration_diff) / 2
                Q[i, j] -= diversification_bonus * diversity_score
                Q[j, i] = Q[i, j]  # Symmetric
    
    # Portfolio size constraint
    target_bonds = max(5, min(12, n_bonds // 2))
    constraint_penalty = 0.3
    
    for i in range(n_bonds):
        for j in range(n_bonds):
            if i == j:
                Q[i, j] += constraint_penalty * (1 - 2 * target_bonds)
            else:
                Q[i, j] += constraint_penalty
    
    return Q, {'strategy': 'risk_adjusted_return', 'target_bonds': target_bonds}

def simple_cost_optimization(df):
    """Fallback when insufficient data available"""
    n_bonds = len(df)
    Q = np.zeros((n_bonds, n_bonds))
    
    print(f"   🎯 Fallback: SIMPLE DIVERSIFICATION")
    
    # Just encourage selecting a reasonable number of bonds
    target_bonds = max(3, min(8, n_bonds // 3))
    
    for i in range(n_bonds):
        for j in range(n_bonds):
            if i == j:
                Q[i, j] = 1 - 2 * target_bonds / n_bonds
            else:
                Q[i, j] = 1 / n_bonds
    
    return Q, {'strategy': 'simple_diversification', 'target_bonds': target_bonds}

def interpret_trading_results(solution, df, strategy_info):
    """Interpret the quantum solution in trading terms"""
    selected_bonds = df[solution.astype(bool)]
    
    print(f"\n💼 TRADING STRATEGY RESULTS")
    print("-" * 50)
    print(f"Strategy: {strategy_info['strategy']}")
    print(f"Bonds selected: {len(selected_bonds)}")
    
    if len(selected_bonds) == 0:
        print("❌ No bonds selected!")
        return
    
    # Calculate trading metrics
    if 'price' in selected_bonds.columns:
        total_cost = selected_bonds['price'].sum()
        avg_price = selected_bonds['price'].mean()
        print(f"💰 Total purchase cost: ${total_cost:,.2f}")
        print(f"💵 Average bond price: ${avg_price:.2f}")
    
    if 'oas' in selected_bonds.columns:
        avg_spread = selected_bonds['oas'].mean()
        total_yield_potential = selected_bonds['oas'].sum()
        print(f"📈 Average credit spread: {avg_spread:.0f} bps")
        print(f"🎯 Total yield potential: {total_yield_potential:.0f} bps")
    
    if 'spreadDur' in selected_bonds.columns:
        avg_duration = selected_bonds['spreadDur'].mean()
        duration_risk = selected_bonds['spreadDur'].std()
        print(f"⏰ Average duration: {avg_duration:.2f} years")
        print(f"⚠️ Duration risk (std): {duration_risk:.2f}")
    
    if 'fund_enriched.notionalMktValue' in selected_bonds.columns:
        total_liquidity = selected_bonds['fund_enriched.notionalMktValue'].sum()
        print(f"🏦 Total liquidity: ${total_liquidity:,.0f}")
    
    # Trading recommendation
    print(f"\n💡 TRADING RECOMMENDATION:")
    if strategy_info['strategy'] == 'cost_minimization':
        print(f"   📉 BUY these bonds to minimize purchase costs")
        print(f"   🎯 Focus on lower-priced, liquid instruments")
    elif strategy_info['strategy'] == 'yield_maximization':
        print(f"   📈 BUY these bonds to maximize yield potential") 
        print(f"   ⚠️ Higher yield may mean higher credit risk")
    elif strategy_info['strategy'] == 'risk_adjusted_return':
        print(f"   ⚖️ BUY these bonds for optimal risk-adjusted returns")
        print(f"   🎯 Balanced approach to risk and return")

# Example usage
if __name__ == "__main__":
    # This would be integrated into the main quantum optimization
    print("💰 BOND TRADING OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Example with sample data
    sample_data = {
        'price': [98.5, 102.3, 95.7, 101.1, 99.8],
        'oas': [125, 87, 156, 98, 134],
        'spreadDur': [4.2, 6.1, 3.8, 5.5, 4.9],
        'fund_enriched.notionalMktValue': [1000000, 2500000, 800000, 1800000, 1200000]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Test different strategies
    strategies = ['cost_minimization', 'yield_maximization', 'risk_adjusted_return']
    
    for strategy in strategies:
        print(f"\n🔍 Testing {strategy}:")
        Q, info = build_trading_qubo_matrix(df_sample, strategy)
        print(f"   QUBO matrix eigenvalues: [{np.linalg.eigvals(Q).min():.3f}, {np.linalg.eigvals(Q).max():.3f}]")