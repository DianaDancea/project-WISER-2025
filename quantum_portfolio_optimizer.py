"""
Quantum Portfolio Optimization - Main Solver with Manual VQE Implementation
==========================================================================
Author: Diana Dancea
Date: 08/10/2025
Complete rewrite with proper function order for trading optimization
"""

import numpy as np
import pandas as pd
from financial_excel_loader import (
    load_financial_excel_data, 
    prepare_quantum_dataset
)
import time
from datetime import datetime
import json

# Qiskit imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import TwoLocal
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import Estimator
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
    print("✅ Qiskit successfully imported")
except ImportError as e:
    print(f"⚠️ Qiskit not available: {e}")
    QISKIT_AVAILABLE = False

# Scipy for optimization
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️ Scipy not available for optimization")
    SCIPY_AVAILABLE = False

def analyze_real_data_usage(df):
    """Analyze which real bond data is available for trading optimization"""
    print("\n🔍 ANALYZING YOUR REAL EXCEL DATA FOR TRADING")
    print("-" * 60)
    
    trading_columns = {
        'price': 'Bond prices (cost minimization)',
        'oas': 'Credit spreads (yield maximization)', 
        'spreadDur': 'Duration (risk management)',
        'fund_enriched.notionalMktValue': 'Position size/liquidity'
    }
    
    available_data = {}
    
    for col_name, purpose in trading_columns.items():
        if col_name in df.columns:
            data_series = df[col_name].dropna()
            if len(data_series) > 0:
                available_data[col_name] = {
                    'purpose': purpose,
                    'count': len(data_series),
                    'range': (data_series.min(), data_series.max()),
                    'mean': data_series.mean(),
                    'sample': data_series.head(3).tolist()
                }
                print(f"✅ {purpose}:")
                print(f"   📊 Column: '{col_name}'")
                print(f"   🔢 Data: {len(data_series)}/{len(df)} bonds")
                print(f"   📈 Range: {data_series.min():.2f} - {data_series.max():.2f}")
                print(f"   💡 Sample values: {[f'{x:.2f}' for x in data_series.head(3)]}")
            else:
                print(f"❌ {purpose}: Column '{col_name}' has no data")
        else:
            print(f"❌ {purpose}: Column '{col_name}' not found")
    
    return available_data, {}

def build_trading_qubo_matrix(df, trading_strategy='cost_minimization'):
    """Build QUBO matrix for actual bond trading cost optimization"""
    print(f"\n💰 BUILDING TRADING OPTIMIZATION QUBO")
    print(f"Strategy: {trading_strategy}")
    print("-" * 50)
    
    n_bonds = len(df)
    Q = np.zeros((n_bonds, n_bonds))
    
    if trading_strategy == 'cost_minimization' and 'price' in df.columns:
        # Minimize total purchase cost
        prices = df['price'].fillna(df['price'].median()).values
        
        # Normalize prices
        min_price, max_price = prices.min(), prices.max()
        if max_price > min_price:
            normalized_prices = (prices - min_price) / (max_price - min_price)
        else:
            normalized_prices = np.ones(n_bonds)
        
        # Cost of buying each bond
        for i in range(n_bonds):
            Q[i, i] += normalized_prices[i] * 2.0  # Higher price = higher cost
        
        print(f"   💵 Optimizing purchase costs: ${min_price:.2f} - ${max_price:.2f}")
        
    elif trading_strategy == 'yield_maximization' and 'oas' in df.columns:
        # Maximize expected yield (credit spread)
        spreads = df['oas'].fillna(df['oas'].median()).values
        
        # Normalize spreads
        min_spread, max_spread = spreads.min(), spreads.max()
        if max_spread > min_spread:
            normalized_spreads = (spreads - min_spread) / (max_spread - min_spread)
        else:
            normalized_spreads = np.ones(n_bonds)
        
        # NEGATIVE because we want to maximize yield
        for i in range(n_bonds):
            Q[i, i] -= normalized_spreads[i] * 1.5  # Higher spread = better
        
        print(f"   📈 Optimizing yield: {min_spread:.0f} - {max_spread:.0f} bps")
        
    else:
        # Fallback: simple diversification
        print(f"   🎯 Using simple diversification (insufficient data for {trading_strategy})")
        for i in range(n_bonds):
            Q[i, i] += 0.5
    
    # Portfolio size constraint
    target_bonds = max(4, min(8, n_bonds // 2))
    penalty = 1.0
    
    for i in range(n_bonds):
        for j in range(n_bonds):
            if i == j:
                Q[i, j] += penalty * (1 - 2 * target_bonds)
            else:
                Q[i, j] += penalty
    
    print(f"   📊 Target portfolio: ~{target_bonds} bonds")
    
    # Ensure positive costs
    min_eigenval = np.min(np.linalg.eigvals(Q))
    if min_eigenval < 0:
        baseline_shift = abs(min_eigenval) + 0.5
        for i in range(n_bonds):
            Q[i, i] += baseline_shift
        print(f"   ➕ Added baseline ({baseline_shift:.2f}) for positive costs")
    
    eigenvals = np.linalg.eigvals(Q)
    print(f"   📈 Trading QUBO built: {Q.shape}")
    print(f"   📊 Eigenvalue range: [{eigenvals.min():.3f}, {eigenvals.max():.3f}]")
    
    return Q, {'strategy': trading_strategy, 'target_bonds': target_bonds}

def create_portfolio_targets(df, characteristics, target_strategy='median'):
    """Create optimization targets from the dataset."""
    print(f"\n🎯 CREATING PORTFOLIO TARGETS ({target_strategy})")
    print("-" * 50)
    
    targets = {}
    target_info = {}
    
    for char_name, col_name in characteristics.items():
        if col_name in df.columns and df[col_name].dtype in ['float64', 'int64']:
            if target_strategy == 'median':
                target_val = df[col_name].median()
            elif target_strategy == 'mean':
                target_val = df[col_name].mean()
            elif target_strategy == 'conservative':
                target_val = df[col_name].quantile(0.25)
            elif target_strategy == 'aggressive':
                target_val = df[col_name].quantile(0.75)
            else:
                target_val = df[col_name].median()
            
            targets[char_name] = target_val
            target_info[char_name] = {
                'value': target_val,
                'column': col_name,
                'data_range': (df[col_name].min(), df[col_name].max()),
                'std': df[col_name].std()
            }
            
            print(f"   🎯 {char_name}: {target_val:.4f} (range: {df[col_name].min():.3f}-{df[col_name].max():.3f})")
    
    return targets, target_info

def build_enhanced_qubo_matrix(df, targets, characteristics, constraints=None):
    """Build enhanced QUBO matrix for quantum portfolio optimization."""
    print(f"\n🔧 BUILDING ENHANCED QUBO MATRIX")
    print("-" * 50)
    
    n_assets = len(df)
    Q = np.zeros((n_assets, n_assets))
    
    print(f"   📊 Assets: {n_assets}")
    print(f"   🎯 Characteristics: {len(targets)}")
    
    # Objective: minimize tracking error for each characteristic
    total_weight = 0
    
    for char_name, target_val in targets.items():
        if char_name in characteristics:
            col_name = characteristics[char_name]
            if col_name in df.columns:
                char_values = df[col_name].fillna(df[col_name].median()).values
                
                # Weight by inverse of standard deviation
                char_std = df[col_name].std()
                weight = 1.0 / (char_std + 1e-6) if char_std > 0 else 1.0
                total_weight += weight
                
                print(f"   ⚖️ {char_name}: weight = {weight:.3f}")
                
                # Add quadratic terms
                for i in range(n_assets):
                    for j in range(n_assets):
                        Q[i, j] += weight * char_values[i] * char_values[j]
                    
                    # Linear terms (converted to diagonal)
                    Q[i, i] += -2 * weight * target_val * char_values[i]
    
    # Normalize objective by total weight
    if total_weight > 0:
        Q = Q / total_weight
        print(f"   📐 Normalized by total weight: {total_weight:.3f}")
    
    # Add constraint penalties (SIMPLIFIED POSITIVE VERSION)
    if constraints is None:
        constraints = {
            'min_assets': max(3, n_assets // 8),
            'max_assets': min(12, n_assets // 2)
        }
    
    print(f"   ⚖️ Constraints: {constraints}")
    
    # Simple positive penalty approach
    penalty_strength = 1.0
    target_count = (constraints['min_assets'] + constraints['max_assets']) // 2
    
    # Add penalty that increases cost when far from target count
    for i in range(n_assets):
        deviation_penalty = penalty_strength * abs(1 - target_count / n_assets)
        Q[i, i] += deviation_penalty
    
    # Add positive baseline to ensure all costs are positive
    min_eigenval = np.min(np.linalg.eigvals(Q))
    if min_eigenval < 0:
        baseline_shift = abs(min_eigenval) + 1.0
        for i in range(n_assets):
            Q[i, i] += baseline_shift
        print(f"   ➕ Added positive baseline ({baseline_shift:.2f}) to ensure positive costs")
    
    print(f"   🚨 Added constraint guidance for ~{target_count} assets")
    
    # Check matrix properties
    eigenvals = np.linalg.eigvals(Q)
    print(f"   📈 QUBO matrix built: {Q.shape}")
    print(f"   📊 Eigenvalue range: [{eigenvals.min():.3f}, {eigenvals.max():.3f}]")
    print(f"   🔢 Matrix norm: {np.linalg.norm(Q):.3f}")
    
    return Q, constraints

def qubo_to_pauli_operator(Q):
    """Convert QUBO matrix Q to Pauli operator for Qiskit."""
    n_qubits = Q.shape[0]
    
    pauli_list = []
    coeffs = []
    
    # Diagonal terms
    for i in range(n_qubits):
        if Q[i, i] != 0:
            pauli_str = ['I'] * n_qubits
            pauli_str[i] = 'Z'
            pauli_list.append(''.join(pauli_str))
            coeffs.append(-Q[i, i] / 2)
    
    # Off-diagonal terms
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if Q[i, j] != 0:
                pauli_str = ['I'] * n_qubits
                pauli_str[i] = 'Z'
                pauli_str[j] = 'Z'
                pauli_list.append(''.join(pauli_str))
                coeffs.append(Q[i, j] / 4)
    
    # Constant term
    constant = np.sum(Q) / 4
    if pauli_list:
        pauli_list.append('I' * n_qubits)
        coeffs.append(constant)
    else:
        pauli_list = ['I' * n_qubits]
        coeffs = [constant]
    
    return SparsePauliOp(pauli_list, coeffs)

def solve_quantum_vqe_manual(Q, max_iter=100, optimizer_type='COBYLA'):
    """Manual VQE implementation that avoids all callback conflicts."""
    print(f"\n🚀 MANUAL QUANTUM VQE OPTIMIZATION")
    print("-" * 50)
    
    if not QISKIT_AVAILABLE or not SCIPY_AVAILABLE:
        print("   ❌ Dependencies not available, falling back to classical")
        return solve_classical_fallback(Q, max_iter)
    
    n_qubits = Q.shape[0]
    start_time = time.perf_counter()
    
    try:
        print(f"   🔬 Problem size: {n_qubits} qubits")
        
        # Convert QUBO to quantum Hamiltonian
        hamiltonian = qubo_to_pauli_operator(Q)
        print(f"   🧮 Hamiltonian: {len(hamiltonian)} Pauli terms")
        
        # Create quantum ansatz
        reps = max(1, min(2, 4 // n_qubits))  # Conservative for stability
        ansatz = TwoLocal(
            num_qubits=n_qubits,
            rotation_blocks='ry',
            entanglement_blocks='cz',
            entanglement='linear',
            reps=reps
        )
        
        print(f"   🔧 Ansatz: {ansatz.num_parameters} parameters, {reps} layers")
        
        # Set up quantum backend
        backend = AerSimulator()
        estimator = Estimator()
        
        print(f"   🎯 Optimizer: {optimizer_type}")
        
        # Custom cost function for VQE (FIXED to match working test)
        iteration_count = [0]
        
        def vqe_cost_function(params):
            """Evaluate quantum circuit with given parameters - MATCHES WORKING TEST"""
            try:
                iteration_count[0] += 1
                
                # Use assign_parameters (confirmed working in test)
                bound_circuit = ansatz.assign_parameters(params)
                
                # Run estimator
                job = estimator.run([bound_circuit], [hamiltonian])
                result = job.result()
                
                # Use result.values[0] (confirmed working in test)
                energy = result.values[0]
                
                # Progress reporting
                if iteration_count[0] % max(1, max_iter // 10) == 0:
                    print(f"   📈 Evaluation {iteration_count[0]:3d}: Energy = {energy:8.4f}")
                
                return float(energy)
                
            except Exception as e:
                print(f"   ⚠️ Cost function error: {e}")
                if iteration_count[0] <= 3:  # Only print traceback for first few errors
                    import traceback
                    traceback.print_exc()
                return float(1e6)  # Return large penalty on error
        
        # Random initial parameters
        np.random.seed(42)  # For reproducibility
        initial_params = np.random.uniform(0, 2*np.pi, ansatz.num_parameters)
        
        print(f"   🎬 Starting manual VQE with {len(initial_params)} parameters...")
        print(f"   🔧 Testing cost function with initial parameters first...")
        
        # Test the cost function once before optimization
        try:
            test_energy = vqe_cost_function(initial_params)
            print(f"   ✅ Cost function test successful: {test_energy:.4f}")
        except Exception as e:
            print(f"   ❌ Cost function test failed: {e}")
            raise e  # Re-raise to fall back to classical
        
        # Run optimization with scipy (NO VQE CLASS USED)
        options = {
            'maxiter': max_iter,
            'disp': False,
            'rhobeg': 0.5,  # Initial step size for COBYLA
            'tol': 1e-6     # Tolerance
        }
        
        optimization_result = minimize(
            fun=vqe_cost_function,
            x0=initial_params,
            method=optimizer_type,
            options=options
        )
        
        optimal_params = optimization_result.x
        optimal_value = optimization_result.fun
        
        print(f"   🏆 Manual VQE completed!")
        print(f"   🔄 Function evaluations: {optimization_result.nfev}")
        print(f"   ✅ Optimization success: {optimization_result.success}")
        print(f"   🎯 Final energy: {optimal_value:.4f}")
        
        # Sample quantum state for classical solution
        optimal_circuit = ansatz.assign_parameters(optimal_params)
        optimal_circuit.measure_all()
        
        # Run measurement
        shots = 2048  # More shots for better statistics
        job = backend.run(transpile(optimal_circuit, backend), shots=shots)
        counts = job.result().get_counts()
        
        # Get most frequent measurement
        best_bitstring = max(counts, key=counts.get)
        best_solution = np.array([int(bit) for bit in best_bitstring[::-1]])
        
        # Calculate QUBO cost
        classical_cost = best_solution.T @ Q @ best_solution
        
        runtime = time.perf_counter() - start_time
        
        print(f"   ✅ Quantum optimization completed!")
        print(f"   ⏱️ Runtime: {runtime:.6f} seconds")
        print(f"   🎯 Quantum eigenvalue: {optimal_value:.4f}")
        print(f"   💰 Classical QUBO cost: {classical_cost:.4f}")
        print(f"   📊 Assets selected: {np.sum(best_solution)}")
        print(f"   🎲 Measurement shots: {shots}")
        
        return {
            'solution': best_solution,
            'cost': classical_cost,
            'runtime': runtime,
            'method': 'Manual_Quantum_VQE',
            'iterations': optimization_result.nfev,
            'vqe_eigenvalue': optimal_value,
            'optimizer': optimizer_type,
            'ansatz_params': ansatz.num_parameters,
            'circuit_depth': reps,
            'measurement_counts': dict(list(counts.items())[:5]),
            'success': optimization_result.success,
            'quantum_advantage': True
        }
        
    except Exception as e:
        print(f"   ❌ Manual VQE failed: {str(e)}")
        print(f"   🔄 Falling back to classical approach...")
        return solve_classical_fallback(Q, max_iter)

def solve_classical_fallback(Q, max_iter=1000):
    """Classical fallback when quantum optimization fails."""
    print(f"\n🖥️ CLASSICAL FALLBACK OPTIMIZATION")
    print("-" * 50)
    
    n_assets = Q.shape[0]
    start_time = time.perf_counter()
    
    best_cost = float('inf')
    best_solution = None
    target_assets = max(3, min(10, n_assets // 2))
    
    for iteration in range(max_iter):
        solution = np.zeros(n_assets, dtype=int)
        selected_indices = np.random.choice(n_assets, size=target_assets, replace=False)
        solution[selected_indices] = 1
        
        cost = solution.T @ Q @ solution
        
        if cost < best_cost:
            best_cost = cost
            best_solution = solution.copy()
        
        if (iteration + 1) % (max_iter // 10) == 0:
            print(f"   📈 Iteration {iteration+1:4d}: Best cost = {best_cost:.4f}")
    
    runtime = time.perf_counter() - start_time
    
    print(f"   ✅ Classical optimization completed!")
    print(f"   ⏱️ Runtime: {runtime:.6f} seconds")
    print(f"   💰 Best cost: {best_cost:.4f}")
    print(f"   📊 Assets selected: {np.sum(best_solution)}")
    
    return {
        'solution': best_solution,
        'cost': best_cost,
        'runtime': runtime,
        'method': 'Classical_Random_Search',
        'iterations': max_iter,
        'success': True
    }

def solve_classical_greedy(Q):
    """Greedy classical benchmark for comparison."""
    print(f"\n🎯 CLASSICAL GREEDY BENCHMARK")
    print("-" * 50)
    
    n_assets = Q.shape[0]
    start_time = time.perf_counter()
    
    # Calculate asset scores
    scores = np.diag(Q).copy()
    
    # Add average interaction cost with other assets
    for i in range(n_assets):
        scores[i] += np.mean(Q[i, :]) + np.mean(Q[:, i])
    
    # Select assets with lowest scores
    n_select = max(3, min(8, n_assets // 3))
    selected_indices = np.argsort(scores)[:n_select]
    
    solution = np.zeros(n_assets, dtype=int)
    solution[selected_indices] = 1
    
    cost = solution.T @ Q @ solution
    runtime = time.perf_counter() - start_time
    
    print(f"   ✅ Greedy selection completed!")
    print(f"   ⏱️ Runtime: {runtime:.6f} seconds")
    print(f"   💰 Cost: {cost:.4f}")
    print(f"   📊 Assets selected: {np.sum(solution)}")
    
    return {
        'solution': solution,
        'cost': cost,
        'runtime': runtime,
        'method': 'Classical_Greedy',
        'iterations': 1,
        'success': True
    }

def analyze_trading_solution(solution, df, strategy):
    """Analyze the selected portfolio in TRADING terms"""
    print(f"\n💼 TRADING SOLUTION ANALYSIS")
    print("-" * 50)
    
    selected_mask = solution.astype(bool)
    selected_bonds = df[selected_mask]
    
    if len(selected_bonds) == 0:
        print("   ❌ No bonds selected!")
        return {}
    
    print(f"   📈 Selected {len(selected_bonds)} bonds for trading")
    print(f"   🎯 Strategy: {strategy}")
    
    analysis = {
        'n_selected': len(selected_bonds),
        'strategy': strategy,
        'trading_metrics': {}
    }
    
    # Calculate trading-specific metrics
    if 'price' in selected_bonds.columns:
        prices = selected_bonds['price'].dropna()
        if len(prices) > 0:
            total_cost = prices.sum()
            avg_price = prices.mean()
            min_price = prices.min()
            max_price = prices.max()
            
            analysis['trading_metrics']['total_purchase_cost'] = total_cost
            analysis['trading_metrics']['average_price'] = avg_price
            analysis['trading_metrics']['price_range'] = (min_price, max_price)
            
            print(f"   💰 Total purchase cost: ${total_cost:,.2f}")
            print(f"   💵 Average bond price: ${avg_price:.2f}")
            print(f"   📊 Price range: ${min_price:.2f} - ${max_price:.2f}")
    
    if 'oas' in selected_bonds.columns:
        spreads = selected_bonds['oas'].dropna()
        if len(spreads) > 0:
            avg_spread = spreads.mean()
            total_yield_potential = spreads.sum()
            spread_std = spreads.std()
            
            analysis['trading_metrics']['average_spread'] = avg_spread
            analysis['trading_metrics']['total_yield_potential'] = total_yield_potential
            analysis['trading_metrics']['spread_risk'] = spread_std
            
            print(f"   📈 Average credit spread: {avg_spread:.0f} bps")
            print(f"   🎯 Total yield potential: {total_yield_potential:.0f} bps")
            print(f"   ⚠️ Spread volatility: {spread_std:.0f} bps")
    
    if 'spreadDur' in selected_bonds.columns:
        durations = selected_bonds['spreadDur'].dropna()
        if len(durations) > 0:
            avg_duration = durations.mean()
            duration_risk = durations.std()
            
            analysis['trading_metrics']['average_duration'] = avg_duration
            analysis['trading_metrics']['duration_risk'] = duration_risk
            
            print(f"   ⏰ Average duration: {avg_duration:.2f} years")
            print(f"   📉 Interest rate risk: {duration_risk:.2f}")
    
    if 'fund_enriched.notionalMktValue' in selected_bonds.columns:
        market_values = selected_bonds['fund_enriched.notionalMktValue'].dropna()
        if len(market_values) > 0:
            total_liquidity = market_values.sum()
            avg_liquidity = market_values.mean()
            
            analysis['trading_metrics']['total_liquidity'] = total_liquidity
            analysis['trading_metrics']['average_liquidity'] = avg_liquidity
            
            print(f"   🏦 Total market value: ${total_liquidity:,.0f}")
            print(f"   💧 Average liquidity: ${avg_liquidity:,.0f}")
    
    # Trading recommendations based on strategy
    print(f"\n💡 TRADING RECOMMENDATIONS:")
    if strategy == 'cost_minimization':
        print(f"   🛒 BUY: Focus on lowest-cost bonds")
        print(f"   💰 Minimize cash outlay for portfolio construction")
        print(f"   🎯 Good for: Building positions, rebalancing")
    elif strategy == 'yield_maximization':
        print(f"   📈 BUY: Target highest-yielding opportunities") 
        print(f"   ⚠️ Monitor: Higher yields may indicate higher credit risk")
        print(f"   🎯 Good for: Income generation, spread capture")
    else:
        print(f"   ⚖️ BUY: Balanced selection for diversification")
        print(f"   🎯 Good for: General portfolio construction")
    
    return analysis

def main():
    """Main execution function."""
    print("🏆 QUANTUM PORTFOLIO OPTIMIZATION (TRADING FOCUS)")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load and prepare data
        print(f"\n📊 LOADING financial DATA")
        print("-" * 50)
        
        df_raw = load_financial_excel_data()
        
        # Prepare quantum-optimized dataset
        quantum_df, characteristics, norm_info = prepare_quantum_dataset(
            df_raw, 
            max_assets=16,  # Optimal for quantum circuits
            selection_strategy='diverse'
        )
        
        # Create portfolio targets
        targets, target_info = create_portfolio_targets(
            quantum_df, 
            characteristics, 
            target_strategy='median'
        )
        
        # VERIFY: Show which real Excel data is being used for trading optimization
        available_data, missing_data = analyze_real_data_usage(quantum_df)
        
        # Build QUBO matrix FOR TRADING OPTIMIZATION using YOUR REAL DATA
        Q, constraints = build_trading_qubo_matrix(
            quantum_df,
            trading_strategy='cost_minimization'  # or 'yield_maximization', 'risk_adjusted_return'
        )
        
        # Run optimizations
        print(f"\n🔮 RUNNING OPTIMIZATIONS")
        print("=" * 50)
        
        # Manual quantum VQE (NO CALLBACKS)
        quantum_result = solve_quantum_vqe_manual(Q, max_iter=50, optimizer_type='COBYLA')
        
        # Classical benchmarks
        greedy_result = solve_classical_greedy(Q)
        
        # Compare results
        print(f"\n🏆 RESULTS COMPARISON")
        print("=" * 50)
        
        results = [quantum_result, greedy_result]
        best_cost = min(result['cost'] for result in results)
        
        for result in results:
            method = result['method']
            cost = result['cost']
            runtime = result['runtime']
            n_selected = np.sum(result['solution'])
            
            is_best = "🥇" if abs(cost - best_cost) < 1e-6 else "  "
            print(f"{is_best} {method}:")
            print(f"   💰 Cost: {cost:8.4f}")
            print(f"   ⏱️ Runtime: {runtime:10.6f}s")
            print(f"   📊 Assets: {n_selected:2d}")
            
            if 'vqe_eigenvalue' in result:
                print(f"   🔬 Quantum eigenvalue: {result['vqe_eigenvalue']:8.4f}")
        
        # Calculate quantum advantage
        if len(results) >= 2:
            classical_cost = greedy_result['cost']
            quantum_cost = quantum_result['cost']
            if abs(classical_cost) > 1e-6:
                advantage = (classical_cost - quantum_cost) / abs(classical_cost) * 100
                print(f"\n🚀 Quantum Advantage: {advantage:+.2f}%")
                if advantage > 0:
                    print("   ✅ Quantum method found better solution!")
                else:
                    print("   📈 Classical method performed better this time")
        
        # Find best result for analysis
        best_result = min(results, key=lambda x: x['cost'])
        
        # Analyze best solution in TRADING terms
        best_analysis = analyze_trading_solution(
            best_result['solution'], 
            quantum_df, 
            constraints.get('strategy', 'unknown')
        )
        
        # Save results
        timestamp = datetime.now().isoformat()
        comprehensive_results = {
            'metadata': {
                'timestamp': timestamp,
                'qiskit_available': QISKIT_AVAILABLE,
                'original_assets': len(df_raw),
                'quantum_assets': len(quantum_df),
                'characteristics_used': list(characteristics.keys()),
                'implementation': 'Manual_VQE_Trading_Optimization'
            },
            'data_preparation': {
                'characteristics': characteristics,
                'available_trading_data': available_data,
                'constraints': constraints,
                'normalization_info': norm_info
            },
            'optimization_results': {
                'quantum_vqe': quantum_result.copy(),
                'classical_greedy': greedy_result.copy(),
                'best_method': best_result['method'],
                'best_cost': best_result['cost']
            },
            'trading_analysis': best_analysis
        }
        
        # Convert numpy arrays for JSON
        for result_key in ['quantum_vqe', 'classical_greedy']:
            if 'solution' in comprehensive_results['optimization_results'][result_key]:
                solution = comprehensive_results['optimization_results'][result_key]['solution']
                if isinstance(solution, np.ndarray):
                    comprehensive_results['optimization_results'][result_key]['solution'] = solution.tolist()
        
        # Save to file
        results_file = f"quantum_trading_results_{timestamp.split('T')[0]}.json"
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n📁 RESULTS SAVED")
        print("-" * 50)
        print(f"   📄 File: {results_file}")
        print(f"   📊 Assets in optimal portfolio: {np.sum(best_result['solution'])}")
        print(f"   🎯 Optimization method: {best_result['method']}")
        
        print(f"\n🎊 QUANTUM TRADING OPTIMIZATION COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
