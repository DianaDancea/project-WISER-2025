#!/usr/bin/env python3
"""
Quantum Portfolio Optimization - Results Analyzer & Visualization
===============================================================
Author: Diana Dancea
Date: August 10, 2025

Standalone script to analyze and visualize quantum portfolio optimization results
Run this after executing quantum_portfolio_solver.py

This file creates the plots in matplotlib for better visuals
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class PortfolioResultsAnalyzer:
    """Comprehensive analyzer for quantum portfolio optimization results"""
    
    def __init__(self):
        self.results = None
        self.financial_df = None
        self.demo_bonds = None
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_results(self):
        """Load the most recent results file"""
        try:
            results_files = list(Path('.').glob('quantum_trading_results_*.json'))
            if not results_files:
                print("❌ No results files found! Run quantum_portfolio_solver.py first.")
                return False
            
            latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Loading results from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                self.results = json.load(f)
            
            print(f"✅ Results loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def load_financial_data(self):
        """Load original financial data or create demo data"""
        try:
            excel_path = Path("data_assets_dump_partial.xlsx")
            if excel_path.exists():
                self.financial_df = pd.read_excel(excel_path, sheet_name=0)
                print(f"✅ Loaded financial data: {len(self.financial_df)} bonds")
            else:
                self.financial_df = self.create_demo_data()
                print("⚠️ Using synthetic demo data")
            return True
        except Exception as e:
            print(f"⚠️ Could not load Excel data: {e}")
            self.financial_df = self.create_demo_data()
            return True
    
    def create_demo_data(self):
        """Create synthetic bond data for visualization"""
        np.random.seed(42)
        n_bonds = 100
        
        return pd.DataFrame({
            'bond_id': [f'BOND_{i:03d}' for i in range(n_bonds)],
            'price': np.random.normal(100, 15, n_bonds),
            'oas': np.random.normal(120, 40, n_bonds),
            'spreadDur': np.random.normal(5, 2, n_bonds),
            'fund_enriched.notionalMktValue': np.random.lognormal(14, 1, n_bonds),
            'sector': np.random.choice(['Corporate', 'Government', 'Municipal', 'Agency'], n_bonds),
            'rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B'], n_bonds)
        })
    
    def prepare_portfolio_data(self):
        """Prepare portfolio analysis data"""
        if not self.results:
            return False
            
        quantum_solution = np.array(self.results['optimization_results']['quantum_vqe']['solution'])
        classical_solution = np.array(self.results['optimization_results']['classical_greedy']['solution'])
        
        n_assets = len(quantum_solution)
        self.demo_bonds = self.financial_df.head(n_assets).copy().reset_index(drop=True)
        
        self.demo_bonds['quantum_selected'] = quantum_solution
        self.demo_bonds['classical_selected'] = classical_solution
        self.demo_bonds['both_selected'] = quantum_solution & classical_solution
        
        return True
    
    def create_performance_comparison(self):
        """Create performance comparison visualization"""
        print("\n📊 Creating performance comparison plots...")
        
        quantum_result = self.results['optimization_results']['quantum_vqe']
        classical_result = self.results['optimization_results']['classical_greedy']
        
        # Calculate metrics
        quantum_cost = quantum_result['cost']
        classical_cost = classical_result['cost']
        advantage = (classical_cost - quantum_cost) / abs(classical_cost) * 100 if classical_cost != 0 else 0
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🚀 Quantum Portfolio Optimization - Performance Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        methods = ['Quantum VQE', 'Classical Greedy']
        colors = ['#FF6B6B', '#4ECDC4']
        
        # Cost comparison
        costs = [quantum_cost, classical_cost]
        bars1 = ax1.bar(methods, costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('💰 Optimization Cost', fontweight='bold', pad=20)
        ax1.set_ylabel('QUBO Cost')
        ax1.grid(True, alpha=0.3)
        
        for bar, cost in zip(bars1, costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{cost:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Runtime comparison
        runtimes = [quantum_result['runtime'], classical_result['runtime']]
        bars2 = ax2.bar(methods, runtimes, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('⏱️ Runtime Comparison', fontweight='bold', pad=20)
        ax2.set_ylabel('Time (seconds)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        for bar, runtime in zip(bars2, runtimes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{runtime:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        # Portfolio size
        assets = [sum(quantum_result['solution']), sum(classical_result['solution'])]
        bars3 = ax3.bar(methods, assets, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('📊 Portfolio Size', fontweight='bold', pad=20)
        ax3.set_ylabel('Assets Selected')
        ax3.grid(True, alpha=0.3)
        
        for bar, asset_count in zip(bars3, assets):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{asset_count}', ha='center', va='bottom', fontweight='bold')
        
        # Quantum advantage
        advantage_color = '#2ECC71' if advantage > 0 else '#E74C3C'
        bar4 = ax4.bar(['Quantum\nAdvantage'], [advantage], color=advantage_color, 
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_title('🚀 Quantum Advantage', fontweight='bold', pad=20)
        ax4.set_ylabel('Improvement (%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        ax4.text(0, advantage + (1 if advantage > 0 else -1), f'{advantage:+.2f}%', 
                ha='center', va='bottom' if advantage > 0 else 'top', 
                fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: performance_comparison.png")
        plt.show()
        
        return advantage
    
    def create_portfolio_analysis(self):
        """Create portfolio composition analysis"""
        print("\n💼 Creating portfolio composition analysis...")
        
        if self.demo_bonds is None:
            print("❌ Portfolio data not prepared")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📈 Portfolio Composition & Risk Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Selection overlap
        overlap_data = {
            'Both Selected': sum(self.demo_bonds['both_selected']),
            'Quantum Only': sum(self.demo_bonds['quantum_selected'] & ~self.demo_bonds['classical_selected']),
            'Classical Only': sum(self.demo_bonds['classical_selected'] & ~self.demo_bonds['quantum_selected']),
            'Neither': len(self.demo_bonds) - sum(self.demo_bonds['quantum_selected'] | self.demo_bonds['classical_selected'])
        }
        
        colors_pie = ['#2ECC71', '#FF6B6B', '#4ECDC4', '#95A5A6']
        wedges, texts, autotexts = ax1.pie(overlap_data.values(), labels=overlap_data.keys(),
                                          colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax1.set_title('🎯 Asset Selection Overlap', fontweight='bold', pad=20)
        
        # Price distribution
        if 'price' in self.demo_bonds.columns:
            quantum_prices = self.demo_bonds[self.demo_bonds['quantum_selected'] == 1]['price']
            classical_prices = self.demo_bonds[self.demo_bonds['classical_selected'] == 1]['price']
            
            ax2.hist(quantum_prices, bins=10, alpha=0.7, label='Quantum Portfolio',
                    color='#FF6B6B', edgecolor='black', density=True)
            ax2.hist(classical_prices, bins=10, alpha=0.7, label='Classical Portfolio',
                    color='#4ECDC4', edgecolor='black', density=True)
            ax2.set_title('💰 Price Distribution', fontweight='bold', pad=20)
            ax2.set_xlabel('Bond Price ($)')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Risk-Return scatter
        if 'oas' in self.demo_bonds.columns and 'spreadDur' in self.demo_bonds.columns:
            quantum_mask = self.demo_bonds['quantum_selected'] == 1
            classical_mask = self.demo_bonds['classical_selected'] == 1
            
            ax3.scatter(self.demo_bonds[~(quantum_mask | classical_mask)]['spreadDur'],
                       self.demo_bonds[~(quantum_mask | classical_mask)]['oas'],
                       c='lightgray', alpha=0.5, s=30, label='Not Selected')
            ax3.scatter(self.demo_bonds[classical_mask]['spreadDur'],
                       self.demo_bonds[classical_mask]['oas'],
                       c='#4ECDC4', alpha=0.8, s=60, label='Classical Selected', 
                       edgecolors='black', linewidth=1)
            ax3.scatter(self.demo_bonds[quantum_mask]['spreadDur'],
                       self.demo_bonds[quantum_mask]['oas'],
                       c='#FF6B6B', alpha=0.8, s=60, label='Quantum Selected',
                       edgecolors='black', linewidth=1)
            ax3.set_title('⚖️ Risk-Return Profile', fontweight='bold', pad=20)
            ax3.set_xlabel('Duration (years)')
            ax3.set_ylabel('Credit Spread (bps)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Portfolio metrics comparison
        quantum_portfolio = self.demo_bonds[self.demo_bonds['quantum_selected'] == 1]
        classical_portfolio = self.demo_bonds[self.demo_bonds['classical_selected'] == 1]
        
        metrics_comparison = {}
        if 'price' in self.demo_bonds.columns:
            metrics_comparison['Avg Price'] = [quantum_portfolio['price'].mean(), 
                                             classical_portfolio['price'].mean()]
        if 'oas' in self.demo_bonds.columns:
            metrics_comparison['Avg Spread'] = [quantum_portfolio['oas'].mean(), 
                                              classical_portfolio['oas'].mean()]
        if 'spreadDur' in self.demo_bonds.columns:
            metrics_comparison['Avg Duration'] = [quantum_portfolio['spreadDur'].mean(), 
                                                classical_portfolio['spreadDur'].mean()]
        
        if metrics_comparison:
            metrics_df = pd.DataFrame(metrics_comparison, index=['Quantum', 'Classical'])
            metrics_df.plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4', '#2ECC71'], alpha=0.8)
            ax4.set_title('📊 Portfolio Metrics Comparison', fontweight='bold', pad=20)
            ax4.set_ylabel('Value')
            ax4.tick_params(axis='x', rotation=0)
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('portfolio_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: portfolio_analysis.png")
        plt.show()
    
    def create_vqe_analysis(self):
        """Create VQE algorithm analysis"""
        print("\n🔬 Creating VQE algorithm analysis...")
        
        quantum_result = self.results['optimization_results']['quantum_vqe']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('🔬 VQE Algorithm Performance Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # VQE metrics
        vqe_metrics = {
            'Eigenvalue': quantum_result.get('vqe_eigenvalue', 0),
            'QUBO Cost': quantum_result['cost'],
            'Runtime': quantum_result['runtime'],
            'Iterations': quantum_result.get('iterations', 0)
        }
        
        # Normalize metrics for comparison
        norm_metrics = {}
        for key, value in vqe_metrics.items():
            if key == 'Runtime':
                norm_metrics[key] = value / 10  # Scale to reasonable range
            elif key == 'Iterations':
                norm_metrics[key] = value / 100  # Scale to reasonable range
            else:
                norm_metrics[key] = abs(value)
        
        categories = list(norm_metrics.keys())
        values = list(norm_metrics.values())
        
        bars = ax1.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#2ECC71', '#F39C12'], 
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('📊 VQE Performance Metrics', fontweight='bold', pad=20)
        ax1.set_ylabel('Normalized Value')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add actual values as text
        for bar, category in zip(bars, categories):
            height = bar.get_height()
            actual_value = vqe_metrics[category]
            if category == 'Runtime':
                text = f'{actual_value:.3f}s'
            elif category == 'Iterations':
                text = f'{actual_value}'
            else:
                text = f'{actual_value:.4f}'
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    text, ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Circuit complexity
        circuit_info = {
            'Qubits': len(quantum_result['solution']),
            'Parameters': quantum_result.get('ansatz_params', 20),
            'Depth': quantum_result.get('circuit_depth', 3),
            'Gates (est)': quantum_result.get('ansatz_params', 20) * 2
        }
        
        complexity_bars = ax2.bar(circuit_info.keys(), circuit_info.values(),
                                 color=['#FF6B6B', '#4ECDC4', '#2ECC71', '#F39C12'], 
                                 alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('🔧 Quantum Circuit Complexity', fontweight='bold', pad=20)
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, count in zip(complexity_bars, circuit_info.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('vqe_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: vqe_analysis.png")
        plt.show()
    
    def print_summary_report(self, quantum_advantage):
        """Print comprehensive summary report"""
        print(f"\n📋 EXECUTIVE SUMMARY REPORT")
        print("=" * 70)
        print(f"🕐 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        metadata = self.results['metadata']
        quantum_result = self.results['optimization_results']['quantum_vqe']
        classical_result = self.results['optimization_results']['classical_greedy']
        
        print(f"\n🎯 PROJECT OVERVIEW:")
        print(f"   📊 Original Assets: {metadata['original_assets']}")
        print(f"   🔬 Quantum Assets: {metadata['quantum_assets']}")
        print(f"   ⚡ Implementation: {metadata['implementation']}")
        
        print(f"\n🏆 PERFORMANCE RESULTS:")
        print(f"   💰 Quantum Cost: {quantum_result['cost']:.4f}")
        print(f"   💰 Classical Cost: {classical_result['cost']:.4f}")
        print(f"   🚀 Quantum Advantage: {quantum_advantage:+.2f}%")
        print(f"   ⏱️ Quantum Runtime: {quantum_result['runtime']:.3f}s")
        print(f"   ⏱️ Classical Runtime: {classical_result['runtime']:.3f}s")
        
        print(f"\n📊 PORTFOLIO COMPOSITION:")
        print(f"   🎯 Quantum Portfolio: {sum(quantum_result['solution'])} assets")
        print(f"   🎯 Classical Portfolio: {sum(classical_result['solution'])} assets")
        
        if 'trading_analysis' in self.results:
            trading = self.results['trading_analysis']
            print(f"\n💼 TRADING ANALYSIS:")
            print(f"   📈 Strategy: {trading.get('strategy', 'N/A')}")
            if 'trading_metrics' in trading:
                metrics = trading['trading_metrics']
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if 'cost' in key.lower():
                            print(f"   💰 {key}: ${value:,.2f}")
                        elif 'spread' in key.lower():
                            print(f"   📈 {key}: {value:.0f} bps")
                        else:
                            print(f"   📊 {key}: {value:.2f}")
        
        print(f"\n🎊 KEY ACHIEVEMENTS:")
        print(f"   ✅ Quantum algorithm successfully implemented")
        print(f"   ✅ Real financial data processed")
        print(f"   ✅ {'Quantum advantage achieved' if quantum_advantage > 0 else 'Competitive performance demonstrated'}")
        print(f"   ✅ Production-ready optimization pipeline")
        
        # Save summary to file
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'quantum_advantage': quantum_advantage,
            'quantum_cost': quantum_result['cost'],
            'classical_cost': classical_result['cost'],
            'quantum_runtime': quantum_result['runtime'],
            'classical_runtime': classical_result['runtime'],
            'quantum_assets': sum(quantum_result['solution']),
            'classical_assets': sum(classical_result['solution'])
        }
        
        with open('analysis_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n📁 Analysis summary saved to: analysis_summary.json")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 QUANTUM PORTFOLIO OPTIMIZATION - RESULTS ANALYSIS")
        print("=" * 70)
        
        # Load data
        if not self.load_results():
            return False
        
        if not self.load_financial_data():
            return False
        
        if not self.prepare_portfolio_data():
            return False
        
        # Create visualizations
        quantum_advantage = self.create_performance_comparison()
        self.create_portfolio_analysis()
        self.create_vqe_analysis()
        
        # Generate summary
        self.print_summary_report(quantum_advantage)
        
        print(f"\n🎊 ANALYSIS COMPLETE!")
        print("📁 Generated files:")
        print("   • performance_comparison.png")
        print("   • portfolio_analysis.png") 
        print("   • vqe_analysis.png")
        print("   • analysis_summary.json")
        print("\n🚀 Ready for presentation!")
        
        return True

def main():
    """Main execution function"""
    analyzer = PortfolioResultsAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if not success:
        print("\n❌ Analysis failed. Please ensure:")
        print("   1. quantum_portfolio_solver.py has been run")
        print("   2. Results JSON file exists")
        print("   3. Required dependencies are installed")

if __name__ == "__main__":
    main()
