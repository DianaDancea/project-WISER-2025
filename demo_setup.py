#!/usr/bin/env python3
"""
Demo Setup & Verification Script
===============================
Author: Diana Dancea
Date: August 10, 2025

Verifies all dependencies and runs a quick demo of the quantum portfolio optimization

This is a good starting point for checking that everything is good for running the files
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 CHECKING PYTHON VERSION")
    print("-" * 50)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version compatible")
        return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print(f"\n📦 CHECKING DEPENDENCIES")
    print("-" * 50)
    
    required_packages = {
        'numpy': 'numpy>=1.21.0',
        'pandas': 'pandas>=1.5.0', 
        'matplotlib': 'matplotlib>=3.5.0',
        'seaborn': 'seaborn>=0.11.0',
        'scipy': 'scipy>=1.9.0',
        'qiskit': 'qiskit>=0.45.0',
        'qiskit_aer': 'qiskit-aer>=0.12.0',
        'openpyxl': 'openpyxl>=3.0.0',
        'plotly': 'plotly>=5.10.0'
    }
    
    missing_packages = []
    
    for package, requirement in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Not installed")
            missing_packages.append(requirement)
    
    if missing_packages:
        print(f"\n📥 INSTALL MISSING PACKAGES:")
        print("pip install " + " ".join(missing_packages))
        return False
    else:
        print(f"\n✅ All dependencies satisfied!")
        return True

def check_project_files():
    """Check if all project files are present"""
    print(f"\n📁 CHECKING PROJECT FILES")
    print("-" * 50)
    
    required_files = {
        'quantum_portfolio_optimizer.py': 'Main optimization script',
        'vanguard_excel_loader.py': 'Data loading module',
        'requirements.txt': 'Dependencies list'
    }
    
    optional_files = {
        'data_assets_dump_partial.xlsx': 'Vanguard bond data (will use demo data if missing)',
        'mathematical_formulation.md': 'Mathematical documentation'
    }
    
    all_present = True
    
    for filename, description in required_files.items():
        if Path(filename).exists():
            print(f"✅ {filename}: {description}")
        else:
            print(f"❌ {filename}: {description} - REQUIRED")
            all_present = False
    
    for filename, description in optional_files.items():
        if Path(filename).exists():
            print(f"✅ {filename}: {description}")
        else:
            print(f"⚠️ {filename}: {description} - OPTIONAL")
    
    return all_present

def run_qiskit_test():
    """Test Qiskit functionality"""
    print(f"\n🔬 TESTING QISKIT FUNCTIONALITY")
    print("-" * 50)
    
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import TwoLocal
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import Estimator
        from qiskit_aer import AerSimulator
        import numpy as np
        
        # Test basic circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        print("✅ Basic quantum circuit creation")
        
        # Test ansatz
        ansatz = TwoLocal(2, 'ry', 'cz', reps=1)
        print(f"✅ TwoLocal ansatz ({ansatz.num_parameters} parameters)")
        
        # Test Pauli operators
        pauli_op = SparsePauliOp(['ZI', 'IZ'], [1.0, 1.0])
        print("✅ Pauli operator creation")
        
        # Test estimator
        estimator = Estimator()
        print("✅ Estimator initialization")
        
        # Test simulator
        backend = AerSimulator()
        print("✅ AerSimulator backend")
        
        # Quick VQE test
        params = np.random.random(ansatz.num_parameters) * 2 * np.pi
        bound_circuit = ansatz.assign_parameters(params)
        job = estimator.run([bound_circuit], [pauli_op])
        result = job.result()
        energy = result.values[0]
        print(f"✅ VQE test run (energy: {energy:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Qiskit test failed: {e}")
        return False

def run_quick_demo():
    """Run a quick demonstration"""
    print(f"\n🚀 RUNNING QUICK DEMO")
    print("-" * 50)
    
    try:
        # Import main modules
        from quantum_portfolio_optimizer import (
            solve_quantum_vqe_manual,
            build_trading_qubo_matrix,
            solve_classical_greedy
        )
        from vanguard_excel_loader import prepare_quantum_dataset, load_vanguard_excel_data
        import numpy as np
        import pandas as pd
        
        print("✅ Imported main modules")
        
        # Create demo data
        np.random.seed(42)
        demo_data = pd.DataFrame({
            'price': np.random.normal(100, 10, 8),
            'oas': np.random.normal(100, 20, 8),
            'spreadDur': np.random.normal(5, 1, 8)
        })
        
        print("✅ Created demo dataset (8 bonds)")
        
        # Build QUBO matrix
        Q, constraints = build_trading_qubo_matrix(demo_data, 'cost_minimization')
        print(f"✅ Built QUBO matrix ({Q.shape})")
        
        # Run quantum optimization
        print("🔮 Running mini quantum optimization...")
        quantum_result = solve_quantum_vqe_manual(Q, max_iter=10, optimizer_type='COBYLA')
        
        # Run classical benchmark
        classical_result = solve_classical_greedy(Q)
        
        # Compare results
        print(f"\n🏆 DEMO RESULTS:")
        print(f"   Quantum cost: {quantum_result['cost']:.4f}")
        print(f"   Classical cost: {classical_result['cost']:.4f}")
        print(f"   Quantum selected: {sum(quantum_result['solution'])} bonds")
        print(f"   Classical selected: {sum(classical_result['solution'])} bonds")
        
        if quantum_result['cost'] < classical_result['cost']:
            advantage = (classical_result['cost'] - quantum_result['cost']) / classical_result['cost'] * 100
            print(f"   🚀 Quantum advantage: +{advantage:.2f}%")
        else:
            print(f"   📊 Classical performed better in this demo")
        
        print("✅ Quick demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_demo_notebook():
    """Create or verify demo notebook exists"""
    print(f"\n📓 DEMO NOTEBOOK STATUS")
    print("-" * 50)
    
    notebook_files = [
        'portfolio_demo.ipynb',
        'demo_analysis.ipynb', 
        'quantum_portfolio_demo.ipynb'
    ]
    
    existing_notebooks = [f for f in notebook_files if Path(f).exists()]
    
    if existing_notebooks:
        print(f"✅ Found existing notebooks: {existing_notebooks}")
    else:
        print("ℹ️ No demo notebook found")
        print("📝 Use the provided notebook code to create one")
    
    return True

def main():
    """Main setup verification"""
    print("🚀 QUANTUM PORTFOLIO OPTIMIZATION - DEMO SETUP")
    print("=" * 70)
    print("Verifying environment and dependencies for demo presentation")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies), 
        ("Project Files", check_project_files),
        ("Qiskit Functionality", run_qiskit_test),
        ("Demo Notebook", create_demo_notebook),
        ("Quick Demo", run_quick_demo)
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results[check_name] = False
    
    print(f"\n📋 SETUP SUMMARY")
    print("=" * 50)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n🎊 SETUP COMPLETE!")
        print("✅ All checks passed - ready for demo!")
        print("\n🚀 NEXT STEPS:")
        print("1. Run: python quantum_portfolio_solver.py")
        print("2. Run: python results_analyzer.py")
        print("3. Present your results! 🎯")
    else:
        print(f"\n⚠️ SETUP INCOMPLETE")
        failed_checks = [name for name, passed in results.items() if not passed]
        print(f"❌ Failed checks: {', '.join(failed_checks)}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure all project files are present")
        print("3. Check Python version (3.8+ required)")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)