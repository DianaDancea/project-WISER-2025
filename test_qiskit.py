#!/usr/bin/env python3
"""
Test script to verify Qiskit installation and basic functionality
"""

def test_qiskit_installation():
    """Test if Qiskit is properly installed and working."""
    print("🧪 TESTING QISKIT INSTALLATION")
    print("=" * 50)
    
    try:
        # Test basic Qiskit import
        import qiskit
        print(f"✅ Qiskit imported successfully")
        print(f"   Version: {qiskit.__version__}")
        
        # Test Qiskit Algorithms
        from qiskit_algorithms import VQE
        from qiskit_algorithms.optimizers import COBYLA
        print(f"✅ Qiskit Algorithms imported successfully")
        
        # Test Qiskit Aer
        from qiskit_aer import AerSimulator
        print(f"✅ Qiskit Aer imported successfully")
        
        # Test quantum circuit creation
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import TwoLocal
        
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        print(f"✅ Quantum circuit creation works")
        
        # Test ansatz
        ansatz = TwoLocal(2, 'ry', 'cz', reps=1)
        print(f"✅ TwoLocal ansatz creation works")
        print(f"   Parameters: {ansatz.num_parameters}")
        
        # Test simulator
        backend = AerSimulator()
        print(f"✅ AerSimulator works")
        print(f"   Backend: {backend.name}")
        
        # Test Pauli operators
        from qiskit.quantum_info import SparsePauliOp
        pauli_op = SparsePauliOp(['ZZ', 'XI'], [1.0, 0.5])
        print(f"✅ Pauli operators work")
        print(f"   Operator: {pauli_op}")
        
        # Test estimator
        from qiskit.primitives import Estimator
        estimator = Estimator()
        print(f"✅ Estimator works")
        
        print(f"\n🎉 ALL TESTS PASSED! Qiskit is ready for quantum optimization!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print(f"   Run: pip install qiskit qiskit-algorithms qiskit-aer")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_simple_vqe():
    """Test a simple VQE example to ensure everything works."""
    print(f"\n🔬 TESTING SIMPLE VQE EXAMPLE")
    print("-" * 50)
    
    try:
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import TwoLocal
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import Estimator
        from qiskit_algorithms import VQE
        from qiskit_algorithms.optimizers import COBYLA
        
        # Create simple Hamiltonian (H = Z_0 + Z_1)
        hamiltonian = SparsePauliOp(['ZI', 'IZ'], [1.0, 1.0])
        print(f"   🧮 Hamiltonian: {hamiltonian}")
        
        # Create ansatz
        ansatz = TwoLocal(2, 'ry', 'cz', reps=1)
        print(f"   🔧 Ansatz parameters: {ansatz.num_parameters}")
        
        # Set up VQE
        estimator = Estimator()
        optimizer = COBYLA(maxiter=20)
        vqe = VQE(estimator, ansatz, optimizer)
        
        print(f"   🎯 Running VQE...")
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        
        print(f"   ✅ VQE completed!")
        print(f"   🔬 Eigenvalue: {result.optimal_value:.4f}")
        print(f"   📊 Expected: -2.0000 (both qubits in |1⟩ state)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ VQE test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    install_ok = test_qiskit_installation()
    
    if install_ok:
        vqe_ok = test_simple_vqe()
        
        if vqe_ok:
            print(f"\n🚀 READY FOR QUANTUM PORTFOLIO OPTIMIZATION!")
        else:
            print(f"\n⚠️ Basic installation works, but VQE has issues")
    else:
        print(f"\n❌ Please install Qiskit properly before proceeding")
        print(f"\n💡 INSTALLATION GUIDE:")
        print(f"   1. pip install --upgrade pip")
        print(f"   2. pip install qiskit qiskit-algorithms qiskit-aer")
        print(f"   3. python test_qiskit.py")