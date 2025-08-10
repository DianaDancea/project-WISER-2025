#!/usr/bin/env python3
"""
Simple test to verify VQE functionality with fixed APIs
"""

import numpy as np
import time

def test_simple_vqe():
    """Test basic VQE with correct API calls"""
    print("🧪 TESTING SIMPLE VQE WITH FIXED APIS")
    print("=" * 50)
    
    try:
        from qiskit.circuit.library import TwoLocal
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import Estimator
        from scipy.optimize import minimize
        
        print("✅ All imports successful")
        
        # Create simple 2-qubit problem
        hamiltonian = SparsePauliOp(['ZI', 'IZ'], [1.0, 1.0])
        ansatz = TwoLocal(2, 'ry', 'cz', reps=1)
        estimator = Estimator()
        
        print(f"✅ Created 2-qubit problem")
        print(f"   Hamiltonian: {hamiltonian}")
        print(f"   Ansatz parameters: {ansatz.num_parameters}")
        
        # Define cost function with robust result handling
        iteration_count = [0]
        
        def cost_function(params):
            iteration_count[0] += 1
            
            # Bind parameters
            bound_circuit = ansatz.assign_parameters(params)
            
            # Run estimator
            job = estimator.run([bound_circuit], [hamiltonian])
            result = job.result()
            
            # Handle different result APIs
            try:
                # Try new API
                energy = result[0].data.evs[0]
                api_used = "new API (result[0].data.evs[0])"
            except (TypeError, AttributeError, IndexError):
                try:
                    # Try alternative
                    energy = result.values[0]
                    api_used = "values API (result.values[0])"
                except (TypeError, AttributeError):
                    try:
                        # Try direct
                        energy = float(result[0])
                        api_used = "direct API (result[0])"
                    except:
                        print(f"❌ Cannot access result. Type: {type(result)}")
                        if hasattr(result, '__dict__'):
                            print(f"   Attributes: {list(result.__dict__.keys())}")
                        if hasattr(result, '__dir__'):
                            methods = [m for m in dir(result) if not m.startswith('_')]
                            print(f"   Methods: {methods}")
                        raise ValueError("Cannot access estimator result")
            
            if iteration_count[0] == 1:
                print(f"✅ Successfully accessed energy using: {api_used}")
            
            return float(energy)
        
        # Test cost function
        print("🔧 Testing cost function...")
        test_params = np.random.random(ansatz.num_parameters) * 2 * np.pi
        start_time = time.perf_counter()
        test_energy = cost_function(test_params)
        test_time = time.perf_counter() - start_time
        
        print(f"✅ Cost function works!")
        print(f"   Test energy: {test_energy:.4f}")
        print(f"   Evaluation time: {test_time:.6f} seconds")
        
        # Run quick optimization
        print("🚀 Running mini VQE optimization...")
        start_time = time.perf_counter()
        
        result = minimize(
            cost_function,
            test_params,
            method='COBYLA',
            options={'maxiter': 10, 'disp': False}
        )
        
        optimization_time = time.perf_counter() - start_time
        
        print(f"✅ VQE optimization completed!")
        print(f"   Initial energy: {test_energy:.4f}")
        print(f"   Final energy: {result.fun:.4f}")
        print(f"   Improvement: {test_energy - result.fun:.4f}")
        print(f"   Function evaluations: {result.nfev}")
        print(f"   Total time: {optimization_time:.6f} seconds")
        print(f"   Success: {result.success}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_vqe()
    
    if success:
        print(f"\n🎉 VQE TEST PASSED!")
        print(f"   The quantum portfolio optimization should now work!")
        print(f"   Run: python quantum_portfolio_optimizer.py")
    else:
        print(f"\n❌ VQE test failed. Need to investigate further.")