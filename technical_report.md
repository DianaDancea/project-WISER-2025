# Mathematical Formulation: Quantum Portfolio Optimization for Bond Trading

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Binary Decision Variables](#binary-decision-variables)
3. [Objective Functions](#objective-functions)
4. [Constraints](#constraints)
5. [QUBO Formulation](#qubo-formulation)
6. [Quantum Hamiltonian](#quantum-hamiltonian)
7. [Variational Quantum Eigensolver (VQE)](#variational-quantum-eigensolver-vqe)
8. [Trading Strategy Implementations](#trading-strategy-implementations)

---

## Problem Overview

The quantum portfolio optimization problem addresses the challenge of selecting an optimal subset of bonds from a universe of available securities to minimize trading costs, maximize expected returns, or optimize risk-adjusted performance. This formulation is specifically designed for Vanguard's bond trading scenarios using real market data.

**Core Challenge**: Given a universe of `n` bonds with characteristics (price, credit spread, duration, liquidity), select a portfolio of `k` bonds that optimizes a specific trading objective while satisfying practical constraints.

---

## Binary Decision Variables

### Definition
We define binary decision variables for bond selection:

```
x_i ∈ {0, 1} for i = 1, 2, ..., n
```

Where:
- `x_i = 1` if bond `i` is selected for the portfolio (BUY decision)
- `x_i = 0` if bond `i` is not selected (NO BUY decision)
- `n` is the total number of available bonds

### Portfolio Vector
The complete portfolio selection is represented as:
```
x = [x_1, x_2, x_3, ..., x_n]^T
```

### Example
For a universe of 5 bonds, the solution `x = [1, 0, 1, 1, 0]` means:
- Buy bonds 1, 3, and 4
- Do not buy bonds 2 and 5
- Portfolio size: 3 bonds

---

## Objective Functions

### 1. Cost Minimization Strategy

**Objective**: Minimize total purchase cost of the selected portfolio.

```
minimize: Σ(i=1 to n) c_i × x_i
```

Where `c_i` is the normalized price of bond `i`.

**Quadratic Form**:
```
minimize: x^T Q_cost x
```

Where `Q_cost` is constructed as:
```
Q_cost[i,i] = normalized_price_i
Q_cost[i,j] = 0 for i ≠ j (no interaction terms for pure cost minimization)
```

### 2. Yield Maximization Strategy

**Objective**: Maximize expected yield from credit spreads.

```
maximize: Σ(i=1 to n) y_i × x_i
```

Equivalently (for minimization):
```
minimize: -Σ(i=1 to n) y_i × x_i
```

Where `y_i` is the normalized credit spread (OAS) of bond `i`.

**Quadratic Form**:
```
minimize: x^T Q_yield x
```

Where:
```
Q_yield[i,i] = -normalized_spread_i  (negative for maximization)
Q_yield[i,j] = 0 for i ≠ j
```

### 3. Risk-Adjusted Return Strategy

**Objective**: Maximize return per unit of risk (Sharpe ratio concept).

```
maximize: Σ(i=1 to n) (y_i / σ_i) × x_i
```

Where:
- `y_i` is the expected return (credit spread) of bond `i`
- `σ_i` is the risk measure (duration-based) of bond `i`

**Quadratic Form with Diversification**:
```
minimize: x^T Q_risk x
```

Where:
```
Q_risk[i,i] = -risk_adjusted_score_i
Q_risk[i,j] = -diversification_bonus × similarity(i,j) for i ≠ j
```

---

## Constraints

### 1. Portfolio Size Constraint

**Linear Constraint**:
```
k_min ≤ Σ(i=1 to n) x_i ≤ k_max
```

Typically: `k_min = 3`, `k_max = 12` for practical bond portfolios.

### 2. Budget Constraint (if applicable)

```
Σ(i=1 to n) c_i × x_i ≤ B
```

Where `B` is the available budget.

### 3. Risk Exposure Constraints

```
Σ(i=1 to n) duration_i × x_i ≤ D_max  (duration limit)
Σ(i=1 to n) credit_risk_i × x_i ≤ R_max  (credit risk limit)
```

---

## QUBO Formulation

### Constraint Penalty Method

Since quantum computers excel at solving unconstrained problems, we convert constraints into penalty terms added to the objective function.

**General QUBO Form**:
```
minimize: x^T Q x

Where Q = Q_objective + λ₁Q_constraint1 + λ₂Q_constraint2 + ...
```

### Portfolio Size Constraint Penalty

For target portfolio size `k_target`, the penalty term is:
```
P_size = λ_size × (Σx_i - k_target)²
```

**Expanded Quadratic Form**:
```
P_size = λ_size × [Σx_i² - 2k_target×Σx_i + k_target²]
```

**QUBO Matrix Contribution**:
```
Q_size[i,i] += λ_size × (1 - 2k_target)  for all i
Q_size[i,j] += λ_size  for all i ≠ j
```

### Complete QUBO Matrix Construction

```
Q[i,i] = objective_diagonal[i] + constraint_penalties_diagonal[i]
Q[i,j] = objective_interaction[i,j] + constraint_penalties_interaction[i,j]  for i ≠ j
```

**Implementation Example** (Cost Minimization):
```python
# Objective terms
for i in range(n):
    Q[i,i] += normalized_prices[i]

# Portfolio size constraint
penalty = 1.0
target_bonds = 5
for i in range(n):
    for j in range(n):
        if i == j:
            Q[i,j] += penalty * (1 - 2 * target_bonds)
        else:
            Q[i,j] += penalty
```

---

## Quantum Hamiltonian

### QUBO to Ising Transformation

The QUBO problem is transformed into an Ising Hamiltonian for quantum computation:

```
H = Σᵢⱼ Jᵢⱼ σᵢᶻ σⱼᶻ + Σᵢ hᵢ σᵢᶻ + constant
```

Where:
- `σᵢᶻ` are Pauli-Z operators
- `Jᵢⱼ` are coupling strengths
- `hᵢ` are local magnetic fields

### Pauli Operator Representation

The Hamiltonian is expressed as a sum of Pauli operators:

```
H = Σₖ αₖ Pₖ
```

Where `Pₖ` are Pauli strings (e.g., "ZZII", "ZIZI") and `αₖ` are coefficients.

**Implementation in Code**:
```python
def qubo_to_pauli_operator(Q):
    pauli_list = []
    coeffs = []
    
    # Diagonal terms: σᵢᶻ
    for i in range(n_qubits):
        if Q[i,i] != 0:
            pauli_str = ['I'] * n_qubits
            pauli_str[i] = 'Z'
            pauli_list.append(''.join(pauli_str))
            coeffs.append(-Q[i,i] / 2)
    
    # Off-diagonal terms: σᵢᶻσⱼᶻ
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            if Q[i,j] != 0:
                pauli_str = ['I'] * n_qubits
                pauli_str[i] = 'Z'
                pauli_str[j] = 'Z'
                pauli_list.append(''.join(pauli_str))
                coeffs.append(Q[i,j] / 4)
    
    return SparsePauliOp(pauli_list, coeffs)
```

---

## Variational Quantum Eigensolver (VQE)

### Ansatz Circuit

We use a parameterized quantum circuit as the ansatz:

```
|ψ(θ)⟩ = U(θ)|0⟩ⁿ
```

**TwoLocal Ansatz Structure**:
- **Rotation blocks**: RY gates with parameters θᵢ
- **Entanglement blocks**: CZ gates creating quantum correlations
- **Layers**: Repeated application for expressibility

### Cost Function

The VQE cost function to minimize:

```
C(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
```

**Implementation**:
```python
def vqe_cost_function(params):
    bound_circuit = ansatz.assign_parameters(params)
    job = estimator.run([bound_circuit], [hamiltonian])
    energy = job.result().values[0]
    return float(energy)
```

### Classical Optimization

The parameter optimization uses classical algorithms:

```
θ* = argmin_θ ⟨ψ(θ)|H|ψ(θ)⟩
```

**Optimization Methods**:
- COBYLA (Constrained Optimization by Linear Approximation)
- SLSQP (Sequential Least Squares Programming)
- Powell's method

### Measurement and Solution Extraction

After optimization, measure the quantum state:

```python
optimal_circuit = ansatz.assign_parameters(optimal_params)
optimal_circuit.measure_all()
job = backend.run(optimal_circuit, shots=2048)
counts = job.result().get_counts()

# Extract most probable bitstring
best_bitstring = max(counts, key=counts.get)
solution = [int(bit) for bit in best_bitstring[::-1]]
```

---

## Trading Strategy Implementations

### Cost Minimization Matrix

```
Q_cost[i,i] = (price_i - price_min) / (price_max - price_min)
Q_cost[i,j] = λ_constraint  [for portfolio size constraint]
```

### Yield Maximization Matrix

```
Q_yield[i,i] = -(spread_i - spread_min) / (spread_max - spread_min)
Q_yield[i,j] = λ_constraint + λ_diversification × correlation(i,j)
```

### Risk-Adjusted Matrix

```
risk_adjusted_score_i = expected_return_i / risk_i
Q_risk[i,i] = -normalized(risk_adjusted_score_i)
Q_risk[i,j] = λ_constraint - λ_diversification × abs(profile_i - profile_j)
```

### Penalty Parameter Selection

**Guidelines for penalty strengths**:
- `λ_constraint = 1.0` (baseline constraint enforcement)
- `λ_diversification = 0.05` (encourage diversity)
- `λ_liquidity = 0.1` (penalize illiquid bonds)

### Matrix Properties

**Ensuring Positive Semi-Definiteness**:
```python
min_eigenval = np.min(np.linalg.eigvals(Q))
if min_eigenval < 0:
    baseline_shift = abs(min_eigenval) + 0.5
    for i in range(n):
        Q[i,i] += baseline_shift
```

---

## Performance Metrics

### Solution Quality
- **QUBO Cost**: `x^T Q x`
- **Trading Metrics**: Total cost, expected yield, risk measures
- **Constraint Violation**: Deviation from target portfolio size

### Quantum Performance
- **Convergence Rate**: Energy reduction per VQE iteration
- **Circuit Depth**: Number of ansatz layers required
- **Measurement Fidelity**: Consistency of quantum measurements

### Classical Comparison
- **Quantum Advantage**: `(Classical_Cost - Quantum_Cost) / |Classical_Cost|`
- **Runtime Efficiency**: Quantum vs classical optimization time
- **Scalability**: Performance vs problem size

---

This mathematical formulation provides the complete theoretical foundation for the quantum portfolio optimization implementation, ensuring that the quantum algorithm correctly solves the bond trading problem while maintaining interpretability and practical relevance for Vanguard's investment processes.