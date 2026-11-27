import math
import unittest
from algorithm import (
    solve_weights_and_energy, find_beta_star,
    compute_free_energy, compute_default_prob,
    find_rate_function_I, binomial_tail_probability,
    predict_systemic_crash, solve_tilted_weights
)


class TestGibbsWeights(unittest.TestCase):
    
    def test_weights_sum_to_one(self):
        """TEST 1: Weights must sum to 1 (probability constraint)"""
        beta = 1.0
        L = [0.5, 0.2, 0.3]
        W = [
            [0.0, 0.1, 0.05],
            [0.1, 0.0, 0.08],
            [0.05, 0.08, 0.0]
        ]
        
        E, w = solve_weights_and_energy(beta, L, W)
        weight_sum = sum(w)
        
        self.assertAlmostEqual(weight_sum, 1.0, places=10)
        print(f"[PASS] TEST 1: sum(w) = {weight_sum}")
    
    
    def test_weights_non_negative(self):
        """TEST 2: All weights must be non-negative (probability property)"""
        test_cases = [
            (0.01, [0.5, 0.2, 0.3], "High temperature"),
            (1.0,  [0.5, 0.2, 0.3], "Normal temperature"),
            (10.0, [0.5, 0.2, 0.3], "Low temperature"),
            (1.0,  [0.0, 100.0, 50.0], "Extreme energy differences"),
            (0.5,  [-10.0, 0.0, 10.0], "Negative energies"),
        ]
        
        W = [
            [0.0, 0.1, 0.05],
            [0.1, 0.0, 0.08],
            [0.05, 0.08, 0.0]
        ]
        
        for beta, L, description in test_cases:
            with self.subTest(beta=beta, description=description):
                E, w = solve_weights_and_energy(beta, L, W)
                
                for i, weight in enumerate(w):
                    self.assertGreaterEqual(weight, 0.0)
                    self.assertTrue(math.isfinite(weight))
                    if weight > 0:
                        self.assertGreater(weight, 1e-300)
        
        print("[PASS] TEST 2: All weights non-negative across temperature regimes")
    
    
    def test_weights_prefer_low_energy(self):
        """TEST 3: Lower energy states receive higher weight"""
        L = [0.1, 0.5, 1.0]
        W = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        
        beta = 1.0
        E, w = solve_weights_and_energy(beta, L, W)
        
        self.assertGreater(w[0], w[1], msg="Low energy state should have higher weight")
        self.assertGreater(w[1], w[2], msg="Medium energy should beat high energy")
        
        ratio_01_theory = math.exp(-beta * (L[0] - L[1]))
        ratio_01_actual = w[0] / w[1]
        self.assertAlmostEqual(ratio_01_actual, ratio_01_theory, places=5)
        
        print(f"[PASS] TEST 3: Weights prefer low energy (w={[round(x,4) for x in w]})")
    
    
    def test_expected_energy_computation(self):
        """TEST 4: Expected energy E computed as weighted average of Hamiltonians"""
        S = 3
        beta = 1.0
        L = [0.1, 0.5, 0.3]
        W = [
            [0.0, 0.05, 0.03],
            [0.05, 0.0, 0.04],
            [0.03, 0.04, 0.0]
        ]
        
        E, w = solve_weights_and_energy(beta, L, W)
        
        H_values = []
        for s in range(S):
            h_s = L[s] + sum(W[s][r] * w[r] for r in range(S))
            H_values.append(h_s)
        
        min_H = min(H_values)
        max_H = max(H_values)
        
        self.assertGreaterEqual(E, min_H - 1e-6, msg="E below minimum Hamiltonian")
        self.assertLessEqual(E, max_H + 1e-6, msg="E above maximum Hamiltonian")
        
        self.assertAlmostEqual(sum(w), 1.0, places=10)
        
        print(f"[PASS] TEST 4: Expected energy E = {E:.6f} (bounded correctly)")
    
    
    def test_monotonic_weight_ordering(self):
        """TEST 5: Weight ordering matches energy ordering (no interactions case)"""
        L = [0.1, 0.5, 1.0]
        W = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        
        for beta in [0.1, 1.0, 10.0]:
            E, w = solve_weights_and_energy(beta, L, W)
            
            self.assertGreater(w[0], w[1])
            self.assertGreater(w[1], w[2])
            
            if beta > 1.0:
                gap_01 = w[0] - w[1]
                gap_12 = w[1] - w[2]
                self.assertGreater(gap_01, 0)
                self.assertGreater(gap_12, 0)
        
        print("[PASS] TEST 5: Weight ordering consistent with energy across temperatures")
    
    
    def test_weight_persistence_across_runs(self):
        """TEST 6: Same inputs produce same outputs (determinism check)"""
        L = [0.3, 0.6, 0.2, 0.5]
        W = [
            [0.0, 0.05, 0.02, 0.03],
            [0.05, 0.0, 0.04, 0.02],
            [0.02, 0.04, 0.0, 0.05],
            [0.03, 0.02, 0.05, 0.0]
        ]
        beta = 1.5
        
        E1, w1 = solve_weights_and_energy(beta, L, W)
        E2, w2 = solve_weights_and_energy(beta, L, W)
        
        self.assertAlmostEqual(E1, E2, places=15)
        for i in range(len(w1)):
            self.assertAlmostEqual(w1[i], w2[i], places=15)
        
        self.assertAlmostEqual(sum(w1), 1.0, places=10)
        self.assertAlmostEqual(sum(w2), 1.0, places=10)
        
        print(f"[PASS] TEST 6: Deterministic results (w={[round(x,4) for x in w1]})")


class TestBisectionSolver(unittest.TestCase):
    
    def test_energy_monotonicity_in_beta(self):
        """TEST 7: Energy E(beta) increases monotonically with beta"""
        L = [0.5, 0.2, 0.8, 0.3]
        W = [
            [0.0, 0.05, 0.02, 0.03],
            [0.05, 0.0, 0.04, 0.02],
            [0.02, 0.04, 0.0, 0.05],
            [0.03, 0.02, 0.05, 0.0]
        ]
        
        betas = [0.01, 0.1, 1.0, 10.0, 100.0]
        energies = []
        
        for beta in betas:
            E, w = solve_weights_and_energy(beta, L, W)
            energies.append(E)
        
        for i in range(len(energies) - 1):
            self.assertLess(energies[i], energies[i+1] + 1e-10)
        
        print(f"[PASS] TEST 7: Energy monotonicity (E={[round(e,4) for e in energies]})")
    
    
    def test_bisection_convergence(self):
        """TEST 8: Bisection finds beta* such that E(beta*) ≈ H_real"""
        L = [0.5, 0.2, 0.8, 0.3, 0.6]
        W = [
            [0.0, 0.05, 0.02, 0.03, 0.01],
            [0.05, 0.0, 0.04, 0.02, 0.03],
            [0.02, 0.04, 0.0, 0.05, 0.02],
            [0.03, 0.02, 0.05, 0.0, 0.04],
            [0.01, 0.03, 0.02, 0.04, 0.0]
        ]
        
        H_real = 0.45
        
        beta_star, w_star = find_beta_star(H_real, L, W, tolerance=1e-6)
        E_check, _ = solve_weights_and_energy(beta_star, L, W)
        
        self.assertAlmostEqual(E_check, H_real, places=6)
        self.assertGreater(beta_star, 0.0)
        self.assertAlmostEqual(sum(w_star), 1.0, places=10)
        
        print(f"[PASS] TEST 8: Bisection convergence (beta*={beta_star:.6f}, E={E_check:.6f})")
    
    
    def test_bisection_bracketing_property(self):
        """TEST 9: Bisection properly brackets solution with monotonic E(beta)"""
        L = [0.1, 0.4, 0.7, 0.3]
        W = [
            [0.0, 0.03, 0.02, 0.01],
            [0.03, 0.0, 0.04, 0.02],
            [0.02, 0.04, 0.0, 0.03],
            [0.01, 0.02, 0.03, 0.0]
        ]
        
        E_low, _ = solve_weights_and_energy(1e-6, L, W)
        E_high, _ = solve_weights_and_energy(100.0, L, W)
        
        H_target = 0.5 * (E_low + E_high)
        
        beta_star, w_star = find_beta_star(H_target, L, W, tolerance=1e-6)
        E_check, _ = solve_weights_and_energy(beta_star, L, W)
        
        self.assertAlmostEqual(E_check, H_target, places=5)
        self.assertAlmostEqual(sum(w_star), 1.0, places=10)
        
        print(f"[PASS] TEST 9: Bisection bracketing (E_range=[{E_low:.4f}, {E_high:.4f}])")
    
    
    def test_barrier_crossing_indicator(self):
        """TEST 10: Barrier crossing detection (indicator function)"""
        B = -0.1
        alpha = 1.0
        
        path_crosses = [0.0, -0.05, -0.15, -0.1, 0.05]
        L_crosses = alpha if min(path_crosses) <= B else 0.0
        self.assertEqual(L_crosses, alpha)
        
        path_stays = [0.0, -0.05, 0.02, 0.01, 0.03]
        L_stays = alpha if min(path_stays) <= B else 0.0
        self.assertEqual(L_stays, 0.0)
        
        path_at = [0.0, -0.05, -0.1, 0.02]
        L_at = alpha if min(path_at) <= B else 0.0
        self.assertEqual(L_at, alpha)
        
        print("[PASS] TEST 10: Barrier crossing detection works correctly")


class TestFreeEnergyAndRateFunction(unittest.TestCase):
    
    def test_free_energy_positivity(self):
        """TEST 11: Free energy is positive (KL divergence property)
        
        F(v) = KL(v || Uniform) + (1/beta) * energy_term
        Since KL >= 0, we have F >= energy_term / beta > 0
        """
        S = 4
        beta = 1.5
        weights = [0.3, 0.2, 0.3, 0.2]
        L = [0.1, 0.5, 0.2, 0.4]
        W = [
            [0.0, 0.05, 0.02, 0.01],
            [0.05, 0.0, 0.04, 0.02],
            [0.02, 0.04, 0.0, 0.03],
            [0.01, 0.02, 0.03, 0.0]
        ]
        
        F = compute_free_energy(weights, L, W, beta)
        
        # F should be positive
        self.assertGreater(F, 0.0)
        # F should be finite
        self.assertTrue(math.isfinite(F))
        
        print(f"[PASS] TEST 11: Free energy positivity (F={F:.6f})")
    
    
    def test_default_probability_bounded(self):
        """TEST 12: Default probability is in [0, 1]
        
        p* = sum_s w_s * d_s where w_s are weights, d_s are binary indicators
        Since w_s >= 0 and sum(w_s) = 1, we have 0 <= p* <= 1
        """
        weights = [0.25, 0.25, 0.25, 0.25]
        d = [0, 1, 1, 0]
        
        p_star = compute_default_prob(weights, d)
        
        # Must be in [0, 1]
        self.assertGreaterEqual(p_star, 0.0)
        self.assertLessEqual(p_star, 1.0)
        # For this case: 0.25*0 + 0.25*1 + 0.25*1 + 0.25*0 = 0.5
        self.assertAlmostEqual(p_star, 0.5, places=10)
        
        print(f"[PASS] TEST 12: Default probability bounded (p*={p_star:.4f})")
    
    
    def test_binomial_tail_probability_bounded(self):
        """TEST 13: Binomial tail probability in [0, 1]
        
        P(X >= k) = sum_{j=k}^N C(N,j) p^j (1-p)^(N-j)
        This is a valid probability, so must be in [0, 1]
        """
        test_cases = [
            (5, 10, 0.5),   # P(X >= 5) where X ~ Binomial(10, 0.5)
            (10, 10, 0.1),  # P(X >= 10) where X ~ Binomial(10, 0.1) = p^10
            (0, 5, 0.9),    # P(X >= 0) = 1 always
        ]
        
        for k, N, p in test_cases:
            prob = binomial_tail_probability(k, N, p)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertTrue(math.isfinite(prob))
        
        print("[PASS] TEST 13: Binomial tail probabilities bounded in [0,1]")
    
    
    def test_binomial_edge_case_all_default(self):
        """TEST 14: Binomial P(X >= N) = p^N for X ~ Binomial(N, p)
        
        Only one term: C(N, N) * p^N * (1-p)^0 = p^N
        """
        N = 10
        p = 0.1
        k = N
        
        prob = binomial_tail_probability(k, N, p)
        expected = p ** N
        
        self.assertAlmostEqual(prob, expected, places=15)
        
        print(f"[PASS] TEST 14: Binomial edge case P(X >= N) = {prob:.2e}")
    
    
    def test_typical_regime_zero_rate_function(self):
        """TEST 15: In typical regime (q <= p*), rate function I* = 0
        
        When q is below the typical default probability p*, we're in the
        normal regime and no large deviations are needed. Thus I*(q) = 0.
        """
        S = 5
        beta_star = 1.0
        L = [0.1, 0.4, 0.2, 0.3, 0.5]
        W = [
            [0.0, 0.02, 0.01, 0.02, 0.01],
            [0.02, 0.0, 0.03, 0.01, 0.02],
            [0.01, 0.03, 0.0, 0.02, 0.01],
            [0.02, 0.01, 0.02, 0.0, 0.03],
            [0.01, 0.02, 0.01, 0.03, 0.0]
        ]
        
        # Get baseline weights and default prob
        E, w_star = solve_weights_and_energy(beta_star, L, W)
        d = [0, 1, 0, 1, 0]
        p_star = compute_default_prob(w_star, d)
        
        # Test with q < p_star (typical regime)
        q_typical = 0.5 * p_star  # Well below typical prob
        I_q, v_q = find_rate_function_I(q_typical, p_star, w_star, d, L, W, beta_star)
        
        # In typical regime, I* should be 0
        self.assertAlmostEqual(I_q, 0.0, places=6)
        
        print(f"[PASS] TEST 15: Typical regime rate function (I*={I_q:.2e}, p*={p_star:.4f})")
    
    
    def test_rare_event_rate_function_positive(self):
        """TEST 16: In rare regime (q > p*), rate function I* > 0
        
        When q is above typical default probability, we're in rare event regime
        and must shift the ensemble. Thus I*(q) > 0.
        """
        S = 5
        beta_star = 1.0
        L = [0.1, 0.4, 0.2, 0.3, 0.5]
        W = [
            [0.0, 0.02, 0.01, 0.02, 0.01],
            [0.02, 0.0, 0.03, 0.01, 0.02],
            [0.01, 0.03, 0.0, 0.02, 0.01],
            [0.02, 0.01, 0.02, 0.0, 0.03],
            [0.01, 0.02, 0.01, 0.03, 0.0]
        ]
        
        E, w_star = solve_weights_and_energy(beta_star, L, W)
        d = [0, 1, 0, 1, 0]
        p_star = compute_default_prob(w_star, d)
        
        # Test with q > p_star (rare event regime)
        q_rare = 1.5 * p_star  # Well above typical prob
        if q_rare <= 1.0:
            I_q, v_q = find_rate_function_I(q_rare, p_star, w_star, d, L, W, beta_star)
            
            # In rare regime, I* should be positive
            self.assertGreater(I_q, 0.0)
        
        print(f"[PASS] TEST 16: Rare event rate function (I* > 0 when q > p*)")
    
    
    def test_systemic_crash_probability_monotonicity(self):
        """TEST 17: Crash probability P(k/N >= q) decreases as k increases
        
        As we increase the number of defaulting firms k, the crash probability
        should decrease (rarer events have lower probability).
        """
        N = 10
        S = 20
        beta_star = 1.0
        L = [0.1*i for i in range(S)]
        W = [[0.01 if i != j else 0.0 for j in range(S)] for i in range(S)]
        d = [1 if i < S//2 else 0 for i in range(S)]
        
        E, w_star = solve_weights_and_energy(beta_star, L, W)
        
        # Test multiple k values
        crash_probs = []
        for k in range(1, N):
            result = predict_systemic_crash(k, N, w_star, d, L, W, beta_star)
            crash_probs.append(result["crash_probability"])
        
        # Probabilities should decrease (or stay finite)
        for i in range(len(crash_probs) - 1):
            # Allow some numerical slack
            self.assertGreaterEqual(crash_probs[i], crash_probs[i+1] - 1e-6)
        
        print(f"[PASS] TEST 17: Crash probability monotonicity verified")
    
    
    def test_systemic_crash_output_structure(self):
        """TEST 18: predict_systemic_crash returns all required fields
        
        Output must include: k, N, q, p_star, I_q, exponent, crash_probability
        """
        N = 5
        k = 3
        S = 10
        beta_star = 0.8
        L = [0.1, 0.3, 0.2, 0.4, 0.15, 0.25, 0.35, 0.2, 0.3, 0.1]
        W = [[0.01 if i != j else 0.0 for j in range(S)] for i in range(S)]
        d = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
        
        E, w_star = solve_weights_and_energy(beta_star, L, W)
        
        result = predict_systemic_crash(k, N, w_star, d, L, W, beta_star)
        
        # Verify all required keys exist
        required_keys = ["k", "N", "q", "p_star", "I_q", "exponent", 
                        "crash_probability", "log_crash_probability", "underflow_flag"]
        for key in required_keys:
            self.assertIn(key, result)
        
        # Verify values
        self.assertEqual(result["k"], k)
        self.assertEqual(result["N"], N)
        self.assertAlmostEqual(result["q"], k / N)
        self.assertGreaterEqual(result["p_star"], 0.0)
        self.assertLessEqual(result["p_star"], 1.0)
        self.assertGreaterEqual(result["I_q"], 0.0)
        self.assertGreaterEqual(result["exponent"], 0.0)
        
        print(f"[PASS] TEST 18: Crash output structure complete (exponent={result['exponent']:.2f})")


if __name__ == '__main__':
    unittest.main()
