import math
import random


def run_algorithm(P_prior, P, N, K_prior, K, k_target):
    T = 1.0
    dt = T / K
    B = -0.10
    alpha = 1.0
    gamma = 0.5

    # 1.1 Estimate per-step drift mu_hat[i] from the prior window
    # Formula: mu_hat_i = (1 / K_prior) * log( P_prior_i(K_prior) / P_prior_i(0) )
    mu_hat = [0.0 for i in range(N)]
    for i in range(N):
        initial_prior_price = P_prior[i][0]
        final_prior_price = P_prior[i][K_prior]
        total_prior_log_return = math.log(final_prior_price / initial_prior_price)
        mu_hat[i] = total_prior_log_return / K_prior

    # 1.2 Cumulative log-returns in active window: Y_i(t_m) = log(P_i(t_m) / P_i(0))
    Y = [[0.0 for m in range(K + 1)] for i in range(N)]
    for i in range(N):
        initial_price = P[i][0]
        for m in range(K + 1):
            Y[i][m] = math.log(P[i][m] / initial_price)

    # 1.3 Remove prior linear drift to construct centered trajectories omega_i(t_m)
    # Formula: omega_i(t_m) = Y_i(t_m) - m * mu_hat_i
    omega = [[0.0 for m in range(K + 1)] for i in range(N)]
    for i in range(N):
        for m in range(K + 1):
            omega[i][m] = Y[i][m] - (m * mu_hat[i])

    # 1.4 Time increments: d_omega_i(t_m) = omega_i(t_m) - omega_i(t_{m-1})
    d_omega = [[0.0 for m in range(K + 1)] for i in range(N)]
    for i in range(N):
        for m in range(1, K + 1):
            d_omega[i][m] = omega[i][m] - omega[i][m - 1]

    # 1.5 Idiosyncratic residuals: subtract cross-sectional average of increments at each step m
    epsilon = [[0.0 for m in range(K + 1)] for i in range(N)]
    for m in range(1, K + 1):
        sum_across_stocks = 0.0
        for j in range(N):
            sum_across_stocks += d_omega[j][m]
        market_average = sum_across_stocks / N

        for i in range(N):
            epsilon[i][m] = d_omega[i][m] - market_average

    # 1.6 Calibrate baseline volatility sigma_0 via MLE
    sum_squared_epsilon = 0.0
    for i in range(N):
        for m in range(1, K + 1):
            sum_squared_epsilon += epsilon[i][m] ** 2

    sigma_0 = math.sqrt(sum_squared_epsilon / (N * K * dt))

    # 1.7 Compute realized intensive energy H_real
    sum_individual_losses = 0.0
    for i in range(N):
        if min(omega[i]) <= B:
            sum_individual_losses += alpha
    avg_individual_loss = sum_individual_losses / N

    sum_pairwise_interactions = 0.0
    for i in range(N):
        for j in range(N):
            integral_ij = 0.0
            for m in range(1, K + 1):
                integral_ij += omega[i][m] * omega[j][m] * dt
            sum_pairwise_interactions += gamma * integral_ij
    avg_pairwise_interaction = sum_pairwise_interactions / (N ** 2)

    H_real = avg_individual_loss + 0.5 * avg_pairwise_interaction

    print(f"Estimated per-step drifts mu_hat: {[round(x, 5) for x in mu_hat]}")
    print(f"Stage 1 -> sigma_0: {sigma_0:.6f}, H_real: {H_real:.6f}")

    S = 200  # Number of synthetic Monte Carlo paths
    random.seed(42)  # Fixed seed for reproducibility

    # 2.1 Generate S scaled Brownian motion paths omega_tilde[s][m]
    omega_tilde = [[0.0 for m in range(K + 1)] for s in range(S)]
    step_std = sigma_0 * math.sqrt(dt)

    for s in range(S):
        omega_tilde[s][0] = 0.0
        for m in range(1, K + 1):
            z = random.gauss(0.0, 1.0)
            omega_tilde[s][m] = omega_tilde[s][m - 1] + step_std * z

    # 2.2 Compute synthetic individual loss vector L (length S) and default indicator d
    L = [0.0 for s in range(S)]
    d = [0.0 for s in range(S)]

    for s in range(S):
        if min(omega_tilde[s]) <= B:
            L[s] = alpha
            d[s] = 1.0
        else:
            L[s] = 0.0
            d[s] = 0.0

    # 2.3 Compute synthetic pairwise interaction matrix W (size S x S)
    W = [[0.0 for r in range(S)] for s in range(S)]

    for s in range(S):
        for r in range(s, S):
            integral_sr = 0.0
            for m in range(1, K + 1):
                integral_sr += omega_tilde[s][m] * omega_tilde[r][m] * dt
            value = gamma * integral_sr
            W[s][r] = value
            W[r][s] = value

    print(f"Stage 2 -> Generated {S} paths. Defaulted synthetic paths: {int(sum(d))}/{S}")

    beta_star, w_star = find_beta_star(H_real, L, W)
    E_check, _ = solve_weights_and_energy(beta_star, L, W)

    print(f"Stage 3 -> Calibrated beta*: {beta_star:.6f}")
    print(f"Stage 3 -> Energy check E(beta*): {E_check:.6f} vs H_real: {H_real:.6f}")

    k_target = 3  # at least 3
    result = predict_systemic_crash(k_target, N, w_star, d, L, W, beta_star)

    print("\n--- STAGE 4 FINAL RESULTS ---")
    print(f"Single-stock equilibrium default prob (p*): {result['p_star']:.6f}")
    print(f"Target crash fraction (q = k/N):            {result['q']:.6f}")
    print(f"Rate function I*(q):                        {result['I_q']:.6f}")
    print(f"Exponent N * I*(q):                         {result['exponent']:.6f}")
    print(f"Underflow flag (>= 700):                    {result['underflow_flag']}")
    print(f"Crash Probability P_N(Crash_{{k,N}}):         {result['crash_probability']:.6e}")
    print(f"Log Crash Probability:                      {result['log_crash_probability']:.6f}")



def solve_weights_and_energy(beta, L, W, tolerance=1e-8, max_iterations=500):
    """
    INNER LOOP: For a fixed candidate temperature beta, finds the
    self-consistent equilibrium weights w*(beta) and expected energy E(beta).
    """
    S = len(L)

    # Initialize uniform weights: w_s^{(0)} = 1 / S
    w = [1.0 / S for s in range(S)]

    for iteration in range(max_iterations):
        # Step 1: Compute interaction field h^{(n)} = W * w^{(n)}
        h = [0.0 for s in range(S)]
        for s in range(S):
            sum_interaction = 0.0
            for r in range(S):
                sum_interaction += W[s][r] * w[r]
            h[s] = sum_interaction

        # Step 2: Compute unnormalized weights w_tilde^{(n+1)}
        # Using max-shift inside exp() to prevent float64 underflow/overflow
        exponents = [0.0 for s in range(S)]
        for s in range(S):
            exponents[s] = -(L[s] + h[s]) / beta

        max_exponent = max(exponents)

        w_tilde = [0.0 for s in range(S)]
        for s in range(S):
            w_tilde[s] = math.exp(exponents[s] - max_exponent)

        # Step 3: Normalize to enforce sum(w_new) = 1
        sum_w_tilde = sum(w_tilde)
        w_new = [0.0 for s in range(S)]
        for s in range(S):
            w_new[s] = w_tilde[s] / sum_w_tilde

        # Apply 0.5 damping for stability and check max coordinate change
        max_diff = 0.0
        for s in range(S):
            damped_value = 0.5 * w[s] + 0.5 * w_new[s]
            diff = abs(damped_value - w[s])
            if diff > max_diff:
                max_diff = diff
            w[s] = damped_value

        if max_diff < tolerance:
            break

    # Compute final interaction field h* for the converged weights w
    h_star = [0.0 for s in range(S)]
    for s in range(S):
        sum_interaction = 0.0
        for r in range(S):
            sum_interaction += W[s][r] * w[r]
        h_star[s] = sum_interaction

    # Compute expected Hamiltonian energy: E(beta) = w^T L + 0.5 * w^T h*
    expected_L = 0.0
    expected_W = 0.0
    for s in range(S):
        expected_L += w[s] * L[s]
        expected_W += 0.5 * w[s] * h_star[s]

    E_beta = expected_L + expected_W
    return E_beta, w


def find_beta_star(H_real, L, W, tolerance=1e-6, max_bisection_steps=60):
    """
    OUTER LOOP: Solves E(beta*) - H_real = 0 via bisection.
    """
    beta_low = 1e-4
    beta_high = 1.0

    # Expand upper bracket until E(beta_high) >= H_real
    # (Since E(beta) increases as beta increases)
    E_high, _ = solve_weights_and_energy(beta_high, L, W)
    while E_high < H_real:
        beta_high *= 2.0
        E_high, _ = solve_weights_and_energy(beta_high, L, W)
        if beta_high > 1e5:
            print("Warning: H_real exceeds maximum achievable model energy E(inf).")
            break

    # Bisect the intervals
    beta_mid = 0.5 * (beta_low + beta_high)
    w_star = []

    for step in range(max_bisection_steps):
        beta_mid = 0.5 * (beta_low + beta_high)
        E_mid, w_star = solve_weights_and_energy(beta_mid, L, W)

        error = E_mid - H_real
        if abs(error) < tolerance:
            break

        if error < 0.0:
            # E(beta_mid) is too small
            beta_low = beta_mid
        else:
            # E(beta_mid) is too large
            beta_high = beta_mid

    return beta_mid, w_star





def compute_free_energy(weights, L, W, beta_star):
    """
    Computes F(v) = KL(v || Uniform) + (1 / beta*) * (v^T L + 0.5 * v^T W v)
    """
    S = len(weights)

    # 1. Discrete KL divergence: sum_s v_s * log(S * v_s)
    kl_div = 0.0
    for s in range(S):
        if weights[s] > 0.0:
            kl_div += weights[s] * math.log(S * weights[s])

    # 2. Expected individual loss: sum_s v_s * L_s
    expected_L = 0.0
    for s in range(S):
        expected_L += weights[s] * L[s]

    # 3. Expected pairwise interaction: 0.5 * sum_s sum_r v_s * W_{s,r} * v_r
    expected_W = 0.0
    for s in range(S):
        interaction_s = 0.0
        for r in range(S):
            interaction_s += W[s][r] * weights[r]
        expected_W += 0.5 * weights[s] * interaction_s

    return kl_div + (expected_L + expected_W) / beta_star


def solve_tilted_weights(theta, d, L, W, beta_star):
    """
    INNER LOOP FOR v(theta):
    Shifts L_s -> L_s - beta* * theta * d_s and reuses the exact Stage 3 solver.
    """
    S = len(L)
    L_shifted = [0.0 for s in range(S)]
    for s in range(S):
        L_shifted[s] = L[s] - beta_star * theta * d[s]

    _, v_theta = solve_weights_and_energy(beta_star, L_shifted, W)
    return v_theta


def compute_default_prob(weights, d):
    """
    Computes total probability mass assigned to defaulted paths: sum_s weights_s * d_s
    """
    prob = 0.0
    for s in range(len(weights)):
        prob += weights[s] * d[s]
    return prob


def find_rate_function_I(q, p_star, w_star, d, L, W, beta_star, tolerance=1e-6, max_steps=60):
    """
    OUTER LOOP FOR theta*:
    Finds theta* such that sum_s v_s(theta*) * d_s = q, then computes I*(q).
    """
    # Case 1: Typical regime (q <= p*), no tilting needed
    if q <= p_star:
        return 0.0, w_star

    # Case 2: Rare crash regime (q > p*), find theta* via bisection
    theta_low = 0.0
    theta_high = 0.5

    # Same as before
    v_high = solve_tilted_weights(theta_high, d, L, W, beta_star)
    p_high = compute_default_prob(v_high, d)

    while p_high < q:
        theta_high *= 2.0
        v_high = solve_tilted_weights(theta_high, d, L, W, beta_star)
        p_high = compute_default_prob(v_high, d)
        if theta_high > 1e4:
            print("Warning: Target q is too close to 1.0 for the available synthetic paths.")
            break

    # Bisection loop
    theta_mid = 0.5 * (theta_low + theta_high)
    v_star = w_star

    for step in range(max_steps):
        theta_mid = 0.5 * (theta_low + theta_high)
        v_star = solve_tilted_weights(theta_mid, d, L, W, beta_star)
        p_mid = compute_default_prob(v_star, d)

        error = p_mid - q
        if abs(error) < tolerance:
            break

        if error < 0.0:
            # p(theta_mid) is too small
            theta_low = theta_mid
        else:
            # p(theta_mid) is too large
            theta_high = theta_mid

    # Evaluate I*(q) = F(v*) - F(w*)
    F_min = compute_free_energy(w_star, L, W, beta_star)
    F_tilted = compute_free_energy(v_star, L, W, beta_star)
    I_q = max(0.0, F_tilted - F_min)

    return I_q, v_star


def binomial_tail_probability(k, N, p_star):
    """
    Computes exact Binomial upper tail: sum_{j=k}^N C(N, j) * (p*)^j * (1 - p*)^{N - j}
    """
    total_prob = 0.0
    for j in range(k, N + 1):
        combinations = math.comb(N, j)
        term = combinations * (p_star ** j) * ((1.0 - p_star) ** (N - j))
        total_prob += term
    return total_prob


def predict_systemic_crash(k, N, w_star, d, L, W, beta_star):
    """
    output: Evaluates P_N(Crash_{k, N}) using the N * I*(q) < 700 decisioning.
    """
    q = k / N
    p_star = compute_default_prob(w_star, d)
    I_q, v_star = find_rate_function_I(q, p_star, w_star, d, L, W, beta_star)

    exponent = N * I_q

    if exponent < 700.0:
        underflow_flag = False
        crash_prob = binomial_tail_probability(k, N, p_star)
        log_crash_prob = math.log(crash_prob) if crash_prob > 0.0 else -exponent
    else:
        underflow_flag = True
        crash_prob = 0.0
        log_crash_prob = -exponent

    return {
        "k": k,
        "N": N,
        "q": q,
        "p_star": p_star,
        "I_q": I_q,
        "exponent": exponent,
        "underflow_flag": underflow_flag,
        "crash_probability": crash_prob,
        "log_crash_probability": log_crash_prob,
    }

