# models.py
import numpy as np

# --- Définition des Fonctions Modèles ---
def langmuir_model(Ce, qm, KL):
    KL = max(KL, 0)
    Ce_safe = np.maximum(0, Ce)
    epsilon = 1e-12
    denominator = 1 + KL * Ce_safe
    denominator = np.where(np.abs(denominator) < epsilon, np.sign(denominator + epsilon) * epsilon, denominator)
    return np.where(denominator != 0, (qm * KL * Ce_safe) / denominator, 0)

def freundlich_model(Ce, KF, n_inv):
    KF = max(KF, 0)
    n_inv = max(n_inv, 0)
    epsilon = 1e-12
    Ce_safe = np.maximum(Ce, epsilon)
    return KF * Ce_safe**n_inv

def pfo_model(t, qe, k1):
    k1 = max(k1, 0)
    k1_safe = max(k1, 1e-12)
    t_safe = np.clip(t, 0, None)
    exp_arg = -k1 * np.clip(t_safe, 0, 700 / k1_safe if k1_safe > 0 else 700) 
    exp_term = np.exp(exp_arg)
    return qe * (1 - exp_term)

def pso_model(t, qe, k2):
    qe = max(qe, 0)
    k2 = max(k2, 0)
    qe_safe = max(qe, 1e-12)
    k2_safe = max(k2, 1e-12)
    t_safe = np.clip(t, 0, None)
    epsilon = 1e-12
    denominator = 1 + k2_safe * qe_safe * t_safe
    # Ensure denominator is not zero
    denominator = np.where(np.abs(denominator) < epsilon, 1e-9 * np.sign(denominator + epsilon), denominator)
    return (k2_safe * qe_safe**2 * t_safe) / denominator
def temkin_model_nonlinear(Ce, B1, K_T):
    """
    Non-linear Temkin isotherm model.
    qe = B1 * ln(K_T * Ce)
    B1 = RT/bT
    K_T = Temkin equilibrium binding constant
    """
    Ce_safe = np.maximum(Ce, 1e-9) # Avoid log(0)
    K_T = max(K_T, 1e-9) # Ensure K_T is positive
    # B1 can be positive or negative depending on the system, so no strict positive constraint here during fitting.
    # However, for physical meaning, bT is usually positive.
    return B1 * np.log(K_T * Ce_safe)