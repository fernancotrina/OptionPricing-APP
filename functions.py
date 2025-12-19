from __future__ import annotations

import math
from scipy.stats import norm
import numpy as np
import plotly.graph_objects as go

# =========================
# 1) Black–Scholes (Europeax)
# =========================

def bs_price(S, K, T, r, q, sigma, is_call=True):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if is_call:
        return (
            S * math.exp(-q * T) * norm.cdf(d1)
            - K * math.exp(-r * T) * norm.cdf(d2)
        )
    else:
        return (
            K * math.exp(-r * T) * norm.cdf(-d2)
            - S * math.exp(-q * T) * norm.cdf(-d1)
        )

# ==========================================
# 2. Griegas para BSM
# ==========================================

def bs_greeks(S, K, T, r, q, sigma, is_call=True):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    Nd1 = norm.cdf(d1)
    nd1 = norm.pdf(d1)

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    # Delta
    delta = disc_q * Nd1 if is_call else disc_q * (Nd1 - 1)

    # Gamma
    gamma = disc_q * nd1 / (S * sigma * math.sqrt(T))

    # Vega
    vega = S * disc_q * nd1 * math.sqrt(T)

    # Theta
    term1 = -(S * disc_q * nd1 * sigma) / (2 * math.sqrt(T))
    if is_call:
        theta = term1 - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * Nd1
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        theta = term1 + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        rho = -K * T * disc_r * norm.cdf(-d2)

    price = bs_price(S, K, T, r, q, sigma, is_call)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }

# ==========================================
# 3. Gráfico para BSM
# ==========================================

def plot_bs_surface(S_center, K, T, r, q, sigma_center, is_call=True):

    spot_range = np.linspace(max(0.01, S_center * 0.5), S_center * 1.5, 50)
    vol_range = np.linspace(0.05, 1.0, 50)
    
    X_spot, Y_vol = np.meshgrid(spot_range, vol_range)
    
    
    d1 = (np.log(X_spot / K) + (r - q + 0.5 * Y_vol**2) * T) / (Y_vol * np.sqrt(T))
    d2 = d1 - Y_vol * np.sqrt(T)
    
    if is_call:
        Z_price = (X_spot * np.exp(-q * T) * norm.cdf(d1) - 
                   K * np.exp(-r * T) * norm.cdf(d2))
        titulo = "Call Option Price Surface"
        color_scale = "Viridis"
    else:
        Z_price = (K * np.exp(-r * T) * norm.cdf(-d2) - 
                   X_spot * np.exp(-q * T) * norm.cdf(-d1))
        titulo = "Put Option Price Surface"
        color_scale = "Plasma"

    fig = go.Figure(data=[go.Surface(
        z=Z_price, 
        x=X_spot, 
        y=Y_vol * 100, # Mostrar volatilidad en %
        colorscale=color_scale,
        opacity=0.9,
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    )])

    fig.update_layout(
        title=titulo,
        scene=dict(
            xaxis_title='Precio Spot ($)',
            yaxis_title='Volatilidad (%)',
            zaxis_title='Opción ($)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500
    )
    
    return fig

# =========================
# 4) Binomial (Americasna)
# ========================

def bi_price(S, K, T, r, q, sigma, N=100, is_call=True, is_american=True):
    if N < 1: N = 1
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # Probabilidad neutral al riesgo
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    # Factor de descuento por paso
    disc = np.exp(-r * dt)

    # Precios en el vencimiento (nodos finales)
    j = np.arange(N + 1)
    S_T = S * (u ** j) * (d ** (N - j))
    
    # Payoff en el vencimiento
    if is_call:
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    # Inducción hacia atrás
    for i in range(N - 1, -1, -1):
        # Valor de continuación
        valor_cont = disc * (p * payoffs[1:i+2] + (1 - p) * payoffs[0:i+1])
        
        if is_american:
            # Recalcular el precio del activo subyacente en el paso 'i'
            S_node = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            
            # Valor de ejercicio inmediato
            if is_call:
                valor_ejer = np.maximum(S_node - K, 0)
            else:
                valor_ejer = np.maximum(K - S_node, 0)
            
            # Esperar o Ejerceer
            payoffs = np.maximum(valor_cont, valor_ejer)
        else:
            payoffs = valor_cont

    return payoffs[0]

# ==========================================
# 5. Griegas para Binomial
# ==========================================


def bi_greeks(S, K, T, r, q, sigma, N=100, is_call=True, is_american=True):
    price = bi_price(S, K, T, r, q, sigma, N, is_call, is_american)
    
    # Delta & Gamma
    dS = S * 0.01
    p_up = bi_price(S + dS, K, T, r, q, sigma, N, is_call, is_american)
    p_down = bi_price(S - dS, K, T, r, q, sigma, N, is_call, is_american)
    
    delta = (p_up - p_down) / (2 * dS)
    gamma = (p_up - 2 * price + p_down) / (dS ** 2)
    
    # Vdega
    dSigma = 0.01 # 1% cambio absoluto
    p_vega = bi_price(S, K, T, r, q, sigma + dSigma, N, is_call, is_american)
    vega = (p_vega - price) # Cambio por 1% de volatilidad
    
    # Theta
    dT = 1/365.0 
    if T > dT:
        p_theta = bi_price(S, K, T - dT, r, q, sigma, N, is_call, is_american)
        theta = (p_theta - price) # Cambio por un día menos
    else:
        theta = 0.0
        
    # Rho
    dr = 0.01 # 1% cambio
    p_rho = bi_price(S, K, T, r + dr, q, sigma, N, is_call, is_american)
    rho = (p_rho - price) # Cambio por 1% de tasa

    return {
        "price": price, "delta": delta, "gamma": gamma, 
        "vega": vega, "theta": theta, "rho": rho
    }

# ==========================================
# 6. Gráfico para Binomial
# ==========================================

def plot_binomial_convergence(S, K, T, r, q, sigma, is_call=True, is_american=True):
    steps_range = list(range(10, 201, 5))
    prices = []
    
    for n in steps_range:
        p = bi_price(S, K, T, r, q, sigma, N=n, is_call=is_call, is_american=is_american)
        prices.append(p)
        
    # Línea de referencia (Black-Scholes Europea) para comparar
    bs_ref = bs_price(S, K, T, r, q, sigma, is_call)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps_range, y=prices, mode='lines+markers', name='Binomial (N)'))
    
    # Solo agregamos referencia BS si es Europea o si el usuario entiende la diferencia
    if not is_american or (is_american and q == 0 and is_call):
         # Call americana sin dividentos = Europea
         fig.add_trace(go.Scatter(x=steps_range, y=[bs_ref]*len(steps_range), 
                                  mode='lines', name='Black-Scholes (Ref)', line=dict(dash='dash', color='gray')))

    fig.update_layout(
        title="Convergencia del Modelo Binomial",
        xaxis_title="Número de Pasos (N)",
        yaxis_title="Precio de la Opción",
        height=400,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

