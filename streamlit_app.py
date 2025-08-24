import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración
st.set_page_config(page_title="Modelos de Opciones", layout="wide")
st.title("Modelos de Valoración de Opciones")

# Sidebar (igual que antes)
with st.sidebar:
    st.title("Pricing Model")
    model = st.selectbox("Model", ["Black-Scholes (European)", "Binomial", "Longstaff-Schwartz (American)"])
    current_price = st.number_input("Asset Price (S₀)", value=100.0)
    strike_price = st.number_input("Strike Price (K)", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity - Years (T)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05)
    
    if model == "Longstaff-Schwartz (American)":
        simulations = st.number_input("Simulations", min_value=100, value=10000, step=100)
        steps = st.number_input("Number of Steps", min_value=10, value=50, step=10)
    elif model == "Binomial":
        exercise = st.selectbox("Exercise Type", ["european", "american"])
        steps = st.number_input("Number of Steps", min_value=10, value=50, step=10)


# Funciones de valoración (las mismas que tenías)
def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price

def binomial_model(S, K, T, r, sigma, steps, exercise='european'):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    ST = np.array([S * u**j * d**(steps - j) for j in range(steps + 1)])

    call = np.maximum(0, ST - K)
    put = np.maximum(0, K - ST)

    for i in range(steps - 1, -1, -1):
        call = disc * (p * call[1:] + (1 - p) * call[:-1])
        put = disc * (p * put[1:] + (1 - p) * put[:-1])

        if exercise == 'american':
            ST = ST[:i+1] / u
            call = np.maximum(call, ST - K)
            put = np.maximum(put, K - ST)

    return call[0], put[0], ST

def longstaff_schwartz_american_option(S0, K, r, sigma, T, M=50, N=10000, option_type='put'):
    dt = T / M
    discount = np.exp(-r * dt)
    
    # Simulación de trayectorias
    S = np.zeros((N, M + 1))
    S[:, 0] = S0
    for t in range(1, M + 1):
        Z = np.random.normal(0, 1, N)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    if option_type == 'put':
        h = np.maximum(K - S, 0)
    else:
        h = np.maximum(S - K, 0)
    
    V = h[:, -1].copy()
    for t in range(M - 1, 0, -1):
        itm = h[:, t] > 0
        X = S[itm, t]
        Y = V[itm] * discount
        
        if len(X) > 0:
            A = np.vstack([np.ones_like(X), X, X**2]).T
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation = coeffs[0] + coeffs[1] * X + coeffs[2] * X**2
            exercise = h[itm, t]
            V[itm] = np.where(exercise > continuation, exercise, V[itm] * discount)
    
    return discount * np.mean(V)

# Función para generar superficies 3D (compatible con todos los modelos)
def plot_option_surface(model_type, S0, K, T, r, sigma, steps=50, simulations=10000, exercise='european'):
    S = np.linspace(max(S0 * 0.5, 1), S0 * 1.5, 50)  # Precios entre 50% y 150% de S0 (mínimo 1)
    volatility_range = np.linspace(0.1, 1, 50)  # Volatilidad entre 10% y 100%
    
    call_prices = np.zeros((len(S), len(volatility_range)))
    put_prices = np.zeros((len(S), len(volatility_range)))
    
    for i, s in enumerate(S):
        for j, vol in enumerate(volatility_range):
            try:
                if model_type == "Black-Scholes":
                    call, put = black_scholes(s, K, T, r, vol)
                elif model_type == "Binomial":
                    call, put, _ = binomial_model(s, K, T, r, vol, steps, exercise)
                elif model_type == "Longstaff-Schwartz":
                    call = longstaff_schwartz_american_option(s, K, r, vol, T, steps, simulations, 'call')
                    put = longstaff_schwartz_american_option(s, K, r, vol, T, steps, simulations, 'put')
                
                call_prices[i, j] = call
                put_prices[i, j] = put
            except:
                call_prices[i, j] = np.nan
                put_prices[i, j] = np.nan
    
    # Gráfico Call
    fig_call = go.Figure(go.Surface(
        z=call_prices, 
        x=S, 
        y=volatility_range, 
        colorscale='Viridis',
        contours={
            "x": {"show": True, "color":"grey"},
            "y": {"show": True, "color":"grey"}
        }
    ))
    fig_call.update_layout(
        title=f'Call Option ({model_type})',
        scene=dict(
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            zaxis_title='Price',
            camera={"eye": {"x": 1.2, "y": 2, "z": 0.2}}
        ),
        margin={"r": 20, "l": 10, "b": 10}
    )
    
    # Gráfico Put
    fig_put = go.Figure(go.Surface(
        z=put_prices, 
        x=S, 
        y=volatility_range,
        colorscale='Plasma',
        contours={
            "x": {"show": True, "color":"grey"},
            "y": {"show": True, "color":"grey"}
        }
    ))
    fig_put.update_layout(
        title=f'Put Option ({model_type})',
        scene=dict(
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            zaxis_title='Price',
            camera={"eye": {"x": 2, "y": 2, "z": 2}}
        ),
        margin={"r": 20, "l": 10, "b": 10}
    )
    
    return fig_call, fig_put

# Resultados para todos los modelos
if model == "Black-Scholes (European)":
    call, put = black_scholes(current_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    
    # Primera fila: Precios
    st.subheader("Resultados")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precio Call Europea", f"{call:.2f}")
    with col2:
        st.metric("Precio Put Europea", f"{put:.2f}")
    
    # Segunda fila: Superficies
    st.subheader("Superficies de Valoración")
    fig_call, fig_put = plot_option_surface("Black-Scholes", current_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_call, use_container_width=True)
    with col2:
        st.plotly_chart(fig_put, use_container_width=True)

elif model == "Binomial":
    call, put, ST = binomial_model(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, int(steps), exercise)
    
    # Primera fila: Precios
    st.subheader("Resultados")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Precio Call ({exercise.capitalize()})", f"{call:.2f}")
    with col2:
        st.metric(f"Precio Put ({exercise.capitalize()})", f"{put:.2f}")
    
    # Segunda fila: Superficies
    st.subheader("Superficies de Valoración")
    fig_call, fig_put = plot_option_surface("Binomial", current_price, strike_price, time_to_maturity, risk_free_rate, volatility, int(steps))
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_call, use_container_width=True)
    with col2:
        st.plotly_chart(fig_put, use_container_width=True)
    
    # Tercera fila: Trayectorias (opcional)
    st.subheader("Trayectorias Binomiales")
    plt.figure(figsize=(10, 4))
    plt.plot(ST, 'o-', alpha=0.5)
    plt.title("Evolución del Precio en el Árbol Binomial")
    st.pyplot(plt)

elif model == "Longstaff-Schwartz (American)":
    call = longstaff_schwartz_american_option(current_price, strike_price, risk_free_rate, volatility, time_to_maturity, int(steps), int(simulations), 'call')
    put = longstaff_schwartz_american_option(current_price, strike_price, risk_free_rate, volatility, time_to_maturity, int(steps), int(simulations), 'put')
    
    # Primera fila: Precios
    st.subheader("Resultados")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precio Call Americana", f"{call:.2f}")
    with col2:
        st.metric("Precio Put Americana", f"{put:.2f}")
    
    # Segunda fila: Superficies
    st.subheader("Superficies de Valoración")
    fig_call, fig_put = plot_option_surface("Longstaff-Schwartz", current_price, strike_price, time_to_maturity, risk_free_rate, volatility, int(steps), int(simulations))
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_call, use_container_width=True)
    with col2:
        st.plotly_chart(fig_put, use_container_width=True)
    
    # Tercera fila: Simulaciones (opcional)
    st.subheader("Trayectorias Simuladas")
    np.random.seed(42)
    M = int(steps)
    N = 100
    S_sim = np.zeros((N, M + 1))
    S_sim[:, 0] = current_price
    for t in range(1, M + 1):
        Z = np.random.normal(0, 1, N)
        S_sim[:, t] = S_sim[:, t - 1] * np.exp((risk_free_rate - 0.5 * volatility**2) * (time_to_maturity/M) + volatility * np.sqrt(time_to_maturity/M) * Z)
    
    plt.figure(figsize=(10, 4))
    for i in range(N):
        plt.plot(S_sim[i], alpha=0.3)
    plt.axhline(strike_price, color='r', linestyle='--', label='Strike Price')
    plt.title("Simulaciones de Monte Carlo")
    st.pyplot(plt)

st.markdown("---")
st.markdown("Creado por José Cotrina Lejabo | LinkedIn próximamente")