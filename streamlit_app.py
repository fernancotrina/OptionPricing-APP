import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions import (
    OptionSpec, bs_price, bs_greeks, binomial_price, 
    lsm_american_price, binomial_greeks, lsm_greeks
)

# Configuración de la página
st.set_page_config(
    page_title="Analizador Avanzado de Opciones",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Analizador Avanzado de Opciones Financieras")
st.markdown("""
Esta aplicación calcula precios y griegas de opciones usando diferentes métodos de valuación:
- **Black-Scholes** para opciones europeas
- **Modelo Binomial** para opciones europeas y americanas
- **Longstaff-Schwartz (Monte Carlo)** para opciones americanas
""")

# Barra lateral para parámetros
with st.sidebar:
    st.header("📋 Parámetros de la Opción")
    
    # Inputs básicos
    col1, col2 = st.columns(2)
    with col1:
        S = st.number_input("Precio Spot (S)", value=100.0, min_value=0.01, step=1.0)
        K = st.number_input("Precio Ejercicio (K)", value=100.0, min_value=0.01, step=1.0)
        T = st.number_input("Tiempo (T años)", value=0.25, min_value=0.0, max_value=50.0, step=0.05)
    
    with col2:
        r = st.number_input("Tasa Libre Riesgo (r)", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
        q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
        sigma = st.number_input("Volatilidad (σ)", value=0.2, min_value=0.01, max_value=2.0, step=0.01)
    
    # Selectores adicionales
    option_type = st.radio("Tipo de opción", ["Call", "Put"])
    is_call = option_type == "Call"
    
    st.divider()
    st.subheader("⚙️ Configuración de Métodos")
    
    # Configuración para Binomial
    st.markdown("**Método Binomial**")
    binomial_steps = st.slider("Número de pasos", 10, 500, 100, 10)
    american_enabled = st.checkbox("Habilitar ejercicio americano")
    
    # Configuración para LSM
    st.markdown("**Método LSM (Monte Carlo)**")
    lsm_paths = st.slider("Número de paths", 1000, 200000, 10000, 1000)
    lsm_steps = st.slider("Número de steps", 10, 200, 50, 10)
    lsm_degree = st.slider("Grado polinomial", 1, 5, 2)

# Crear especificación de la opción
spec = OptionSpec(S=S, K=K, T=T, r=r, q=q, sigma=sigma, is_call=is_call)

# Calcular precios con diferentes métodos
try:
    # Black-Scholes
    bs_result = bs_greeks(spec)
    bs_price_val = bs_result["price"]
    
    # Binomial
    binomial_price_val = binomial_price(spec, steps=binomial_steps, american=american_enabled)
    
    # LSM (solo para opciones americanas)
    lsm_price_val = lsm_american_price(
        spec, 
        n_paths=lsm_paths, 
        n_steps=lsm_steps, 
        degree=lsm_degree
    )
    
    # Calcular griegas para Binomial y LSM
    binomial_greeks_val = binomial_greeks(spec, steps=binomial_steps, american=american_enabled)
    lsm_greeks_val = lsm_greeks(
        spec, 
        n_paths=lsm_paths, 
        n_steps=lsm_steps, 
        degree=lsm_degree
    )
    
except Exception as e:
    st.error(f"Error en cálculo: {e}")
    st.stop()

# Mostrar resultados
st.header("📈 Resultados de Valuación")

# Métricas de precios
col1, col2, col3 = st.columns(3)
col1.metric("Black-Scholes", f"${bs_price_val:.4f}")
col2.metric(f"Binomial {'(Americano)' if american_enabled else '(Europeo)'}", f"${binomial_price_val:.4f}")
col3.metric("LSM (Americano)", f"${lsm_price_val:.4f}")

# Mostrar diferencias
col1, col2 = st.columns(2)
diff_bs_bin = abs(bs_price_val - binomial_price_val) / bs_price_val * 100
diff_bs_lsm = abs(bs_price_val - lsm_price_val) / bs_price_val * 100

col1.metric("Diff BS-Binomial", f"{diff_bs_bin:.2f}%")
col2.metric("Diff BS-LSM", f"{diff_bs_lsm:.2f}%")

# Mostrar griegas
st.subheader("📊 Greeks Comparativos")

greeks_data = {
    "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    "Black-Scholes": [
        bs_result["delta"],
        bs_result["gamma"],
        bs_result["vega"],
        bs_result["theta"],
        bs_result["rho"]
    ],
    "Binomial": [
        binomial_greeks_val["delta"],
        binomial_greeks_val["gamma"],
        binomial_greeks_val["vega"],
        binomial_greeks_val["theta"],
        binomial_greeks_val["rho"]
    ],
    "LSM": [
        lsm_greeks_val["delta"],
        lsm_greeks_val["gamma"],
        lsm_greeks_val["vega"],
        lsm_greeks_val["theta"],
        lsm_greeks_val["rho"]
    ]
}

# Mostrar tabla de griegas
st.dataframe(greeks_data, use_container_width=True)

# Análisis de sensibilidad
st.header("🔍 Análisis de Sensibilidad")

# Selector de parámetro para análisis
param = st.selectbox("Parámetro para análisis", ["S", "K", "T", "r", "q", "sigma"])

# Rango de valores
if param == "S":
    default_range = (S * 0.5, S * 1.5)
    values = np.linspace(S * 0.5, S * 1.5, 50)
elif param == "K":
    default_range = (K * 0.5, K * 1.5)
    values = np.linspace(K * 0.5, K * 1.5, 50)
elif param == "T":
    default_range = (0.01, T * 2)
    values = np.linspace(0.01, max(0.5, T * 2), 50)
elif param == "r":
    default_range = (0.0, 0.1)
    values = np.linspace(0.0, 0.1, 50)
elif param == "q":
    default_range = (0.0, 0.1)
    values = np.linspace(0.0, 0.1, 50)
elif param == "sigma":
    default_range = (0.05, 0.5)
    values = np.linspace(0.05, 0.5, 50)

# Calcular precios para diferentes valores del parámetro
bs_prices = []
binomial_prices = []
lsm_prices = []

for val in values:
    # Crear nueva especificación con el parámetro modificado
    if param == "S":
        new_spec = OptionSpec(S=val, K=K, T=T, r=r, q=q, sigma=sigma, is_call=is_call)
    elif param == "K":
        new_spec = OptionSpec(S=S, K=val, T=T, r=r, q=q, sigma=sigma, is_call=is_call)
    elif param == "T":
        new_spec = OptionSpec(S=S, K=K, T=val, r=r, q=q, sigma=sigma, is_call=is_call)
    elif param == "r":
        new_spec = OptionSpec(S=S, K=K, T=T, r=val, q=q, sigma=sigma, is_call=is_call)
    elif param == "q":
        new_spec = OptionSpec(S=S, K=K, T=T, r=r, q=val, sigma=sigma, is_call=is_call)
    elif param == "sigma":
        new_spec = OptionSpec(S=S, K=K, T=T, r=r, q=q, sigma=val, is_call=is_call)
    
    # Calcular precios
    bs_prices.append(bs_price(new_spec))
    binomial_prices.append(binomial_price(new_spec, steps=binomial_steps, american=american_enabled))
    lsm_prices.append(lsm_american_price(
        new_spec, 
        n_paths=lsm_paths, 
        n_steps=lsm_steps, 
        degree=lsm_degree
    ))

# Crear gráfico
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(values, bs_prices, label='Black-Scholes', linewidth=2)
ax.plot(values, binomial_prices, label=f'Binomial ({"" if american_enabled else "Europeo"})', linewidth=2)
ax.plot(values, lsm_prices, label='LSM (Americano)', linewidth=2)

ax.set_xlabel(param)
ax.set_ylabel('Precio de la opción')
ax.set_title(f'Sensibilidad del precio a {param}')
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# Información adicional
with st.expander("💡 Información sobre los métodos de valuación"):
    st.markdown("""
    **Black-Scholes**:
    - Modelo analítico para opciones europeas
    - Proporciona griegas exactas
    - No considera ejercicio anticipado
    
    **Modelo Binomial**:
    - Aproximación discreta del precio
    - Puede manejar opciones americanas (ejercicio anticipado)
    - Más preciso con más steps
    
    **Longstaff-Schwartz (LSM)**:
    - Método de Monte Carlo para opciones americanas
    - Usa regresión para estimar el valor de continuación
    - Computacionalmente intensivo pero muy flexible
    """)

# Notas al pie
st.divider()
st.caption("""
*Nota: Esta aplicación es para fines educativos y de análisis. 
Los resultados no deben considerarse como recomendaciones de inversión.*
""")
