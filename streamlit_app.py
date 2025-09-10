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

# Barra lateral para selección del modelo
with st.sidebar:
    st.header("🔧 Configuración")
    
    # Selección del modelo
    modelo = st.radio(
        "Selecciona el modelo de valuación:",
        ["Black-Scholes", "Binomial", "Longstaff-Schwartz (LSM)"],
        index=0
    )
    
    st.divider()
    st.header("📋 Parámetros Comunes")
    
    # Inputs básicos comunes a todos los modelos
    S = st.number_input("Precio Spot (S)", value=100.0, min_value=0.01, step=1.0)
    K = st.number_input("Precio Ejercicio (K)", value=100.0, min_value=0.01, step=1.0)
    T = st.number_input("Tiempo (T años)", value=0.25, min_value=0.0, max_value=50.0, step=0.05)
    r = st.number_input("Tasa Libre Riesgo (r)", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
    sigma = st.number_input("Volatilidad (σ)", value=0.2, min_value=0.01, max_value=2.0, step=0.01)
    
    # Parámetros específicos por modelo
    st.divider()
    st.header("⚙️ Parámetros Específicos")
    

    if modelo == "Binomial":
        q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
        binomial_steps = st.slider("Número de pasos", 10, 500, 100, 10)
        ejercicio = st.radio("Tipo de ejercicio", ["Europeo", "Americano"])
        american_enabled = (ejercicio == "Americano")
        
    elif modelo == "Longstaff-Schwartz (LSM)":
        q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
        lsm_paths = st.slider("Número de paths", 1000, 200000, 10000, 1000)
        lsm_steps = st.slider("Número de steps", 10, 200, 50, 10)
        lsm_degree = st.slider("Grado polinomial", 1, 5, 2)
        basis = st.selectbox("Tipo de base", ["poly", "laguerre"])

# Crear especificaciones para call y put
spec_call = OptionSpec(S=S, K=K, T=T, r=r, q=q, sigma=sigma, is_call=True)
spec_put = OptionSpec(S=S, K=K, T=T, r=r, q=q, sigma=sigma, is_call=False)

# Contenido principal según el modelo seleccionado
if modelo == "Black-Scholes":
    st.header("📈 Modelo Black-Scholes")
    
    # Teoría
    with st.expander("📚 Teoría del Modelo Black-Scholes"):
        st.markdown("""
        El modelo Black-Scholes es un modelo matemático para la valoración de opciones financieras estilo europeo.
        Desarrollado por Fischer Black, Myron Scholes y Robert Merton en 1973, este modelo revolucionó el campo de las finanzas cuantitativas.
        
        **Supuestos del modelo:**
        - El activo subyacente sigue un movimiento browniano geométrico
        - No hay dividendos durante la vida de la opción (aunque se puede extender)
        - Mercados eficientes y sin fricciones (sin costos de transacción)
        - Tasa de interés libre de riesgo constante y conocida
        - Volatilidad constante del activo subyacente
        
        **Fórmulas principales:**
        - Call: $C = S_0N(d_1) - Ke^{-rT}N(d_2)$
        - Put: $P = Ke^{-rT}N(-d_2) - S_0N(-d_1)$
        
        Donde:
        - $d_1 = \\frac{\\ln(S_0/K) + (r + \\sigma^2/2)T}{\\sigma\\sqrt{T}}$
        - $d_2 = d_1 - \\sigma\\sqrt{T}$
        - $N(x)$ es la función de distribución acumulativa de la normal estándar
        """)
    
    # Calcular precios y griegas
    try:
        bs_call = bs_greeks(spec_call)
        bs_put = bs_greeks(spec_put)
        
        # Mostrar precios
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Precio Call")
            st.metric("Valor", f"${bs_call['price']:.4f}")
            
        with col2:
            st.subheader("Precio Put")
            st.metric("Valor", f"${bs_put['price']:.4f}")
        
        # Mostrar griegas en tabs
        st.subheader("Griegas")
        tab1, tab2 = st.tabs(["Call", "Put"])
        
        with tab1:
            greeks_call_data = {
                "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                "Valor": [
                    bs_call["delta"],
                    bs_call["gamma"],
                    bs_call["vega"],
                    bs_call["theta"],
                    bs_call["rho"]
                ]
            }
            st.dataframe(greeks_call_data, use_container_width=True)
            
        with tab2:
            greeks_put_data = {
                "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                "Valor": [
                    bs_put["delta"],
                    bs_put["gamma"],
                    bs_put["vega"],
                    bs_put["theta"],
                    bs_put["rho"]
                ]
            }
            st.dataframe(greeks_put_data, use_container_width=True)
            
        # Análisis de sensibilidad
        st.subheader("Análisis de Sensibilidad")
        param = st.selectbox("Parámetro para análisis", ["S", "K", "T", "r", "sigma"])
        
        # Rango de valores
        if param == "S":
            values = np.linspace(S * 0.5, S * 1.5, 50)
        elif param == "K":
            values = np.linspace(K * 0.5, K * 1.5, 50)
        elif param == "T":
            values = np.linspace(0.01, max(0.5, T * 2), 50)
        elif param == "r":
            values = np.linspace(0.0, 0.1, 50)
        elif param == "sigma":
            values = np.linspace(0.05, 0.5, 50)
        
        # Calcular precios para diferentes valores del parámetro
        call_prices = []
        put_prices = []
        
        for val in values:
            if param == "S":
                new_spec_call = OptionSpec(S=val, K=K, T=T, r=r, q=q, sigma=sigma, is_call=True)
                new_spec_put = OptionSpec(S=val, K=K, T=T, r=r, q=q, sigma=sigma, is_call=False)
            elif param == "K":
                new_spec_call = OptionSpec(S=S, K=val, T=T, r=r, q=q, sigma=sigma, is_call=True)
                new_spec_put = OptionSpec(S=S, K=val, T=T, r=r, q=q, sigma=sigma, is_call=False)
            elif param == "T":
                new_spec_call = OptionSpec(S=S, K=K, T=val, r=r, q=q, sigma=sigma, is_call=True)
                new_spec_put = OptionSpec(S=S, K=K, T=val, r=r, q=q, sigma=sigma, is_call=False)
            elif param == "r":
                new_spec_call = OptionSpec(S=S, K=K, T=T, r=val, q=q, sigma=sigma, is_call=True)
                new_spec_put = OptionSpec(S=S, K=K, T=T, r=val, q=q, sigma=sigma, is_call=False)
            elif param == "sigma":
                new_spec_call = OptionSpec(S=S, K=K, T=T, r=r, q=q, sigma=val, is_call=True)
                new_spec_put = OptionSpec(S=S, K=K, T=T, r=r, q=q, sigma=val, is_call=False)
            
            call_prices.append(bs_price(new_spec_call))
            put_prices.append(bs_price(new_spec_put))
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(values, call_prices, label='Call', linewidth=2)
        ax.plot(values, put_prices, label='Put', linewidth=2)
        ax.set_xlabel(param)
        ax.set_ylabel('Precio de la opción')
        ax.set_title(f'Sensibilidad del precio a {param}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error en cálculo: {e}")

elif modelo == "Binomial":
    st.header("📈 Modelo Binomial")
    
    # Teoría
    with st.expander("📚 Teoría del Modelo Binomial"):
        st.markdown("""
        El modelo de valoración de opciones binomial es un método numérico para la valoración de opciones.
        Fue desarrollado por Cox, Ross y Rubinstein en 1979.
        
        **Características principales:**
        - Modela el precio del activo subyacente como un proceso discreto
        - Puede valorar opciones americanas (con ejercicio anticipado)
        - Más flexible que Black-Scholes para ciertos tipos de opciones
        
        **Cómo funciona:**
        1. Divide el tiempo hasta el vencimiento en intervalos discretos
        2. En cada intervalo, el precio puede subir o bajar por factores determinados
        3. Calcula el valor de la opción working backwards desde el vencimiento
        4. Considera el ejercicio anticipado para opciones americanas
        
        **Ventajas:**
        - Puede manejar dividendos y ejercicio anticipado
        - Fácil de implementar computacionalmente
        - Conceptualmente intuitivo
        """)
    
    # Calcular precios
    try:
        binomial_call = binomial_price(spec_call, steps=binomial_steps, american=american_enabled)
        binomial_put = binomial_price(spec_put, steps=binomial_steps, american=american_enabled)
        
        # Calcular griegas
        binomial_call_greeks = binomial_greeks(spec_call, steps=binomial_steps, american=american_enabled)
        binomial_put_greeks = binomial_greeks(spec_put, steps=binomial_steps, american=american_enabled)
        
        # Mostrar precios
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Precio Call")
            st.metric("Valor", f"${binomial_call:.4f}")
            
        with col2:
            st.subheader("Precio Put")
            st.metric("Valor", f"${binomial_put:.4f}")
            
        st.info(f"Modelo Binomial con {binomial_steps} pasos - Opción {ejercicio}")
        
        # Mostrar griegas en tabs
        st.subheader("Griegas")
        tab1, tab2 = st.tabs(["Call", "Put"])
        
        with tab1:
            greeks_call_data = {
                "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                "Valor": [
                    binomial_call_greeks["delta"],
                    binomial_call_greeks["gamma"],
                    binomial_call_greeks["vega"],
                    binomial_call_greeks["theta"],
                    binomial_call_greeks["rho"]
                ]
            }
            st.dataframe(greeks_call_data, use_container_width=True)
            
        with tab2:
            greeks_put_data = {
                "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                "Valor": [
                    binomial_put_greeks["delta"],
                    binomial_put_greeks["gamma"],
                    binomial_put_greeks["vega"],
                    binomial_put_greeks["theta"],
                    binomial_put_greeks["rho"]
                ]
            }
            st.dataframe(greeks_put_data, use_container_width=True)
            
        # Análisis de convergencia
        st.subheader("Análisis de Convergencia")
        max_steps = st.slider("Máximo número de pasos para análisis", 10, 500, 200, 10)
        
        steps_range = range(10, max_steps + 1, max(1, max_steps // 20))
        call_prices_conv = []
        put_prices_conv = []
        
        for steps in steps_range:
            call_price = binomial_price(spec_call, steps=steps, american=american_enabled)
            put_price = binomial_price(spec_put, steps=steps, american=american_enabled)
            call_prices_conv.append(call_price)
            put_prices_conv.append(put_price)
        
        # Crear gráfico de convergencia
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(steps_range, call_prices_conv, label='Call', linewidth=2)
        ax1.set_xlabel('Número de pasos')
        ax1.set_ylabel('Precio Call')
        ax1.set_title('Convergencia del precio Call')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(steps_range, put_prices_conv, label='Put', color='orange', linewidth=2)
        ax2.set_xlabel('Número de pasos')
        ax2.set_ylabel('Precio Put')
        ax2.set_title('Convergencia del precio Put')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error en cálculo: {e}")

elif modelo == "Longstaff-Schwartz (LSM)":
    st.header("📈 Modelo Longstaff-Schwartz (LSM)")
    
    # Teoría
    with st.expander("📚 Teoría del Modelo Longstaff-Schwartz"):
        st.markdown("""
        El método Longstaff-Schwartz (LSM) es un enfoque de Monte Carlo para valorar opciones americanas.
        Desarrollado por Francis Longstaff y Eduardo Schwartz en 2001.
        
        **Características principales:**
        - Usa simulación Monte Carlo para generar paths de precios
        - Emplea regresión para estimar el valor de continuación
        - Evalúa optimalmente el ejercicio anticipado
        - Apropiado para opciones con múltiples fuentes de incertidumbre
        
        **Cómo funciona:**
        1. Simula múltiples paths de precios del activo subyacente
        2. Comienza desde el vencimiento y va hacia atrás en el tiempo
        3. En cada paso, usa regresión para estimar el valor de continuación
        4. Compara el valor de ejercicio inmediato con el valor de continuación
        5. Decide optimalmente si ejercer o no en cada punto
        
        **Ventajas:**
        - Puede manejar múltiples factores de riesgo
        - Flexible para diferentes tipos de opciones exóticas
        - Más preciso para opciones americanas con múltiples oportunidades de ejercicio
        """)
    
    # Calcular precios
    try:
        lsm_call = lsm_american_price(
            spec_call, 
            n_paths=lsm_paths, 
            n_steps=lsm_steps, 
            degree=lsm_degree,
            basis=basis
        )
        
        lsm_put = lsm_american_price(
            spec_put, 
            n_paths=lsm_paths, 
            n_steps=lsm_steps, 
            degree=lsm_degree,
            basis=basis
        )
        
        # Calcular griegas
        lsm_call_greeks = lsm_greeks(
            spec_call, 
            n_paths=lsm_paths, 
            n_steps=lsm_steps, 
            degree=lsm_degree,
            basis=basis
        )
        
        lsm_put_greeks = lsm_greeks(
            spec_put, 
            n_paths=lsm_paths, 
            n_steps=lsm_steps, 
            degree=lsm_degree,
            basis=basis
        )
        
        # Mostrar precios
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Precio Call")
            st.metric("Valor", f"${lsm_call:.4f}")
            
        with col2:
            st.subheader("Precio Put")
            st.metric("Valor", f"${lsm_put:.4f}")
            
        st.info(f"Modelo LSM con {lsm_paths} paths, {lsm_steps} steps, grado {lsm_degree}, base {basis}")
        
        # Mostrar griegas en tabs
        st.subheader("Griegas")
        tab1, tab2 = st.tabs(["Call", "Put"])
        
        with tab1:
            greeks_call_data = {
                "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                "Valor": [
                    lsm_call_greeks["delta"],
                    lsm_call_greeks["gamma"],
                    lsm_call_greeks["vega"],
                    lsm_call_greeks["theta"],
                    lsm_call_greeks["rho"]
                ]
            }
            st.dataframe(greeks_call_data, use_container_width=True)
            
        with tab2:
            greeks_put_data = {
                "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                "Valor": [
                    lsm_put_greeks["delta"],
                    lsm_put_greeks["gamma"],
                    lsm_put_greeks["vega"],
                    lsm_put_greeks["theta"],
                    lsm_put_greeks["rho"]
                ]
            }
            st.dataframe(greeks_put_data, use_container_width=True)
            
        # Simular algunos paths para visualización
        st.subheader("Simulación de Paths")
        from functions import _simulate_gbm_paths
        
        # Simular paths para visualización (solo unos pocos para no saturar)
        paths = _simulate_gbm_paths(
            S0=S, r=r, q=q, sigma=sigma, T=T, 
            n_paths=10, n_steps=lsm_steps, seed=12345, antithetic=False
        )
        
        # Crear gráfico de paths
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(paths.shape[0]):
            ax.plot(np.linspace(0, T, lsm_steps + 1), paths[i, :], alpha=0.7, linewidth=1)
        
        ax.axhline(y=K, color='r', linestyle='--', label=f'Precio de ejercicio (K={K})')
        ax.set_xlabel('Tiempo (años)')
        ax.set_ylabel('Precio del activo')
        ax.set_title('Ejemplo de paths simulados')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Análisis de sensibilidad al número de paths
        st.subheader("Análisis de Convergencia")
        max_paths = min(50000, lsm_paths * 2)
        paths_range = range(1000, max_paths + 1, max(1, max_paths // 10))
        
        if st.button("Ejecutar análisis de convergencia (puede tardar)"):
            call_prices_conv = []
            put_prices_conv = []
            progress_bar = st.progress(0)
            
            for i, n_paths in enumerate(paths_range):
                call_price = lsm_american_price(
                    spec_call, 
                    n_paths=n_paths, 
                    n_steps=lsm_steps, 
                    degree=lsm_degree,
                    basis=basis,
                    seed=12345  # Semilla fija para comparabilidad
                )
                put_price = lsm_american_price(
                    spec_put, 
                    n_paths=n_paths, 
                    n_steps=lsm_steps, 
                    degree=lsm_degree,
                    basis=basis,
                    seed=12345  # Semilla fija para comparabilidad
                )
                call_prices_conv.append(call_price)
                put_prices_conv.append(put_price)
                progress_bar.progress((i + 1) / len(paths_range))
            
            # Crear gráfico de convergencia
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            ax1.plot(paths_range, call_prices_conv, label='Call', linewidth=2)
            ax1.set_xlabel('Número de paths')
            ax1.set_ylabel('Precio Call')
            ax1.set_title('Convergencia del precio Call')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(paths_range, put_prices_conv, label='Put', color='orange', linewidth=2)
            ax2.set_xlabel('Número de paths')
            ax2.set_ylabel('Precio Put')
            ax2.set_title('Convergencia del precio Put')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error en cálculo: {e}")

# Notas al pie
st.divider()
st.caption("""
*Nota: Esta aplicación es para fines educativos y de análisis. 
Los resultados no deben considerarse como recomendaciones de inversión.*
""")
