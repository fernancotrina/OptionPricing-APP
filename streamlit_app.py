import streamlit as st
import numpy as np
import pandas as pd
from functions import (
    bs_price, bs_greeks, bi_greeks, 
    plot_binomial_convergence, 
    plot_bs_surface
)

# Configuración de la página
st.set_page_config(
    page_title="Analizador Avanzado de Opciones",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Analizador Avanzado de Opciones")

# Barra lateral para selección del modelo
with st.sidebar:
    st.header("🔧 Configuración")
    
    # Selección del modelo
    modelo = st.selectbox(
        "Selecciona el modelo de valuación:",
        ["Black-Scholes", "Binomial"])
    
    st.header("📋 Parámetros Comunes")
    
    # Inputs básicos
    S = st.number_input("Precio Spot (S)", value=100.0, min_value=0.01, step=1.0)
    K = st.number_input("Precio Ejercicio (K)", value=100.0, min_value=0.01, step=1.0)
    T = st.number_input("Tiempo (T años)", value=0.25, min_value=0.0, max_value=50.0, step=0.05)
    r = st.number_input("Tasa Libre Riesgo (r)", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
    sigma = st.number_input("Volatilidad (σ)", value=0.2, min_value=0.01, max_value=2.0, step=0.01)

    # Parámetros específicos por modelo
    st.divider()
    st.header("⚙️ Parámetros Específicos")

    q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, max_value=1.0, step=0.01)  

    if modelo == "Binomial":
        binomial_steps = st.slider("Número de pasos", 10, 500, 100, 10)


# Modelo Black-Scholes
if modelo == "Black-Scholes":
    st.header("📈 Modelo Black-Scholes")
    
    # Teoría
    with st.expander("📚 Teoría: Black-Scholes-Merton"):
        st.markdown(r"""
        El modelo **Black-Scholes-Merton (1973)** es la solución analítica a una Ecuación Diferencial Parcial (PDE) que simula una cartera libre de riesgo mediante cobertura dinámica (*delta hedging*).

        

        **1. La Ecuación Fundamental (PDE)**
        El precio $V(S,t)$ de cualquier derivado debe satisfacer la siguiente ecuación para evitar el arbitraje:
        $$
        \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q)S \frac{\partial V}{\partial S} - rV = 0
        $$
        Esta ecuación establece que el rendimiento de la opción cubierta debe ser igual a la tasa libre de riesgo.

        **2. Solución Analítica (Merton - Dividendos Continuos)**
        Bajo la medida neutral al riesgo $\mathbb{Q}$, el precio es la esperanza descontada del payoff:
        
        $$
        C = \underbrace{S_0 e^{-qT} N(d_1)}_{\text{Valor esperado del Activo}} - \underbrace{K e^{-rT} N(d_2)}_{\text{Valor esperado del Pago}}
        $$

        **3. Interpretación de los Términos (La "Intuición")**
        * **$N(d_2)$**: Es la **probabilidad de ejercicio** en el mundo neutral al riesgo ($Prob(S_T > K)$).
        * **$N(d_1)$**: Es el **Delta ($\Delta$)** de la opción (para activos sin dividendos). Representa la cantidad de activo subyacente que debes comprar para cubrir la opción.
        * **$e^{-qT}$ y $e^{-rT}$**: Factores de descuento. El activo se descuenta a la tasa de dividendos ($q$) y el strike a la tasa libre de riesgo ($r$).

        **Cálculo de $d_1$ y $d_2$:**
        $$
        d_1 = \frac{\ln(S_0/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}
        $$
        $$
        d_2 = d_1 - \sigma\sqrt{T}
        $$
        
        *Nota: El modelo asume que los retornos logarítmicos del activo distribuyen normalmente (el precio sigue una distribución Lognormal).*
        """)

    try:
        # Calcular precios y griegas
        bs_call = bs_greeks(S=S, K=K, T=T, r=r, q=q, sigma=sigma, is_call=True)
        bs_put = bs_greeks(S=S, K=K, T=T, r=r, q=q, sigma=sigma, is_call=False)
        
        # Mostrar precios
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Precio Call")
            st.metric("Valor", f"${bs_call['price']:.4f}")
            
        with col2:
            st.subheader("Precio Put")
            st.metric("Valor", f"${bs_put['price']:.4f}")
        

        # Mostrar griegas
        st.divider()
        st.subheader("Griegas")

        col1, col2 = st.columns([1, 1])

        with col1:
                st.markdown("### 📈 Call Option")

                df_call = pd.DataFrame({
                    "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                    "Valor": [
                        bs_call["delta"],
                        bs_call["gamma"],
                        bs_call["vega"],
                        bs_call["theta"],
                        bs_call["rho"]
                    ]
                })

                st.dataframe(df_call, use_container_width=True)

        with col2:
                st.markdown("### 📉 Put Option")

                df_put = pd.DataFrame({
                    "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                    "Valor": [
                        bs_put["delta"],
                        bs_put["gamma"],
                        bs_put["vega"],
                        bs_put["theta"],
                        bs_put["rho"]
                    ]
                })

                st.dataframe(df_put, use_container_width=True)

        with st.expander("### 📘 Interpretación de las Griegas"):
            st.markdown("""
            **Delta (Δ)**  
            Mide cuánto cambia el precio de la opción ante un cambio unitario en el precio del activo subyacente.

            *Ejemplo:* Si Δ = 0.60 y el precio de la acción sube de 100 a 101, el precio de la opción aumenta aproximadamente en 0.60.

            ---

            **Gamma (Γ)**  
            Mide cómo cambia el Delta cuando el precio del subyacente varía. Captura la convexidad de la opción.

            *Ejemplo:* Si Γ = 0.05 y el Delta actual es 0.60, una subida de 1 unidad en el subyacente hará que el Delta pase a aproximadamente 0.65.

            ---

            **Vega (ν)**  
            Mide la sensibilidad del precio de la opción ante cambios en la volatilidad implícita.

            *Ejemplo:* Si Vega = 0.12 y la volatilidad implícita aumenta de 20% a 21%, el precio de la opción sube aproximadamente en 0.12.

            ---

            **Theta (Θ)**  
            Mide el cambio en el precio de la opción debido al paso del tiempo, manteniendo todo lo demás constante.

            *Ejemplo:* Si Θ = −0.03, la opción pierde aproximadamente 0.03 de valor por cada año que pasa (o 0.03/365 por día).

            ---

            **Rho (ρ)**  
            Mide la sensibilidad del precio de la opción ante cambios en la tasa de interés libre de riesgo.

            *Ejemplo:* Si ρ = 0.08 y la tasa libre de riesgo sube de 5% a 6%, el precio de la opción aumenta aproximadamente en 0.08.
            """)

        # Gráfico de Price Surface
        st.divider()
        st.header("🏔️ Superficies de Precio")

        st.markdown("Visualización de cómo cambia el precio de la opción variando simultáneamente el Precio Spot y la Volatilidad.")

        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            st.subheader("Superficie CALL")
            fig_call = plot_bs_surface(S, K, T, r, q, sigma, is_call=True)
            st.plotly_chart(fig_call, use_container_width=True)

        with col_graph2:
            st.subheader("Superficie PUT")
            fig_put = plot_bs_surface(S, K, T, r, q, sigma, is_call=False)
            st.plotly_chart(fig_put, use_container_width=True)

    except Exception as e:
        st.error(f"Error en cálculo: {e}")


# Modelo CRR
elif modelo == "Binomial":
    st.header("🌳 Modelo Binomial (Cox-Ross-Rubinstein)")
    
    # Teoría
    with st.expander("📚 Teoría: Modelo Binomial (CRR)"):
        st.markdown(r"""
        El modelo **Cox-Ross-Rubinstein (CRR)** es un método numérico de tiempo discreto que modela la dinámica del precio del activo como un camino aleatorio (Random Walk).

        **1. Dinámica del Activo (Rejilla Binomial)**
        Dividimos el tiempo $T$ en $N$ intervalos de longitud $\Delta t = T/N$. En cada paso, el precio $S$ solo puede moverse a dos estados:
        * **Up ($u$):** $S_{t+1} = S_t \cdot u$
        * **Down ($d$):** $S_{t+1} = S_t \cdot d$
        
        Para que el modelo converja a la distribución log-normal (Black-Scholes) cuando $N \to \infty$, los parámetros se calibran basándose en la volatilidad $\sigma$:
        $$
        u = e^{\sigma \sqrt{\Delta t}}, \quad d = \frac{1}{u} = e^{-\sigma \sqrt{\Delta t}}
        $$

        **2. Probabilidad Neutral al Riesgo ($p$)**
        Es la probabilidad teórica bajo la cual el rendimiento esperado del activo es igual a la tasa libre de riesgo ($r$). No es la probabilidad real del mercado, sino una construcción de "no arbitraje":
        $$
        p = \frac{e^{(r-q)\Delta t} - d}{u - d}
        $$
        *(Donde $q$ es el dividend yield)*.

        **3. Valoración por Inducción hacia Atrás (Backward Induction)**
        El precio de la opción se calcula desde el vencimiento ($T$) hacia el presente ($0$).
        
        En un nodo cualquiera del tiempo $t$, el valor de una **Opción Americana** $V_t$ es el máximo entre ejercerla o mantenerla (Valor de Continuación):
        
        $$
        V_t = \max \Bigg( \underbrace{\text{Payoff}(S_t)}_{\text{Ejercer}}, \quad \underbrace{e^{-r \Delta t} [p V_{u} + (1-p) V_{d}]}_{\text{Continuar (Esperanza descontada)}} \Bigg)
        $$
        
        Esta condición de maximización ($\max$) en cada nodo es lo que hace al modelo Binomial superior a Black-Scholes para opciones Americanas, ya que captura la prima por el derecho al ejercicio temprano.
        """)

    # Tipo de Ejercicio - Americana por defecto
    is_american = st.toggle("Estilo Americano", value=True)
    
    try:
        # Calcular precios y griegas
        res_call = bi_greeks(S, K, T, r, q, sigma, N=binomial_steps, is_call=True, is_american=is_american)
        res_put = bi_greeks(S, K, T, r, q, sigma, N=binomial_steps, is_call=False, is_american=is_american)
        
        # Mostrar precios y griegeas
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Call {'Americana' if is_american else 'Europea'}")
            st.metric("Precio", f"${res_call['price']:.4f}")
            st.dataframe(pd.DataFrame(res_call, index=["Valor"]).T.drop("price"), use_container_width=True)
            
        with col2:
            st.subheader(f"Put {'Americana' if is_american else 'Europea'}")
            st.metric("Precio", f"${res_put['price']:.4f}")
            st.dataframe(pd.DataFrame(res_put, index=["Valor"]).T.drop("price"), use_container_width=True)

        with st.expander("### 📘 Interpretación de las Griegas"):
            st.markdown("""
            **Delta (Δ)**  
            Mide cuánto cambia el precio de la opción ante un cambio unitario en el precio del activo subyacente.

            *Ejemplo:* Si Δ = 0.60 y el precio de la acción sube de 100 a 101, el precio de la opción aumenta aproximadamente en 0.60.

            ---

            **Gamma (Γ)**  
            Mide cómo cambia el Delta cuando el precio del subyacente varía. Captura la convexidad de la opción.

            *Ejemplo:* Si Γ = 0.05 y el Delta actual es 0.60, una subida de 1 unidad en el subyacente hará que el Delta pase a aproximadamente 0.65.

            ---

            **Vega (ν)**  
            Mide la sensibilidad del precio de la opción ante cambios en la volatilidad implícita.

            *Ejemplo:* Si Vega = 0.12 y la volatilidad implícita aumenta de 20% a 21%, el precio de la opción sube aproximadamente en 0.12.

            ---

            **Theta (Θ)**  
            Mide el cambio en el precio de la opción debido al paso del tiempo, manteniendo todo lo demás constante.

            *Ejemplo:* Si Θ = −0.03, la opción pierde aproximadamente 0.03 de valor por cada año que pasa (o 0.03/365 por día).

            ---

            **Rho (ρ)**  
            Mide la sensibilidad del precio de la opción ante cambios en la tasa de interés libre de riesgo.

            *Ejemplo:* Si ρ = 0.08 y la tasa libre de riesgo sube de 5% a 6%, el precio de la opción aumenta aproximadamente en 0.08.
            """)


        # Gráfico de Convergencia
        st.divider()
        st.header("📊 Análisis Gráfico")
        
        tipo_visualizacion = st.radio(
            "¿Qué opción quieres analizar en los gráficos?",
            ["Put", "Call"],
            index=0,
            horizontal=True
        )
        ver_call = (tipo_visualizacion == "Call")
        
        st.markdown(f"Observa cómo el precio de la **{tipo_visualizacion}** converge al aumentar los pasos.")
            
        fig_conv = plot_binomial_convergence(
                S, K, T, r, q, sigma, 
                is_call=ver_call, 
                is_american=is_american
            )
        st.plotly_chart(fig_conv, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error en cálculo binomial: {e}")
