import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Configuración de la interfaz y la página
st.set_page_config(
    page_title="Modelo de Valuación y Optimización de Portafolios (Markowitz)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos corporativos / ejecutivos
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .subtitle {
        font-size: 16px;
        color: #475569;
        margin-bottom: 25px;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        color: #1E3A8A;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 5px;
    }
    .kpi-box {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 13px;
        font-weight: bold;
        color: #64748B;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E293B;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Simulador Financiero y Frontera Eficiente de Markowitz</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Réplica digital interactiva del modelo de optimización de portafolios y control estadístico de riesgos corporativos</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# 1. CONFIGURACIÓN DEL PORTAFOLIO (BARRA LATERAL)
# ------------------------------------------------------------------
st.sidebar.header("💼 1. Configuración General")

# Tickers por defecto basados en el modelo solicitado
tickers_default = "AAPL, BTC-USD, GC=F, ^GSPC, HSBC"
tickers_input = st.sidebar.text_input(
    "Códigos de Pizarra (Separados por comas)",
    value=tickers_default,
    help="Ingrese los tickers válidos en Yahoo Finance. Ejemplo: AAPL, BTC-USD, GC=F"
)

# Rango de fechas para histórico
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    fecha_inicio = st.date_input("Fecha de Inicio", datetime.now() - timedelta(days=365*4))
with col_d2:
    fecha_fin = st.date_input("Fecha de Fin", datetime.now())

# Variables financieras iniciales
col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    tasa_libre_riesgo = st.number_input("Tasa Libre Riesgo (%)", value=4.5, min_value=0.0, step=0.1) / 100
with col_p2:
    capital_inicial = st.number_input("Capital Inicial ($)", value=10000.0, min_value=1.0, step=1000.0)

# ------------------------------------------------------------------
# 2. RESTRICCIONES DEL MODELO (BARRA LATERAL)
# ------------------------------------------------------------------
st.sidebar.header("⚖️ 2. Restricciones del Modelo")

# Límites de asignación por activo
col_w1, col_w2 = st.sidebar.columns(2)
with col_w1:
    peso_minimo = st.number_input("Peso Mín por Activo", value=0.00, min_value=0.0, max_value=1.0, step=0.05)
with col_w2:
    peso_maximo = st.number_input("Peso Máx por Activo", value=1.00, min_value=0.0, max_value=1.0, step=0.05)

# Filtros y restricciones de métricas de portafolio
activar_restricciones_avanzadas = st.sidebar.checkbox("Activar Filtros de Métricas Objetivo")

target_ret_min = -1.0
target_vol_max = 1.0
target_sharpe_min = -100.0

if activar_restricciones_avanzadas:
    st.sidebar.subheader("Límites de Frontera")
    target_ret_min = st.sidebar.number_input("Rentabilidad Anualizada Mín (%)", value=12.0, step=1.0) / 100
    target_vol_max = st.sidebar.number_input("Volatilidad Anualizada Máx (%)", value=20.0, step=1.0) / 100
    target_sharpe_min = st.sidebar.number_input("Ratio de Sharpe Mínimo", value=0.5, step=0.1)

# Procesar inputs de texto de tickers
lista_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# ------------------------------------------------------------------
# FUNCIONES FINANCIERAS Y MATEMÁTICAS
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
def descargar_datos_historicos(tickers, start, end):
    try:
        df_precios = yf.download(tickers, start=start, end=end)['Adj Close']
        if isinstance(df_precios, pd.Series):
            df_precios = df_precios.to_frame(name=tickers[0])
        return df_precios.dropna()
    except Exception as e:
        st.error(f"Error al descargar datos de Yahoo Finance: {e}")
        return pd.DataFrame()

def calcular_metricas_estadisticas(df_precios, rf, benchmark):
    # Rendimientos diarios
    df_rendimientos = df_precios.pct_change().dropna()
    
    # Métricas individuales por activo
    rend_diario_prom = df_rendimientos.mean()
    vol_diaria = df_rendimientos.std()
    
    rend_anualizado = rend_diario_prom * 252
    vol_anualizada = vol_diaria * np.sqrt(252)
    sharpe_individual = (rend_anualizado - rf) / vol_anualizada
    
    # Covarianza y Correlación
    matriz_cov = df_rendimientos.cov()
    matriz_corr = df_rendimientos.corr()
    
    # Cálculo de Beta respecto al activo benchmark o mercado de referencia
    dic_betas = {}
    if benchmark in df_rendimientos.columns:
        var_mercado = df_rendimientos[benchmark].var()
        for activo in df_rendimientos.columns:
            if var_mercado > 0:
                dic_betas[activo] = df_rendimientos[activo].cov(df_rendimientos[benchmark]) / var_mercado
            else:
                dic_betas[activo] = 1.0
    else:
        rend_mercado_proxy = df_rendimientos.mean(axis=1)
        var_mercado = rend_mercado_proxy.var()
        for activo in df_rendimientos.columns:
            dic_betas[activo] = df_rendimientos[activo].cov(rend_mercado_proxy) / var_mercado if var_mercado > 0 else 1.0

    # Value at Risk (VaR 95% Paramétrico Diario)
    dic_var_95 = {}
    for activo in df_rendimientos.columns:
        dic_var_95[activo] = -(rend_diario_prom[activo] - 1.645 * vol_diaria[activo])

    # DataFrame unificado de Indicadores
    df_indicadores = pd.DataFrame({
        'Rentabilidad Diaria': rend_diario_prom,
        'Rentabilidad Anualizado': rend_anualizado,
        'Volatilidad Diaria': vol_diaria,
        'Volatilidad Anualizada': vol_anualizada,
        'Índice de Sharpe': sharpe_individual,
        'Beta': pd.Series(dic_betas),
        'VaR Diario (95%)': pd.Series(dic_var_95)
    })
    
    return df_rendimientos, df_indicadores, matriz_cov, matriz_corr

def obtener_metricas_portafolio(pesos, df_rendimientos, matriz_cov, rf):
    pesos = np.array(pesos)
    ret_port = np.sum(df_rendimientos.mean() * pesos) * 252
    vol_port = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov * 252, pesos)))
    sharpe_port = (ret_port - rf) / vol_port if vol_port > 0 else 0
    return ret_port, vol_port, sharpe_port

# Funciones de optimización
def objetivo_min_volatilidad(pesos, df_rendimientos, matriz_cov, rf):
    return obtener_metricas_portafolio(pesos, df_rendimientos, matriz_cov, rf)[1]

def objetivo_max_sharpe(pesos, df_rendimientos, matriz_cov, rf):
    return -obtener_metricas_portafolio(pesos, df_rendimientos, matriz_cov, rf)[2]

def ejecutar_optimizador(df_rendimientos, matriz_cov, rf, min_w, max_w, criterio='sharpe',
                         r_min=-1.0, v_max=1.0, s_min=-100.0, usar_restricciones=False):
    num_activos = len(df_rendimientos.columns)
    pesos_iniciales = num_activos * [1.0 / num_activos]
    limites_activos = tuple((min_w, max_w) for _ in range(num_activos))
    
    # Restricción base: La suma de todos los pesos es igual a 1 (100% invertido)
    restricciones = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    if usar_restricciones:
        restricciones.append({'type': 'ineq', 'fun': lambda x: obtener_metricas_portafolio(x, df_rendimientos, matriz_cov, rf)[0] - r_min})
        restricciones.append({'type': 'ineq', 'fun': lambda x: v_max - obtener_metricas_portafolio(x, df_rendimientos, matriz_cov, rf)[1]})
        restricciones.append({'type': 'ineq', 'fun': lambda x: obtener_metricas_portafolio(x, df_rendimientos, matriz_cov, rf)[2] - s_min})

    if criterio == 'sharpe':
        resultado = sco.minimize(objetivo_max_sharpe, pesos_iniciales, args=(df_rendimientos, matriz_cov, rf),
                                 method='SLSQP', bounds=limites_activos, constraints=restricciones)
    else:
        resultado = sco.minimize(objetivo_min_volatilidad, pesos_iniciales, args=(df_rendimientos, matriz_cov, rf),
                                 method='SLSQP', bounds=limites_activos, constraints=restricciones)
    return resultado

# ------------------------------------------------------------------
# CONTROLADOR Y CONTROL DE FLUJO
# ------------------------------------------------------------------
if not lista_tickers:
    st.info("💡 Por favor, introduzca una lista de tickers válidos en el panel lateral.")
else:
    precios = descargar_datos_historicos(lista_tickers, fecha_inicio, fecha_fin)
    
    if precios.empty:
        st.warning("⚠️ No se recuperaron datos de la API. Compruebe los símbolos e intente de nuevo.")
    else:
        # Selección del benchmark de mercado para el cálculo dinámico de la Beta
        st.sidebar.subheader("🎯 Parámetros de Mercado")
        ticker_mercado = st.sidebar.selectbox("Elegir Benchmark de Mercado (Beta)", options=list(precios.columns), index=len(precios.columns)-1)
        
        # Ejecutar cálculos matemáticos básicos
        rendimientos, df_indicadores, cov_matrix, corr_matrix = calcular_metricas_estadisticas(precios, tasa_libre_riesgo, ticker_mercado)
        
        # Procesar optimizaciones fundamentales
        res_opt_sharpe = ejecutar_optimizador(rendimientos, cov_matrix, tasa_libre_riesgo, peso_minimo, peso_maximo, 'sharpe',
                                               target_ret_min, target_vol_max, target_sharpe_min, activar_restricciones_avanzadas)
        
        res_opt_vol = ejecutar_optimizador(rendimientos, cov_matrix, tasa_libre_riesgo, peso_minimo, peso_maximo, 'volatilidad',
                                            target_ret_min, target_vol_max, target_sharpe_min, activar_restricciones_avanzadas)
        
        # Si la optimización con restricciones estrictas falla, recalculamos sin filtros para mantener operabilidad
        if not res_opt_sharpe.success and activar_restricciones_avanzadas:
            st.error("❌ El optimizador no pudo encontrar una mezcla que cumpla con los límites de métricas avanzadas. Mostrando óptimos estándar.")
            res_opt_sharpe = ejecutar_optimizador(rendimientos, cov_matrix, tasa_libre_riesgo, peso_minimo, peso_maximo, 'sharpe')
            res_opt_vol = ejecutar_optimizador(rendimientos, cov_matrix, tasa_libre_riesgo, peso_minimo, peso_maximo, 'volatilidad')

        # Extraer métricas de los portafolios óptimos hallados
        pesos_sh = res_opt_sharpe.x
        pesos_vol = res_opt_vol.x
        
        r_sh, v_sh, s_sh = obtener_metricas_portafolio(pesos_sh, rendimientos, cov_matrix, tasa_libre_riesgo)
        r_vol, v_vol, s_vol = obtener_metricas_portafolio(pesos_vol, rendimientos, cov_matrix, tasa_libre_riesgo)
        
        # Selector del portafolio objetivo para la vista detallada
        st.markdown('<div class="section-header">Selección de Portafolio Objetivo para Análisis Estratégico</div>', unsafe_allow_html=True)
        seleccion_portafolio = st.radio(
            "Seleccione qué composición desea evaluar y modelar en el panel de control:",
            ["Máximo Ratio de Sharpe (Mezcla Eficiente Tangente)", "Mínima Volatilidad (Riesgo Mínimo Global)"],
            horizontal=True
        )
        
        if seleccion_portafolio == "Máximo Ratio de Sharpe (Mezcla Eficiente Tangente)":
            pesos_elegidos = pesos_sh
            ret_final, vol_final, sharpe_final = r_sh, v_sh, s_sh
            st.success("🎯 Se ha configurado el portafolio optimizado bajo la maximización del Ratio de Sharpe.")
        else:
            pesos_elegidos = pesos_vol
            ret_final, vol_final, sharpe_final = r_vol, v_vol, s_vol
            st.info("🛡️ Se ha configurado el portafolio optimizado bajo la minimización del riesgo del portafolio.")

        # Cálculo de VaR Paramétrico del Portafolio unificado
        rend_diario_port = np.sum(rendimientos.mean() * pesos_elegidos)
        vol_diaria_port = (vol_final / np.sqrt(252))
        var_diario_port_95 = -(rend_diario_port - 1.645 * vol_diaria_port)
        valor_var_capital = var_diario_port_95 * capital_inicial

        # ------------------------------------------------------------------
        # 3. VISUALIZACIÓN DE CUADROS DE MANDO (KPIS)
        # ------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.markdown(f'<div class="kpi-box"><div class="kpi-title">Rentabilidad Esperada Anual</div><div class="kpi-value">{ret_final*100:.2f}%</div></div>', unsafe_allow_html=True)
        with col_k2:
            st.markdown(f'<div class="kpi-box"><div class="kpi-title">Volatilidad Anual (Riesgo)</div><div class="kpi-value">{vol_final*100:.2f}%</div></div>', unsafe_allow_html=True)
        with col_k3:
            st.markdown(f'<div class="kpi-box"><div class="kpi-title">Índice de Sharpe</div><div class="kpi-value">{sharpe_final:.2f}</div></div>', unsafe_allow_html=True)
        with col_k4:
            st.markdown(f'<div class="kpi-box"><div class="kpi-title">VaR Diario del Portafolio (95%)</div><div class="kpi-value">{var_diario_port_95*100:.2f}% (${valor_var_capital:,.2f})</div></div>', unsafe_allow_html=True)

        # Separación en pestañas ejecutivas de trabajo
        tab_pesos, tab_curva, tab_matrices = st.tabs([
            "📊 Pesos Asignados e Indicadores", 
            "📈 Frontera Eficiente de Markowitz", 
            "🗂️ Análisis de Covarianzas y Correlación"
        ])
        
        with tab_pesos:
            c1, c2 = st.columns([4, 5])
            with c1:
                st.subheader("Distribución Óptima de Capital")
                df_distribucion = pd.DataFrame({
                    'Activo Pizarra': precios.columns,
                    'Peso Porcentual': pesos_elegidos,
                    'Asignación Capital ($)': pesos_elegidos * capital_inicial
                })
                df_distribucion['Peso Porcentual'] = df_distribucion['Peso Porcentual'].map(lambda x: f"{x*100:.2f}%")
                df_distribucion['Asignación Capital ($)'] = df_distribucion['Asignación Capital ($)'].map(lambda x: f"${x:,.2f}")
                st.dataframe(df_distribucion, use_container_width=True, hide_index=True)
            with c2:
                st.subheader("Métricas Estadísticas e Indicadores por Activo")
                st.dataframe(df_indicadores.style.format({
                    'Rentabilidad Diaria': '{:.4f}%',
                    'Rentabilidad Anualizado': '{:.2f}%',
                    'Volatilidad Diaria': '{:.4f}%',
                    'Volatilidad Anualizada': '{:.2f}%',
                    'Índice de Sharpe': '{:.2f}',
                    'Beta': '{:.2f}',
                    'VaR Diario (95%)': '{:.2f}%'
                }), use_container_width=True)

        with tab_curva:
            # Generación de portafolios simulados aleatorios para rellenar el espacio
            num_simulaciones = 400
            sim_ret = np.zeros(num_simulaciones)
            sim_vol = np.zeros(num_simulaciones)
            sim_sharpe = np.zeros(num_simulaciones)
            
            for k in range(num_simulaciones):
                w_sim = np.random.random(len(precios.columns))
                w_sim /= np.sum(w_sim)
                r_s, v_s, s_s = obtener_metricas_portafolio(w_sim, rendimientos, cov_matrix, tasa_libre_riesgo)
                sim_ret[k] = r_s
                sim_vol[k] = v_s
                sim_sharpe[k] = s_s

            # Curva de frontera matemática óptima
            rendimientos_objetivo = np.linspace(df_indicadores['Rentabilidad Anualizado'].min(), df_indicadores['Rentabilidad Anualizado'].max(), 25)
            volatilidades_frontera = []
            
            for r_obj in rendimientos_objetivo:
                restricciones_curva = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: obtener_metricas_portafolio(x, rendimientos, cov_matrix, tasa_libre_riesgo)[0] - r_obj}
                ]
                sol_c = sco.minimize(objetivo_min_volatilidad, len(precios.columns)*[1./len(precios.columns)],
                                     args=(rendimientos, cov_matrix, tasa_libre_riesgo), method='SLSQP',
                                     bounds=tuple((peso_minimo, peso_maximo) for _ in range(len(precios.columns))), constraints=restricciones_curva)
                if sol_c.success:
                    volatilidades_frontera.append(sol_c.fun)
                else:
                    volatilidades_frontera.append(None)

            # Gráfico de Frontera Eficiente mediante Plotly
            fig_frontera = go.Figure()
            
            # Dibujar portafolios simulados
            fig_frontera.add_trace(go.Scatter(
                x=sim_vol, y=sim_ret, mode='markers',
                marker=dict(color=sim_sharpe, colorscale='Viridis', showscale=True, size=5, title="Sharpe"),
                name='Portafolios Aleatorios', opacity=0.3
            ))
            
            # Dibujar línea teórica de frontera eficiente
            puntos_validos = [(v, r) for v, r in zip(volatilidades_frontera, rendimientos_objetivo) if v is not None]
            if puntos_validos:
                v_line, r_line = zip(*puntos_validos)
                fig_frontera.add_trace(go.Scatter(x=v_line, y=r_line, mode='lines', line=dict(color='#1E293B', width=2.5, dash='dash'), name='Frontera Eficiente'))

            # Resaltar Máximo Sharpe
            fig_frontera.add_trace(go.Scatter(
                x=[v_sh], y=[r_sh], mode='markers',
                marker=dict(color='#EF4444', size=13, symbol='star'),
                name='Máximo Sharpe (Tangente)'
            ))
            
            # Resaltar Mínima Volatilidad
            fig_frontera.add_trace(go.Scatter(
                x=[v_vol], y=[r_vol], mode='markers',
                marker=dict(color='#F59E0B', size=13, symbol='diamond'),
                name='Mínima Volatilidad'
            ))

            fig_frontera.update_layout(
                title='Frontera Eficiente de Markowitz y Universos Combinatorios',
                xaxis_title='Volatilidad Anualizada (Riesgo)',
                yaxis_title='Rentabilidad Anualizada Esperada',
                template='plotly_white',
                legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_frontera, use_container_width=True)

            # Gráfico de Torta / Distribución del portafolio actual
            fig_torta = px.pie(
                names=precios.columns,
                values=pesos_elegidos,
                title=f'Composición Sectorial del Portafolio ({seleccion_portafolio})',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Slate
            )
            st.plotly_chart(fig_torta, use_container_width=True)

        with tab_matrices:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.subheader("Matriz de Covarianzas (Anualizada)")
                st.dataframe(cov_matrix * 252, use_container_width=True)
            with col_m2:
                st.subheader("Mapa de Correlación de Pearson")
                fig_calor = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title="Análisis Colectivo de Correlación Cruzada"
                )
                st.plotly_chart(fig_calor, use_container_width=True)