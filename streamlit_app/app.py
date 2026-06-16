# -*- coding: utf-8 -*-


import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. CARREGANDO A IA E OS PARÂMETROS ---
@st.cache_resource
def carregar_arquivos():
    # Avisamos ao Python que os arquivos estão dentro da pasta 'models/'
    modelo = joblib.load('models/modelo_isolation_forest.pkl')
    scaler = joblib.load('models/scaler_os.pkl')
    limiar = joblib.load('models/limiar_otimizado.pkl')
    return modelo, scaler, limiar

modelo, scaler, limiar = carregar_arquivos()

# --- 2. DICIONÁRIOS DE MAPEAMENTO (UX) ---
# Traduz o texto visual amigável para o número exato que a IA aprendeu
ESTADOS_DICT = {
    "Acre (AC)": 0, "Alagoas (AL)": 1, "Amazonas (AM)": 2, "Bahia (BA)": 3,
    "Ceará (CE)": 4, "Distrito Federal (DF)": 5, "Espírito Santo (ES)": 6,
    "Goiás (GO)": 7, "Maranhão (MA)": 8, "Mato Grosso (MT)": 9,
    "Mato Grosso do Sul (MS)": 10, "Minas Gerais (MG)": 11, "Pará (PA)": 12,
    "Paraíba (PB)": 13, "Paraná (PR)": 14, "Pernambuco (PE)": 15,
    "Piauí (PI)": 16, "Rio de Janeiro (RJ)": 17, "Rio Grande do Norte (RN)": 18,
    "Rio Grande do Sul (RS)": 19, "Rondônia (RO)": 20, "Roraima (RR)": 21,
    "Santa Catarina (SC)": 22, "São Paulo (SP)": 23, "Sergipe (SE)": 24,
    "Tocantins (TO)": 25
}

LINHAS_DICT = {
    "Chopeira": 0,
    "Congelador": 1,
    "Máquina de Café": 2,
    "Pit Stop": 3,
    "Post Mix": 4,
    "Pré Resfriador": 5,
    "Refrigerador": 6,
    "Vending Machine": 7
}

# --- 3. INTERFACE DO USUÁRIO ---
st.title("Detector de Anomalias em Ordens de Serviço")
st.write("Insira os dados da Ordem de Serviço para verificar o risco de anomalia.")

# Criando os campos para o usuário digitar
col1, col2 = st.columns(2)
with col1:
    total_pecas = st.number_input("Total de Peças Solicitadas", min_value=0, value=1)
    tempo_resolucao = st.number_input("Tempo de Resolução (Horas)", min_value=0.0, value=2.0, step=0.5)
with col2:
    estado_nome = st.selectbox("Estado do Atendimento", list(ESTADOS_DICT.keys()))
    linha_nome = st.selectbox("Linha de Produto", list(LINHAS_DICT.keys()))

# --- 4. PROCESSAMENTO E PREVISÃO ---
if st.button("Analisar Risco", type="primary", use_container_width=True):
    
    # Traduzindo o texto escolhido pelo usuário para o número (código)
    estado_cod = ESTADOS_DICT[estado_nome]
    linha_cod = LINHAS_DICT[linha_nome]
    
    # Criar um DataFrame com o formato exato que a IA espera
    novos_dados = pd.DataFrame({
        'total_pecas': [total_pecas],
        'tempo_resolucao_horas': [tempo_resolucao],
        'estado_cod': [estado_cod],
        'linha_cod': [linha_cod]
    })

    try:
        # Aplicar a mesma escala (Scaler) que foi usada no Colab
        dados_padronizados = scaler.transform(novos_dados)

        # Calcular o score da OS nova
        score_os = modelo.decision_function(dados_padronizados) * -1
        nota_final = score_os[0]

        # Veredito final comparando com o seu limiar
        st.markdown("---")
        if nota_final > limiar:
            st.error(f"**ALERTA DE ANOMALIA!** (Score: `{nota_final:.4f}` / Limite: `{limiar:.4f}`)")
            st.write("Esta OS tem um comportamento suspeito e deve ser verificada.")
        else:
            st.success(f"**ATENDIMENTO NORMAL** (Score: `{nota_final:.4f}` / Limite: `{limiar:.4f}`)")
            st.write("As métricas estão dentro do padrão aceitável.")
            
    except Exception as e:
        st.warning("Ocorreu um erro no processamento matemático. Verifique se os arquivos .pkl na pasta 'models' estão atualizados.")
        st.write(f"Detalhes: {e}")
