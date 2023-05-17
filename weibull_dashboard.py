import streamlit as st
import weibull
import numpy as np
import pandas as pd
# Ustawienia systemowe
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide", page_title="Reliability assesment acc. Weibull")
hide_default_format= """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden}
        </style>
        """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Nagłówek
st.header('Interaktywny dashboard do szacowania niezawodności pracy urządzenia/komponentu')
st.sidebar.write("Poniżej należy wprowadzić dane do wyliczenia niezawodności pracy maszyny")

ilosc_awrii = st.sidebar.number_input("Ilość danych do wprowadzenia odnośnie znanych czasów",min_value=0)

# Wprowadzane dane w zależności od ilości dostępnych danych
dane = []
for i in range(ilosc_awrii):
    ttf = st.sidebar.number_input(f"Wprowadż czas między awariami dla {i+1} zdarzenia")
    dane.append(ttf)
# Podanie planowanego czasu pracy dla którego ma być wylizony poziom niezawodności
tc = st.sidebar.number_input("Planowany czas pracy")

# Przycisk rozpoczynający wyliczenia
obliczenia = st.sidebar.button('Obliczenia')
#st.write(pd.Series(dane))

if obliczenia:
    analysis = weibull.Analysis(dane, unit='godziny')
    analysis.fit()
    st.caption("Wyliczone współczynniki właściwości rozkładu Weibulla")
    col_a, col_b, col_c, col_d, col_e, col_f= st.columns(6)
    with col_a:
        st.latex(r'''\beta''')
    with col_b:
        st.metric(label= '', value = f' {analysis.beta: .02f}')
    with col_c:
        st.latex(r'''\eta''')
    with col_d:
        st.metric(label= '', value= f' {analysis.eta: .02f}')
    with col_e:
        st.latex('MTBF')
    with col_f:
        st.metric(label ='', value=f'{analysis.mean: .02f}')

# wykresy z dokonanych obliczeń rozkładu Weibulla
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Wykres prawdopodobieństwa")
        fig = analysis.probplot()
        st.pyplot()
    with col2:
        st.subheader("Funkcja przeżycia")
        fig2 = analysis.sf()
        st.pyplot(fig2)
    with col3:
        st.subheader("Funkcja zagrożenia")
        fig3 = analysis.hazard()
        st.pyplot(fig3)

# wyliczenie poziomu niezawodnej pracy przy zadanym czasie
    eta = analysis.eta
    beta = analysis.beta
    re = np.exp(-(tc/eta)**(beta))
    st.write(f'Niezawodność przy szacowanym czasie {tc} godzin wynosi: ')
    st.metric(label='', value=f'  {re*100: .02f} %')






