# -------------------------
# Concluding note always visible
# -------------------------
st.markdown("---")
st.markdown(
    """
    <div style="font-size:16px; text-align:justify; line-height:1.5;">
    <b>Fiscalização Marítima</b><br><br>
    Este trabalho é inspirado no artigo publicado na <i>Nature Scientific Data (2024)</i>, que disponibiliza dados anonimizados de fiscalizações realizadas em embarcações, incluindo variáveis como localização, tipo de navio, dimensões (ex.: LOA), artes de pesca utilizadas e resultados da inspeção (com destaque para os casos <b>PRESUM</b>, presumíveis infrações).<br><br>
    O objetivo deste <b>dashboard interativo</b> é identificar as áreas com maior densidade de embarcações suspeitas, otimizando as rotas de fiscalização. 
    Com este painel, a <b>Marinha</b> pode priorizar áreas de intervenção, planear patrulhas mais eficientes e reduzir custos operacionais.
    </div>
    """,
    unsafe_allow_html=True
)
