# Q5 - exemplos qualitativos (sem RAG vs com RAG)

Motor: models/Qwen3-8B. Modo RAG: standard (recupera + gera).

## Exemplo 1
**Pergunta:** Qual é o artigo que trata da possibilidade de adjudicação quando a negociação de melhor condição frustra-se?

**Resposta de referencia:** 5.12.2

**Sem RAG (so o modelo):** O artigo que trata da possibilidade de adjudicação quando a negociação de melhor condição frustra-se é o **artigo 42 do Código de Processo Civil (CPC/2015)**.

Esse artigo prevê que, em leilão ou pregão, se a negociação de melhor condição frustrar-se, o bem será adjudicado ao maior lance ou ao melhor preço ofertado, conforme o caso.

**Com RAG:** O artigo que trata da possibilidade de adjudicação quando a negociação de melhor condição frustra-se é o **item 5.12.2** (ou **item 3.12.2** ou **item 2.12.2**, conforme o contexto fornecido).

## Exemplo 2
**Pergunta:** Qual o número do ato que corresponde à transferência de 10.000,00 para a TRANSF.ALDIR BLANC LEI N?

**Resposta de referencia:** 1.719.0

**Sem RAG (so o modelo):** Para identificar o número do ato que corresponde à transferência de R$ 10.000,00 para a "TRANSF.ALDIR BLANC LEI N?", é necessário:

1. **Aferir o contexto** em que essa transferência foi realizada (ex.: orçamento, licitação, contrato, etc.).
2. **Consultar os registros oficiais** de transferências financeiras, como o sistema de controle de recursos públicos da prefeitura ou órgão responsável.
3. **Verificar a Lei nº** mencionada, pois ela pode estar relacionada ao fundo, programa ou ação que justifica a transferência.

Se você tiver mais informações, como o município, o órgão, o ano ou o tipo de transferência, posso ajudar a localizar o ato correspondente.

**Com RAG:** O número do ato que corresponde à transferência de 10.000,00 para a TRANSF.ALDIR BLANC LEI N é **Lei nº 14.399/2022**.

## Exemplo 3
**Pergunta:** Qual é a data de assinatura do documento?

**Resposta de referencia:** 28 de fevereiro de 2025

**Sem RAG (so o modelo):** Para responder à sua pergunta, preciso de mais informações, como o nome ou o conteúdo do documento em questão. Por favor, forneça detalhes adicionais para que eu possa ajudar você com precisão.

**Com RAG:** A data de assinatura do documento é 28 de agosto de 2025.

## Exemplo 4
**Pergunta:** Qual é o nome da divisão responsável pela gestão de imóveis na Diretoria de Investimentos?

**Resposta de referencia:** Assessoria de Gestão de Imóveis

**Sem RAG (so o modelo):** A divisão responsável pela gestão de imóveis na Diretoria de Investimentos é geralmente chamada de **Divisão de Gestão de Ativos** ou **Divisão de Operações Imobiliárias**, dependendo da estrutura organizacional específica da empresa.

**Com RAG:** A divisão responsável pela gestão de imóveis na Diretoria de Investimentos é a Assessoria de Gestão de Imóveis.

## Exemplo 5
**Pergunta:** Qual é o número da portaria que institui a Comissão Especial para discutir a resolução normativa que regula a oferta da Educação Infantil?

**Resposta de referencia:** Portaria Nº 004/2023/CME/SEMEC

**Sem RAG (so o modelo):** A portaria que institui a Comissão Especial para discutir a Resolução Normativa que regula a oferta da Educação Infantil é a **Portaria nº 1.116, de 20 de abril de 2016**, publicada no Diário Oficial da União.

**Com RAG:** A Portaria que institui a Comissão Especial para discutir a resolução normativa que regula a oferta da Educação Infantil é a **Portaria Nº 004/2023/CME/SEMEC**.
