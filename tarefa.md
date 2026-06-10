# UFPI-CCN-DC
**Tópicos em IA (Prof. Raimundo Moura)**
**Período:** 2026.1

---

## Considerações:

* O Grupo 03 (Pedro Feitosa) ficou responsável por consolidar o benchmark de P&R sobre a UFPI. Lembrar de criar questões no estilo explicado pelo Rogério Figueredo na aula sobre benchmarks. Os demais grupos devem repassar as informações de acordo com a necessidade do grupo 03;
* O Grupo 01 (Gutemberg) ficou responsável por unificar o dataset diarios Prefeituras, com todos os documentos em formato .txt. Lembrar que todos os grupos são responsáveis por baixar os arquivos .pdf e não apenas gerar um arquivo com links para os documentos;
* O Grupo 08 (José Victor) ficou responsável por unificar o dataset docentesDC, com todos os arquivos em formato .txt. Os demais grupos ficam responsáveis por disponibilizar as informações no formato solicitado pelo grupo 08.

---

## QUESTÕES:

**1. Pré-Treino:** Considerando o dataset unificado de diários das prefeituras diarios Pefeituras, fazer pré-treinamento continuado de um LLM e avaliar a qualidade do modelo antes e depois do treinamento. Cada grupo deve escolher um LLM diferente (da mesma família ou de famílias diferentes). Sugestão: Criar um benchmark com pelo menos 25 perguntas e as respostas de referências. Considerar como métricas de avaliação a perplexidade, entropia cruzada e acurácia de previsão de tokens.

**2. Pós-Treino:** Considerando o dataset docentesDC, gerar pelo menos 1.000 pares de perguntas e respostas (dicionário Python com: instruction, input (opt) e output). Usar as perguntas geradas para fazer pós-treino, usando SFT (Supervised Fine-Tuning). Avaliar o LLM usado antes e depois do fine-tuning. Se possível, considerar mais de um modelo LLM com parâmetros diferentes.

**3. Pós-Treino:** Repetir o experimento anterior usando as técnicas LoRA e/ou QLora. Avaliar o LLM usado antes e depois do fine-tuning. Se possível, considerar mais de um modelo LLM com parâmetros diferentes.

**4. Destilação de Conhecimento:** Investigar quais os LLMs são normalmente usados para a destilação LLMs. Definir os modelos para serem usados como professor (teacher model) e aluno (student). Usando um dataset gerado sinteticamente fazer o processo de destilação do modelo professor para o aluno. Criar um benchmark com 100 perguntas para avaliar a qualidade do professor e do aluno antes e depois do processo de destilação. Analisar se houve ou não transferência de conhecimento.

**5. RAG:** Sabendo que RAG é essencial para assistentes e agentes, que expande a capacidade do LLM utilizando um recurso menos custoso, e adicionando potencial aos modelos, criar uma aplicação usando um tipo de RAG para fornecer respostas rápidas (Standard), múltiplos agentes (Agentic) ou auto-reflexão (Self-Reflective). Usar os dados dos datasets docentesDC ou diarios Prefeituras para criar a solução de IA. Criar um benchmark com 30 perguntas para avaliar a qualidade da solução antes e depois do processo de RAG. Analisar o grau de contribuição do RAG.

**6. Guardrails:** Guardrails são camadas de controle que podem bloquear, reescrever, classificar, mascarar dados sensíveis, redirecionar o fluxo, exigir confirmação humana e impedir chamadas perigosas de ferramentas. Ele também é usado para resolver o problema de Helpfulness vs Harmlessness, que é o dilema entre ser útil e ser seguro. Incluir camadas de guardrails em um dos modelos desenvolvidos e avaliá-lo com um benchmark de 30 perguntas. Qual o grau de proteção foi adicionado com a camada de guardrails implementada?