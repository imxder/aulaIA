from dotenv import load_dotenv
load_dotenv()

import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool
from litellm import completion

# Resposta simples usando completion
response = completion(
    model="groq/llama3-8b-8192",
    messages=[
        {"role": "user", "content": "hello"}
    ],
)
print(response)

# Definição do agente pesquisador de mercado focado em carros usados
pesquisador_mercado = Agent(
    role="Pesquisador de Mercado Automotivo",
    goal="Garanta que a compra do carro usado seja respaldada por pesquisas e dados sólidos. Realize uma pesquisa abrangente e realista para o modelo {modelo} de carro usado, com foco em SUVs, considerando preços, ano de fabricação e disponibilidade em Jaraguá do Sul, SC, Brasil.",
    backstory="Você é um especialista em pesquisa de mercado automotivo, com vasta experiência em avaliar a compra de veículos usados.",
    allow_delegation=False,
    verbose=True,
    llm="groq/llama3-8b-8192"
)

# Definição do agente comprador de carros usados
comprador_carros = Agent(
    role="Comprador de Carros Usados",
    goal="Criar um plano para a compra de um SUV usado no melhor preço possível, considerando as opções em Jaraguá do Sul, SC, Brasil.",
    backstory="Você tem experiência na compra de carros usados, sendo focado na pesquisa de melhores ofertas e veículos em boas condições.",
    allow_delegation=False,
    verbose=True,
    llm="groq/llama3-8b-8192"
)

# Instalação da ferramenta de busca SerperDevTool
ferramenta = SerperDevTool()

# Definir as tarefas para o pesquisador de mercado automotivo
tarefa_pesquisador_mercado = Task(
    description="Analise os modelos de SUV usados disponíveis para compra em Jaraguá do Sul, SC, Brasil. \
                Considere o ano de fabricação, os preços e o estado do veículo. \
                Avalie a existência de outras opções de SUV na cidade. \
                Forneça insights para a decisão de compra, incluindo sites de venda.",
    expected_output=(
        "Um relatório detalhado da pesquisa de mercado de SUVs usados mencionando os melhores modelos, preços, opções disponíveis em Jaraguá do Sul, SC, Brasil, e sites relevantes para compra."
    ),
    tools=[ferramenta],
    agent=pesquisador_mercado,
    search_query="SUVs usados à venda em Jaraguá do Sul, SC, Brasil"
)

# Definir as tarefas para o comprador de carros usados
tarefa_comprador_carros = Task(
    description="Baseado na pesquisa realizada, crie um plano detalhado para a compra de um SUV usado em Jaraguá do Sul, SC, Brasil. \
                Garanta que o plano inclua as melhores opções de preço, ano e estado do veículo, bem como os sites recomendados para a compra. \
                Certifique-se de que todos os fatores importantes na compra do SUV foram cobertos.",
    expected_output=(
        "A saída deve conter um plano detalhado de compra para um SUV usado com os melhores modelos e preços disponíveis em Jaraguá do Sul, SC, Brasil, e links para sites de venda."
    ),
    tools=[ferramenta],
    agent=comprador_carros,
    output_file='plano_de_compra_suv.md'
)

# Criar grupo de agentes
equipe = Crew(
    agents=[pesquisador_mercado, comprador_carros],
    tasks=[tarefa_pesquisador_mercado, tarefa_comprador_carros],
    verbose=True,
    max_rpm=25
)

# Iniciar execução com a ideia de compra de carro SUV usada
resultado = equipe.kickoff(
    inputs={"modelo": "SUV"}
)

print(resultado)