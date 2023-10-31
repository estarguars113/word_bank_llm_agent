import sys

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain_experimental.agents.agent_toolkits import create_csv_agent


if __name__ == "__main__":
    input_string = sys.argv[1]
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        "./data/world_bank/formatted_Series_metadata.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        pandas_kwargs={'header': 0, 'encoding': 'unicode_escape'}
    )

    response = agent.run(input_string)
    print(f"{input_string}: {response}")