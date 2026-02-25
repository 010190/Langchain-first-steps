from dataclasses import dataclass
import os

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()


@tool('locate_user', description="Look up a user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Vienna'
        case 'XTZ456':
            return 'London'
        case 'HJKL111':
            return 'Paris'
        case _:
            return 'Unknown'


api_key = os.getenv("OPENAI_API_KEY")

model = init_chat_model(
    model='gpt-4.1-mini',
    temperature=0.3
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt='You are helpful weather assistant',
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke({
        'messages': [
            {'role': 'user', 'content': 'What is the weather like?'}
        ]},
        config=config,
        context=Context(user_id='ABC123')

)
print(response['structured_response'])
print(response['structured_response'].summary)
print(response['structured_response'].temperature_celsius)




# conversation = [
#     SystemMessage("You are helpful assistant for question regarding programming"),
#     HumanMessage('What is Python'),
#     AIMessage('Python is an interpreted programming language.'),
#     HumanMessage('When was it released?')
# ]
#
# response = model.invoke('Hello, what is Python')
#
# print(response)
# print(response.content)
#
# for chunk in model.stream('Hello, what is Python'):
#     print(chunk.text, end='', flush=True)
'''''Wyswietlanie na biezaco, strumieniowanie'''
# print(response)
# print(response.content)
