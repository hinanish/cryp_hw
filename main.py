from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel,Runner, RunConfig , function_tool
from dotenv import load_dotenv
import os
import chainlit as cl
import requests


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

@function_tool
def crypto_price(symbol:str) -> str:
    """
    Get real-time crypto price in USD using Binance public API.
    
    Args:
        symbol (str): The cryptocurrency symbol, e.g., BTC. ETH.

    Returns:
        str: Current price in USD.
    """
    trading_pair = symbol.upper() + "USDT"
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={trading_pair}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        price = data.get("price")
        return f"The current price of {trading_pair} is ${price}"
    
    except requests.exceptions.HTTPError:
        return f"Symbol {trading_pair} not found on Binance.Try BTC or ETH"
    except Exception as e:
        return f"Error fetching price: {str(e)}"

client = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
  model = 'gemini-2.0-flash',
  openai_client = client
)
agent = Agent(
    name = 'Crypto Agent',
    instructions = """
    You are a helpful crypto agent. When user asks for the price of any cryptocurrency (like BTC, ETH, SHIB) always call the ' crypto_price' tool
    Donot try to answer from your own knowledge -- always use tool.
    
""",
    tools = [crypto_price]
)

config = RunConfig(
    model = model,
    model_provider= client,
    tracing_disabled= True
)

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content= f"Assalamoalekum, I am {agent.name}").send()

@cl.on_message
async def handle_message(message):
    history = cl.user_session.get("history")
    history.append({'role':'user','content': message.content})
    result = Runner.run_sync(
        agent,
        input = history,
        run_config = config,
  
    
   
)
    
    history.append({'role':' assistant', 'content' : result.final_output})
    cl.user_session.set('history', history)
    await cl.Message(content=result.final_output).send()