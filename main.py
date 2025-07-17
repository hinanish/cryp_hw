from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunConfig, function_tool
from dotenv import load_dotenv
import chainlit as cl
import requests
import os

# Load API keys
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# === Tool: Crypto Price Checker ===
@function_tool
def crypto_price(symbol: str) -> str:
    """
    Get real-time crypto price in USD using Binance public API.
    """
    trading_pair = symbol.upper() + "USDT"
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={trading_pair}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        price = data.get("price")
        return f"The current price of {symbol.upper()} is ${price}"
    except requests.exceptions.HTTPError:
        return f"Symbol '{symbol}' not found on Binance. Try 'BTC' or 'ETH'."
    except Exception as e:
        return f"Error fetching price: {str(e)}"

# === Gemini Model Config ===
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

agent = Agent(
    name="Crypto Agent",
    instructions="""
You are a helpful crypto agent. When the user provides the name or symbol of a cryptocurrency (like BTC, ETH, SHIB), always call the 'crypto_price' tool to fetch its current price.

If you're unsure, ask the user for the symbol. Do not guess or generate your own responses.
""",
    tools=[crypto_price]
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# === Chainlit Handlers ===
@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content=f"👋 Assalamoalekum, I am {agent.name}. Ask me about any cryptocurrency price.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    
    # Add user message to history
    history.append({'role': 'user', 'content': message.content})
    
    try:
        # Pass full chat history
        result = await Runner.run(
            agent,
            input=history,
            run_config=config
        )

        # Save agent reply to history
        history.append({'role': 'assistant', 'content': result.final_output})
        cl.user_session.set("history", history)

        await cl.Message(content=result.final_output).send()
    except Exception as e:
        await cl.Message(content=f"⚠️ Error: {str(e)}").send()
