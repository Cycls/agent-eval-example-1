"""
Cycls Agent Wrapper - Thin wrapper around agent_core.py
"""
import cycls
import os
import dotenv
from ui import header, intro
from agent_core import market_research_core

dotenv.load_dotenv()

agent = cycls.Agent(
    pip=["requests", "openai", "python-dotenv"], 
    copy=[".env", "ui.py", "agent_core.py"]
)

@agent("market-researcher", header=header, intro=intro)
async def market_research_agent(context):
    """Cycls wrapper - delegates to core logic"""
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in context.messages]
    
    async for item in market_research_core(messages):
        if item["type"] == "loader":
            yield item["content"]
        elif item["type"] in ["text", "research", "error"]:
            yield item["content"]
        # tool_call type is internal, not yielded to user

agent.deploy(prod=False)
