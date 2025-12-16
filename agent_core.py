"""
Core agent logic - NO cycls dependency.
This module contains the pure business logic that can be tested independently.
"""
import os
import json
import requests
import asyncio
import dotenv
from typing import List, Dict, Any, AsyncGenerator
from openai import OpenAI

dotenv.load_dotenv()

# ==================================================================================
# 🛠️ UTILITIES
# ==================================================================================

def get_env(key: str) -> str:
    """Get env var with fallback to reading .env file"""
    val = os.getenv(key)
    if not val:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.strip() and f'{key}=' in line:
                        val = line.split('=', 1)[1].strip().strip('"').strip("'")
                        os.environ[key] = val
                        break
        except: pass
    return val

# ==================================================================================
# 🔍 EXA RESEARCH
# ==================================================================================

def exa_request(api_key: str, endpoint: str, payload: Dict) -> List[Dict]:
    """Make request to Exa API"""
    try: 
        return requests.post(
            f"https://api.exa.ai/{endpoint}", 
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, 
            json=payload, 
            timeout=30
        ).json().get("results", [])
    except: 
        return []

def perform_research(exa_key: str, openai_client, company_name: str) -> str:
    """Perform market research using Exa and OpenAI"""
    searches = [
        (f"{company_name} official website about", 3), 
        (f"competitors alternatives to {company_name}", 8), 
        (f"{company_name} industry analysis comparison", 5)
    ]
    all_results = [r for q, n in searches for r in exa_request(exa_key, "search", {"query": q, "numResults": n, "useAutoprompt": True})]
    result_ids = list({r["id"]: r for r in all_results if "id" in r}.keys())[:15]
    contents = exa_request(exa_key, "contents", {"ids": result_ids, "text": True, "maxCharacters": 15000})
    sources = [{"url": c.get("url", ""), "title": c.get("title", ""), "text": c.get("text", "")[:2000]} for c in contents]
    
    prompt = f"Based on sources about {company_name}:\n{json.dumps(sources, indent=2)}\n\nProvide: 1. Brief overview of {company_name} 2. 4-6 main competitors 3. Comparison table (markdown) with: Company, Category, Target Customer, Key Features, Pricing, Website 4. Key insights and differentiators"
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}], 
        temperature=0.2
    ).choices[0].message.content
    
    source_urls = list(set([s["url"] for s in sources if s["url"]]))[:10]
    return response + "\n\n---\n**Sources:**\n" + "\n".join([f"- {url}" for url in source_urls])

# ==================================================================================
# 🤖 CORE AGENT LOGIC
# ==================================================================================

SYSTEM_PROMPT = """You are a helpful market research assistant. Your job is to help users with market research.

IMPORTANT RULES:
1. If the user mentions ANY company name in their message, IMMEDIATELY use the research_company tool. Do NOT ask for confirmation.
2. If the user's message contains a company name (even with typos or alongside other text), extract it and call the tool.
3. Only ask "which company?" if truly no company is mentioned at all.
4. For greetings like "hi" or "hello" with no company, respond warmly and ask what company they'd like to research.
5. For off-topic questions (weather, etc.), politely redirect to market research.

Examples of when to USE the tool immediately:
- "Research Tesla" → Call tool with "Tesla"
- "Find competitors for Spotify" → Call tool with "Spotify"  
- "What's Apple's pricing?" → Call tool with "Apple"
- "SWOT analysis for Nike" → Call tool with "Nike"
- "Tell me about Microsoft and their competitors" → Call tool with "Microsoft"

Examples of when to ASK for company:
- "hi" → Greet and ask what company
- "I need market research" → Ask which company
- "help me" → Ask what company they want to research"""

TOOLS = [{
    "type": "function", 
    "function": {
        "name": "research_company", 
        "description": "Performs detailed market research on a company and its competitors", 
        "parameters": {
            "type": "object", 
            "properties": {
                "company_name": {
                    "type": "string", 
                    "description": "The company to research"
                }
            }, 
            "required": ["company_name"]
        }
    }
}]

async def market_research_core(messages: List[Dict[str, str]]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Core agent logic - processes messages and yields responses.
    
    Yields dicts with:
    - {"type": "text", "content": "..."} for text responses
    - {"type": "loader", "content": "...", "company": "..."} for loading indicators
    - {"type": "research", "content": "...", "company": "..."} for research results
    - {"type": "error", "content": "..."} for errors
    - {"type": "tool_call", "name": "...", "args": {...}} for tool call info
    """
    dotenv.load_dotenv()
    api_key = get_env("OPENAI_API_KEY")
    
    if not api_key:
        yield {"type": "error", "content": "Error: OPENAI_API_KEY not found. Please set it in your .env file."}
        return
    
    openai_client = OpenAI(api_key=api_key)
    
    # Build conversation
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    llm_messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in messages])
    
    # Call LLM with tools
    response_msg = openai_client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=llm_messages, 
        tools=TOOLS, 
        tool_choice="auto", 
        temperature=0.7
    ).choices[0].message
    
    # Check if tool was called
    if response_msg.tool_calls:
        tool_call = response_msg.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        company_name = args.get("company_name", "Unknown")
        
        # Yield tool call info
        yield {"type": "tool_call", "name": tool_call.function.name, "args": args}
        
        # Yield loader
        loader_html = f'<div style="padding:24px;text-align:center;max-width:400px;margin:0 auto;"><div style="display:inline-flex;align-items:center;gap:12px;padding:16px;background:linear-gradient(135deg,#667eea15 0%,#764ba215 100%);border-radius:12px;border:1px solid #667eea40;"><div style="width:24px;height:24px;border:3px solid #e0e7ff;border-top-color:#667eea;border-radius:50%;animation:spin 0.8s linear infinite;"></div><div style="text-align:left;"><div style="font-weight:700;color:#3730a3;margin-bottom:4px;">🔍 Researching {company_name}</div><div style="font-size:13px;color:#64748b;">Gathering market data... This may take 30-60 seconds</div></div></div></div><style>@keyframes spin{{0%{{transform:rotate(0deg);}}100%{{transform:rotate(360deg);}}}}</style>'
        yield {"type": "loader", "content": loader_html, "company": company_name}
        
        await asyncio.sleep(0.1)
        
        # Get Exa key and perform research
        exa_key = get_env("EXA_API_KEY")
        if not exa_key:
            yield {"type": "error", "content": "Error: EXA_API_KEY not found. Please set it in your .env file."}
            return
        
        research_result = perform_research(exa_key, openai_client, company_name)
        yield {"type": "research", "content": f"🔍 **Market Research for {company_name}**\n\n{research_result}", "company": company_name}
    else:
        # No tool call - just return the text response
        yield {"type": "text", "content": response_msg.content or ""}


async def market_research_simple(messages: List[Dict[str, str]]) -> str:
    """
    Simplified interface - returns concatenated string response.
    Useful for testing.
    """
    chunks = []
    async for item in market_research_core(messages):
        if item["type"] in ["text", "research", "error"]:
            chunks.append(item["content"])
        elif item["type"] == "loader":
            # Skip loader HTML for simple interface
            pass
    return "\n".join(chunks)


def get_tool_info() -> Dict[str, Any]:
    """Return tool definition for inspection"""
    return {
        "system_prompt": SYSTEM_PROMPT,
        "tools": TOOLS
    }

