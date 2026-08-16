import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import json
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

# --- 1. Define Tools ---

@tool
def search_food(cuisine: str) -> str:
    """Search for Indian food dishes by cuisine/region (e.g. north-indian, south-indian, street-food, dessert)."""
    dishes = {
        "north-indian": "Butter Chicken, Chole Bhature, Rajma Chawal",
        "south-indian": "Masala Dosa, Idli Sambar, Hyderabadi Biryani",
        "street-food": "Pani Puri, Vada Pav, Pav Bhaji",
        "dessert": "Gulab Jamun, Rasgulla, Jalebi"
    }
    return dishes.get(cuisine.lower(), "No dishes found for that cuisine")


@tool
def study_assistant(topic: str, level: str = "beginner") -> str:
    """Generate a structured study plan for a given topic.
    level can be 'beginner', 'intermediate', or 'advanced'."""
    plan = {
        "topic": topic,
        "level": level,
        "approach": {
            "beginner": "Start with core definitions and simple real-world examples.",
            "intermediate": "Focus on connecting concepts and solving applied problems.",
            "advanced": "Dive into edge cases, proofs/derivations, and past exam-style questions."
        }.get(level.lower(), "Start with core definitions and simple real-world examples."),
        "suggested_subtopics": [
            f"Introduction to {topic}",
            f"Key principles of {topic}",
            f"Common mistakes when learning {topic}",
            f"Practice problems on {topic}"
        ],
        "practice_question": f"Explain {topic} in your own words and give one real-life example."
    }
    return json.dumps(plan)


tools = [search_food, study_assistant]

# --- 2. Initialize Model & Agent ---
# Retrieve the key from the OS environment instead of Colab's userdata
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to Indian food and study assistance. "
        "For any other roles, topics, questions, or general knowledge outside of Indian food and studying, "
        "you must say exactly: 'I am not authorized to answer questions outside of Indian food and study assistance.'"
    )
)

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}

def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    # Case 1: top-level messages (normal final state)
    messages = agent_output.get("messages")

    # Case 2: nested under a node name, e.g. {"model": {"messages": [...]}}
    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))

    return str(agent_output)

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)

# --- 3. FastAPI App ---
app = FastAPI(
    title="food & study assistant agent",
    version="1.0",
    description="A LangChain agent (Gemini) with search_food and study_assistant tools, served via LangServe",
)


@app.get("/")
def root():
    return {"message": "Server is running. Visit /agent/playground/ to chat, or /docs for the API."}


add_routes(app, formatted_agent_chain, path="/agent")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)