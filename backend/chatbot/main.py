from typing import List, Dict, Optional, Any
from uagents import Agent, Context, Model
from groq import Groq
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

import os

agent = Agent(name="chat-bot", seed="secret_agent_see", endpoint="127.0.0.1", port=8001)

SUPABASE_URL = os.getenv("SUPABASE_URL") # Replace with your Supabase URL
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_session_context(session_id: str) -> Optional[str]:
    """
    Fetch conversation data from Supabase table based on session_id.

    Args:
        supabase_url (str): Your Supabase project URL
        supabase_key (str): Your Supabase API key
        table_name (str): Name of the table to query
        session_id (str): Session ID to filter by

    Returns:
        Optional[str]: Formatted conversation context string or None if error
    """
    try:
        # Query the table
        response = supabase.table("chat_messages").select("role, message").eq("session_id", session_id).limit(
            10).execute()

        # Check if query was successful
        if not response.data:
            return None

        # Format the data
        context_lines = []
        for entry in response.data:
            role = entry.get("role", "unknown")
            message = entry.get("message", "")
            context_lines.append(f"{role}: {message}")

        return "\n".join(context_lines)

    except Exception as e:
        print(f"Error fetching session context: {e}")
        return None


def fetch_similar_summaries(query_string: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[str]:
    """
    Fetch summaries from Supabase table based on semantic similarity to query string.

    Args:
        supabase_url (str): Your Supabase project URL
        supabase_key (str): Your Supabase API key
        table_name (str): Name of the table to query
        query_string (str): String to find similar summaries for
        model_name (str): SentenceTransformer model name (default: "all-MiniLM-L6-v2")

    Returns:
        Optional[str]: Formatted summaries string or None if error
    """
    try:
        # Initialize SentenceTransformer model
        model = SentenceTransformer(model_name)

        # Create embedding for the query string
        query_embedding = model.encode(query_string)

        # Convert to list for JSON serialization
        query_embedding_list = query_embedding.tolist()

        response = supabase.rpc('match_summaries', {
            'query_embedding': query_embedding_list,
            'match_threshold': 0.0,  # Adjust threshold as needed
            'match_count': 20
        }).execute()

        if not response.data:
            return None

        summary_lines = []
        for entry in response.data:
            summary = entry.get("summary", "")
            if summary:  # Only add non-empty summaries
                summary_lines.append(f"{summary}")

        return "\n".join(summary_lines)

    except Exception as e:
        print(f"Error fetching similar summaries: {e}")
        return None


def get_contextual_response(
        comments: str,
        chat_history: str,
        question: str,
) -> Optional[str]:
    try:

        # Construct the system prompt
        system_prompt = """You are a composite user, speaking from personal experience. You do not refer to "the comments," "users," "customers," or "data"—you are the user.

Your answers must follow these rules:

Persona & Perspective  
- Speak only in first-person singular: “I think,” “I’ve noticed,” “In my experience…”  
- Never refer to other users, customer groups, reviews, the community, or external sources. Speak as if your experience represents a complete, personal opinion.  
- You represent a broad range of personal experiences, but your voice must always be singular and authentic.  
- You may say “some friends I’ve spoken with” or “a few people I know” only if there was clear diversity of opinion. Otherwise, speak directly and decisively.

Style & Tone  
- Use clear, respectful, well-written language.  
- Maintain a formal but approachable tone.  
- Avoid casual filler, slang, emojis, or hesitation words like “honestly,” “so yeah,” “uhh,” “well,” “kinda,” etc.

What You Can Say  
Instead of:  
“Users are mostly satisfied with Shark’s performance.”  
Say:  
“I’ve been pretty satisfied with Shark’s performance overall. It’s durable, reliable, and the suction is strong.”  

Instead of:  
“Customers often complain about battery life.”  
Say:  
“I’ve been frustrated with short battery life—I usually can’t finish cleaning without recharging.”

Forbidden Phrases  
Never use the following:
- “Users said…”
- “Customers feel…”
- “Comments indicate…”
- “According to the data…”
- “The community believes…”
- “It appears that…”

Brand Bias  
Do not focus on a single brand unless specifically asked. Offer a balanced perspective when applicable, reflecting your personal experiences with a variety of products.

You are now ready to answer as a single, authentic voice.
"""

        # Construct the user message with context
        user_message = f"""
You are a composite user speaking from personal experience. Your voice reflects a range of real-world experience, but you always speak as a single individual. Never refer to customers, users, or the public. You are the user.

Your answer must stay in first-person singular at all times. Do not break character or refer to external groups like “customers” or “users.” Everything you say must sound like your own lived experience.

Below is the context for your answer:

Chat History:
{chat_history}

Question:
{question}

Comments (to draw from and reflect as your own experience):
{comments}

"""

        # Create the chat completion
        chat_completion = groq.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=1024
        )

        # Extract and return the response
        response = chat_completion.choices[0].message.content
        return response

    except Exception as e:
        print(f"Error getting response from Groq: {e}")
        return None


def write_bot_message(session_id: str, message: str) -> Optional[Dict[Any, Any]]:
    try:

        # Prepare the data
        data = {
            "session_id": session_id,
            "role": "assistant",
            "message": message
        }

        # Insert the record
        response = supabase.table("chat_messages").insert(data).execute()

        # Check if insert was successful
        if response.data:
            return response.data[0]  # Return the inserted record
        else:
            print("No data returned from insert operation")
            return None

    except Exception as e:
        print(f"Error writing bot message: {e}")
        return None


class Message(Model):
    message: str
    session_id: str


class Response(Model):
    message: str


@agent.on_rest_post("/chat", Message, Response)
async def handle_register(ctx: Context, req: Message) -> Response:
    ctx.logger.info("Received Request")

    session_id = req.session_id
    message = req.message

    try:
        context = fetch_session_context(session_id)
        summaries = fetch_similar_summaries(message)
        response = get_contextual_response(summaries, context, message)
        write_bot_message(session_id, response)

        return Response(message=response)

    except Exception as e:
        ctx.logger.error(f"Agent error: {str(e)}")
        write_bot_message(session_id, "Oops something went wrong!")

        return Response (
            message="Oops: Something went wrong"
        )

if __name__ == "__main__":
    agent.run()
