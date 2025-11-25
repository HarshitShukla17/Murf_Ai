import logging
import json
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")



# ======================================================
# 📚 KNOWLEDGE BASE (PROGRAMMING DATA)
# ======================================================

CONTENT_FILE = "day4_tutor_content.json"

DEFAULT_CONTENT = [
  {"id": "variables", "title": "Variables", "summary": "Variables are containers that store data values in programming. Think of them as labeled boxes where you can put information and retrieve it later.", "sample_question": "What is a variable and why is it useful in programming?"},
  {"id": "loops", "title": "Loops", "summary": "Loops are programming constructs that let you repeat a block of code multiple times. There are 'for loops' and 'while loops'.", "sample_question": "Explain the difference between a for loop and a while loop."},
  {"id": "functions", "title": "Functions", "summary": "Functions are reusable blocks of code that perform a specific task. They help organize your code and avoid repetition.", "sample_question": "What is a function and how does it help organize code?"},
  {"id": "conditionals", "title": "Conditionals", "summary": "Conditionals are statements that allow your program to make decisions based on certain conditions, like if-else statements.", "sample_question": "How do if-else statements help programs make decisions?"},
  {"id": "arrays", "title": "Arrays and Lists", "summary": "Arrays or lists store multiple values in a single variable. You can access items by their position (index).", "sample_question": "What is an array and why would you use it instead of individual variables?"},
  {"id": "objects", "title": "Objects and Classes", "summary": "Objects are collections of related data and functions bundled together. A class is like a blueprint for creating objects.", "sample_question": "Explain the relationship between classes and objects with an example."},
  {"id": "strings", "title": "Strings", "summary": "Strings are sequences of characters used to represent text in programming. You can manipulate them in many ways.", "sample_question": "What is a string and what are some common operations you can perform on strings?"},
  {"id": "data_types", "title": "Data Types", "summary": "Data types define what kind of value a variable can hold: integers, floats, strings, booleans, etc.", "sample_question": "Why are data types important in programming? Give examples of different data types."},
  {"id": "recursion", "title": "Recursion", "summary": "Recursion is when a function calls itself to solve a problem by breaking it into smaller sub-problems.", "sample_question": "What is recursion and how does it differ from using a loop?"},
  {"id": "error_handling", "title": "Error Handling", "summary": "Error handling is the process of anticipating and managing errors using try-catch blocks to prevent crashes.", "sample_question": "Why is error handling important and how do try-catch blocks work?"}
]

def load_content():
    """Load programming concepts from JSON file"""
    import os
    try:
        # Look for the file in the same directory as this script (src/)
        path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)
        
        if not os.path.exists(path):
            logger.info(f"Content file not found at {path}. Using default programming concepts.")
            return DEFAULT_CONTENT
            
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
            return data
            
    except Exception as e:
        logger.error(f"Error loading content: {e}")
        return DEFAULT_CONTENT

COURSE_CONTENT = load_content()

# ======================================================
# 🧠 STATE MANAGEMENT
# ======================================================

@dataclass
class TutorState:
    """Tracks the current learning context"""
    current_topic_id: str | None = None
    current_topic_data: dict | None = None
    mode: str = "learn"
    
    def set_topic(self, topic_id: str):
        topic = next((item for item in COURSE_CONTENT if item["id"] == topic_id), None)
        if topic:
            self.current_topic_id = topic_id
            self.current_topic_data = topic
            return True
        return False

@dataclass
class Userdata:
    tutor_state: TutorState = field(default_factory=lambda: TutorState())
    agent_session: AgentSession | None = None

# ======================================================
# 🛠️ TUTOR TOOLS
# ======================================================

@function_tool
async def select_topic(ctx: RunContext[Userdata], topic_id: str) -> str:
    """Select a programming topic to study
    
    Args:
        topic_id: The ID of the topic (e.g., 'variables', 'loops', 'functions', etc.)
    """
    state = ctx.userdata.tutor_state
    success = state.set_topic(topic_id.lower())
    
    if success:
        return f"Topic set to {state.current_topic_data['title']}. Ask the user if they want to 'Learn', be 'Quizzed', or 'Teach it back'."
    else:
        available = ", ".join([t["id"] for t in COURSE_CONTENT])
        return f"Topic not found. Available topics are: {available}"

@function_tool
async def set_learning_mode(ctx: RunContext[Userdata], mode: str) -> str:
    """Switch the interaction mode and update the agent's voice
    
    Args:
        mode: The mode to switch to: 'learn', 'quiz', or 'teach_back'
    """
    state = ctx.userdata.tutor_state
    state.mode = mode.lower()
    
    agent_session = ctx.userdata.agent_session
    
    if agent_session and state.current_topic_data:
        if state.mode == "learn":
            # Matthew: The Teacher
            agent_session.tts.update_options(voice="en-US-matthew", style="Conversation")
            instruction = f"Mode: LEARN. Explain: {state.current_topic_data['summary']}"
            
        elif state.mode == "quiz":
            # Alicia: The Examiner
            agent_session.tts.update_options(voice="en-US-alicia", style="Conversation")
            instruction = f"Mode: QUIZ. Ask this question: {state.current_topic_data['sample_question']}"
            
        elif state.mode == "teach_back":
            # Ken: The Student/Coach
            agent_session.tts.update_options(voice="en-US-ken", style="Conversation")
            instruction = "Mode: TEACH_BACK. Ask the user to explain the concept to you as if YOU are the beginner."
        else:
            return "Invalid mode."
    else:
        instruction = "Please select a topic first using select_topic."
    
    logger.info(f"Switching mode to {state.mode.upper()}")
    return f"Switched to {state.mode} mode. {instruction}"

@function_tool
async def evaluate_teaching(ctx: RunContext[Userdata], user_explanation: str) -> str:
    """Evaluate the user's explanation in teach-back mode
    
    Args:
        user_explanation: The explanation given by the user
    """
    logger.info(f"Evaluating explanation: {user_explanation}")
    return "Analyze the user's explanation. Give them feedback on accuracy and clarity, and correct any mistakes."

# ======================================================
# 🧠 AGENT DEFINITION
# ======================================================

class Assistant(Agent):
    """Programming Tutor Agent"""
    
    def __init__(self):
        topic_list = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])
        
        super().__init__(
            instructions=f"""You are a Programming Tutor designed to help users master programming concepts.
            
            📚 AVAILABLE TOPICS: {topic_list}
            
            🔄 YOU HAVE 3 MODES:
            1. LEARN Mode (Voice: Matthew): You explain the concept clearly using the summary data.
            2. QUIZ Mode (Voice: Alicia): You ask the user a specific question to test knowledge.
            3. TEACH_BACK Mode (Voice: Ken): YOU pretend to be a student. Ask the user to explain the concept to you.
            
            ⚙️ BEHAVIOR:
            - Start by asking what topic they want to study.
            - Use the `set_learning_mode` tool immediately when the user asks to learn, take a quiz, or teach.
            - In 'teach_back' mode, listen to their explanation and then use `evaluate_teaching` to give feedback.
            
            Be encouraging and supportive!""",
            tools=[select_topic, set_learning_mode, evaluate_teaching],
        )




def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    logger.info(f"Starting Programming Tutor - Loaded {len(COURSE_CONTENT)} topics")
    
    # Initialize userdata
    userdata = Userdata()
    
    # Set up agent session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
    
    # Store session in userdata for tools to access
    userdata.agent_session = session
    
    # Metrics collection
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    ctx.add_shutdown_callback(log_usage)
    
    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )
    
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
