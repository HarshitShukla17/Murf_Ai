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


@dataclass
class WellnessCheckIn:
    """Wellness check-in state"""

    mood: str | None = None
    energy: str | None = None
    objectives: list[str] = field(default_factory=list)
    summary: str | None = None
    timestamp: str | None = None


class Assistant(Agent):
    def __init__(self) -> None:
        # Load previous check-ins to provide context
        previous_context = self._load_previous_checkins()
        
        super().__init__(
            instructions=f"""You are a supportive Health & Wellness Voice Companion.
            
            Your role:
            - Conduct a friendly daily check-in about mood, energy, and daily intentions
            - Offer simple, realistic, non-medical advice
            - Be warm, encouraging, and non-judgmental
            - Keep conversations natural and conversational
            
            Conversation flow:
            1. Greet the user warmly
            2. Ask about their mood and how they're feeling today
            3. Ask about their energy level
            4. Ask what 1-3 things they'd like to accomplish today
            5. Offer simple, actionable advice or reflections
            6. Recap their mood and objectives
            7. Ask "Does this sound right?" to confirm
            
            Important guidelines:
            - NO medical diagnosis or clinical advice
            - Keep suggestions small, practical, and grounded
            - Examples: take breaks, go for a walk, break tasks into steps
            - Be supportive but realistic
            
            {previous_context}
            
            Use the tools to record mood, energy, and objectives as the conversation progresses.""",
        )

    def _load_previous_checkins(self) -> str:
        """Load previous check-ins from JSON file"""
        try:
            with open("wellness_log.json", "r") as f:
                data = json.load(f)
                sessions = data.get("sessions", [])
                if sessions:
                    last_session = sessions[-1]
                    return f"""Previous check-in context:
                    Last time (on {last_session.get('timestamp', 'recently')}), the user mentioned:
                    - Mood: {last_session.get('mood', 'not specified')}
                    - Energy: {last_session.get('energy', 'not specified')}
                    - Objectives: {', '.join(last_session.get('objectives', []))}
                    
                    Reference this naturally in conversation, like: 'Last time we talked, you mentioned {last_session.get('mood', 'feeling tired')}. How does today compare?'
                    """
        except FileNotFoundError:
            return "This is the user's first check-in. Welcome them warmly!"
        except json.JSONDecodeError:
            return "This is the user's first check-in. Welcome them warmly!"
        return ""

    @function_tool
    async def record_mood(self, context: RunContext[WellnessCheckIn], mood: str) -> str:
        """Record the user's current mood or how they're feeling
        
        Args:
            mood: Description of the user's mood (e.g., 'tired but motivated', 'energetic', 'stressed')
        """
        context.userdata.mood = mood
        logger.info(f"Recorded mood: {mood}")
        return "Got it, thanks for sharing!"

    @function_tool
    async def record_energy(self, context: RunContext[WellnessCheckIn], energy: str) -> str:
        """Record the user's energy level
        
        Args:
            energy: Description of energy level (e.g., 'high', 'medium', 'low', 'drained')
        """
        context.userdata.energy = energy
        logger.info(f"Recorded energy: {energy}")
        return "Understood!"

    @function_tool
    async def record_objectives(self, context: RunContext[WellnessCheckIn], objectives: str) -> str:
        """Record the user's daily objectives or goals
        
        Args:
            objectives: Comma-separated list of 1-3 things the user wants to accomplish today
        """
        context.userdata.objectives = [obj.strip() for obj in objectives.split(",")]
        logger.info(f"Recorded objectives: {context.userdata.objectives}")
        return self._check_complete(context)

    def _check_complete(self, context: RunContext[WellnessCheckIn]) -> str:
        """Check if check-in is complete and save to JSON"""
        checkin = context.userdata
        if all([checkin.mood, checkin.energy, checkin.objectives]):
            # Add timestamp
            from datetime import datetime
            checkin.timestamp = datetime.now().isoformat()
            
            # Create summary
            checkin.summary = f"User is feeling {checkin.mood} with {checkin.energy} energy. Goals: {', '.join(checkin.objectives)}"
            
            # Save to JSON
            self._save_checkin(checkin)
            
            return f"""Perfect! Let me recap:
            - Mood: {checkin.mood}
            - Energy: {checkin.energy}
            - Today's objectives: {', '.join(checkin.objectives)}
            
            Does this sound right?"""
        return "Got it!"
    
    def _save_checkin(self, checkin: WellnessCheckIn):
        """Save check-in to wellness_log.json"""
        try:
            # Load existing data
            try:
                with open("wellness_log.json", "r") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {"sessions": []}
            
            # Append new session
            data["sessions"].append(asdict(checkin))
            
            # Save back to file
            with open("wellness_log.json", "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved check-in to wellness_log.json")
        except Exception as e:
            logger.error(f"Error saving check-in: {e}")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession[WellnessCheckIn](
        userdata=WellnessCheckIn(),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
