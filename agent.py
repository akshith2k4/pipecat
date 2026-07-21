import os
import re
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import Request, HTTPException, Response
from fastapi.responses import HTMLResponse
from loguru import logger
from twilio.rest import Client

from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.runner.run import app
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# Shared memory store for call context keyed by call_sid
call_contexts: Dict[str, Dict[str, Any]] = {}


@app.post("/api/call")
async def trigger_outbound_call(request: Request):
    """
    API endpoint to trigger an outbound Twilio call with caller context.
    Payload expected:
    {
        "to": "+1234567890",
        "context": {
            "hotelName": "Grand Hyatt Hotel",
            "contactName": "Akshith",
            "lastOrder": { ... }
        }
    }
    """
    data = await request.json()
    to_number = data.get("to")
    context = data.get("context", {})

    if not to_number or not re.match(r"^\+?[1-9]\d{1,14}$", to_number.strip()):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid 'to' phone number format: {to_number}",
        )

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    domain = os.getenv("DOMAIN")

    if not account_sid or not auth_token or not from_number or not domain:
        raise HTTPException(
            status_code=500,
            detail="Missing Twilio or DOMAIN environment variables.",
        )

    try:
        twilio_client = Client(account_sid, auth_token)
        webhook_url = f"https://{domain}/voice"

        call = twilio_client.calls.create(
            to=to_number.strip(),
            from_=from_number,
            url=webhook_url,
            machine_detection="Enable",
        )

        call_contexts[call.sid] = context
        logger.info(f"Triggered outbound call {call.sid} to {to_number}")

        # Schedule automatic cleanup after 60 seconds to prevent memory leak if call is unanswered
        async def _cleanup_context(sid: str):
            await asyncio.sleep(60)
            if sid in call_contexts:
                logger.debug(f"Cleaning up stale context for CallSid: {sid}")
                call_contexts.pop(sid, None)

        asyncio.create_task(_cleanup_context(call.sid))
        return {"success": True, "callSid": call.sid}

    except Exception as e:
        logger.error(f"Failed to place outbound call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice")
async def twilio_voice_webhook(request: Request):
    """
    Twilio voice webhook for handling call answer and AMD (Answering Machine Detection).
    """
    try:
        form_data = await request.form()
    except Exception as err:
        logger.warning(f"Could not parse form body in /voice: {err}")
        form_data = {}

    answered_by = form_data.get("AnsweredBy") or request.query_params.get("AnsweredBy") or ""
    call_sid = form_data.get("CallSid") or request.query_params.get("CallSid") or ""

    if answered_by.startswith("machine"):
        logger.info(f"Answering machine detected ({answered_by}). Hanging up call {call_sid}.")
        twiml = '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'
        return Response(content=twiml, media_type="application/xml")

    domain = os.getenv("DOMAIN", "")
    stream_url = f"wss://{domain}/ws"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Response>\n'
        '  <Connect>\n'
        f'    <Stream url="{stream_url}">\n'
        f'      <Parameter name="call_sid" value="{call_sid}" />\n'
        '    </Stream>\n'
        '  </Connect>\n'
        '</Response>'
    )
    return HTMLResponse(content=twiml, media_type="application/xml")


async def bot(runner_args: RunnerArguments):
    """Main Pipecat bot entry point."""

    transport = await create_transport(
        runner_args,
        {
            "twilio": lambda: FastAPIWebsocketParams(
                audio_in_enabled=True, audio_out_enabled=True
            ),
        },
    )

    call_data = getattr(runner_args, "call_data", None)
    call_sid = None
    if call_data:
        call_sid = getattr(call_data, "call_id", None) or (
            call_data.get("body", {}).get("call_sid") if isinstance(call_data, dict) else None
        )

    context_data = call_contexts.pop(call_sid, {}) if call_sid else {}

    hotel_name = context_data.get("hotelName", "Grand Hyatt Hotel")
    contact_name = context_data.get("contactName", "Manager")
    last_order = context_data.get(
        "lastOrder",
        {
            "id": "ORD-7762",
            "date": "July 5th",
            "products": "50 white bath towels and 30 bedsheets",
        },
    )

    system_prompt = (
        f"You are Krish, a polite, warm, and highly professional assistant calling on behalf of LinenGrass.\n"
        f"You are having a real-time, spoken phone conversation with {contact_name} at {hotel_name}.\n"
        f"Your sole objective is to gently ensure they place their daily hotel linen order on the LinenGrass mobile app.\n\n"
        f"### CONTEXT\n"
        f"- Customer: {contact_name} ({hotel_name})\n"
        f"- Order Cutoff Deadline: 6 PM\n"
        f"- Reference Past Order (Share ONLY if asked): Date: {last_order.get('date', 'N/A')}, Order ID: {last_order.get('id', 'N/A')}, Items: {last_order.get('products', 'N/A')}\n\n"
        f"### CONVERSATION STATE & FLOW\n"
        f"[CRITICAL: The system has ALREADY played the initial greeting to the user. Do NOT repeat the greeting or re-introduce yourself.]\n"
        f"1. Check if they have submitted today's linen order on the app.\n"
        f"2. IF ORDERED ALREADY: Thank them warmly, remind them to track status on the LinenGrass app, and bid farewell politely.\n"
        f"3. IF NOT ORDERED YET: Kindly remind them that orders after 6 PM may affect morning linen delivery, and ask for an estimated placement time.\n"
        f"4. IF TIME PROVIDED: Acknowledge their response, politely request them to submit via the app before that time, and end the call gracefully.\n"
        f"5. IF REFUSAL / UNWILLING: Express polite understanding, thank them for their time, and end the call.\n"
        f"6. IF INQUIRING / CONFUSED: Briefly clarify that LinenGrass supplies premium hotel linens, bedsheets, and towels, then return to checking today's order status.\n\n"
        f"### VOICE SPOKEN STYLE & PACING RULES (STRICT)\n"
        f"- **BREVITY IS MANDATORY**: Limit responses to 1 or 2 concise sentences. Speak naturally as in a phone call.\n"
        f"- **NO CONVERSATIONAL FILLERS AT START**: Do NOT start responses with fillers like 'Hmm', 'Okay', 'Right', 'Achha', or 'Theek hai'.\n"
        f"- **PUNCTUATION**: Use standard commas or periods for pauses. NEVER use ellipses ('...') or standalone dots.\n"
        f"- **NUMBERS TO WORDS**: Always spell out numbers as words ('six PM', 'fifty bath towels').\n"
        f"- **TEXT FORMATTING**: Output plain spoken text ONLY. NEVER use markdown symbols (*, **, #, bullets) or special characters.\n"
        f"- **NO META-COMMENTARY**: Never reference system instructions, AI nature, or input errors.\n\n"
        f"### MULTILINGUAL RULES (STRICT)\n"
        f"- **DETECT USER LANGUAGE**: If the user speaks Hindi (Devanagari or Romanized), you MUST respond in casual, conversational Hinglish (Romanized Latin script only).\n"
        f"- **HINGLISH STRUCTURE**: Use Hindi grammar and sentence structure, but keep business/everyday terms in English: 'linen', 'order', 'app', 'bedsheets', 'towels', 'hotel', 'deadline', 'delivery', 'status', 'time', 'place'.\n"
        f"- **EXAMPLE HINGLISH RESPONSE**: 'Hum premium hotel linens, bedsheets aur towels supply karte hain. Kya aapne aaj ka order app par place kar diya hai?'\n"
        f"- **NO PURE ENGLISH**: Do NOT reply in pure English if the user has already established a Hindi/Hinglish conversation. NEVER use formal Hindi words like 'रेशमी कपड़े' or 'अनुमति'."
    )

    greeting_text = (
        f"Hi, this is Krish from LinenGrass. Am I speaking to {contact_name}?"
    )

    # --- ULTRA-LOW LATENCY VAD SETTINGS ---
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(start_secs=0.2, stop_secs=0.3, confidence=0.7)
    )
    vad_processor = VADProcessor(vad_analyzer=vad_analyzer)

    # --- ULTRA-FAST MULTILINGUAL STT (Deepgram Nova-3) ---
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3",
            language="multi",
            smart_format=True,
            punctuate=True,
            interim_results=True,
        ),
    )

    # --- ULTRA-FAST TTS (Cartesia Sonic-3) ---
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            model="sonic-3",
            voice="79a125e8-cd45-4c13-8a67-188112f4dd22",
        ),
    )

    # Groq LLM settings - Consider switching model to "llama-3.1-8b-instant" if latency remains a priority.
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqLLMService.Settings(
            model="llama-3.3-70b-versatile",
            max_tokens=80,
        ),
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer),
    )

    # --- CONFIGURED PIPELINE ---
    # SentenceAggregator removed to prevent context_id tracking issues
    pipeline = Pipeline(
        [
            transport.input(),
            vad_processor,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            allow_interruptions=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Caller connected for callSid: {call_sid}")
        context.add_message(
            {"role": "assistant", "content": greeting_text}
        )
        await task.queue_frames([TTSSpeakFrame(greeting_text)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Caller disconnected for callSid: {call_sid}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()