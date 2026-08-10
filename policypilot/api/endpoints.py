import logging

from fastapi import APIRouter, HTTPException, Request

from policypilot.core.models import ChatRequest, ChatResponse, GeoDataResponse
from policypilot.services.zip_client import get_geo_data_from_zip


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/geodata/{zip_code}", response_model=GeoDataResponse)
def get_geolocation_data(zip_code: str):
    if not zip_code.isdigit() or len(zip_code) != 5:
        raise HTTPException(status_code=400, detail="Invalid ZIP code format.")
    geo_data = get_geo_data_from_zip(zip_code)
    if not geo_data:
        raise HTTPException(status_code=404, detail="Could not find location data.")
    return GeoDataResponse(
        zip_code=zip_code,
        county=geo_data.county.replace(" County", ""),
        city=geo_data.city,
        state=geo_data.state,
        state_abbreviation=geo_data.state_abbr,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    """Run one bounded PolicyPilot graph turn without logging sensitive profile data."""

    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="PolicyPilot is not ready.")

    logger.info("Processing chat turn for thread_id=%s", body.thread_id)
    inputs = {
        "user_profile": body.user_profile,
        "user_message": body.message,
        "is_profile_complete": body.is_profile_complete,
        "conversation_history": body.conversation_history,
        "web_search_attempted": False,
        "sources": [],
    }
    try:
        final_state = orchestrator.invoke(
            inputs,
            config={"configurable": {"thread_id": body.thread_id}},
        )
    except Exception:
        logger.exception("PolicyPilot graph execution failed for thread_id=%s", body.thread_id)
        raise HTTPException(status_code=500, detail="The PolicyPilot advisor could not complete this request.")

    return ChatResponse(
        agent_response=final_state.get("generation")
        or "PolicyPilot could not generate a grounded response.",
        updated_profile=final_state.get("user_profile", body.user_profile),
        updated_history=final_state.get("conversation_history", body.conversation_history),
        is_profile_complete=final_state.get("is_profile_complete", body.is_profile_complete),
        sources=final_state.get("sources", []),
    )
