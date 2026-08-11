import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from policypilot.config import settings
from policypilot.prompts.prompt_loader import load_prompt

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfileBuilder:
    """
    A service class that manages the entire conversational profile building process.
    It determines the next question and intelligently updates the profile with user answers.
    """

    # Initializes the ProfileBuilder, loading all necessary prompts.
    def __init__(self, llm):
        """Initializes the ProfileBuilder, loading all necessary prompts."""
        self.llm = llm
        try:
            self.question_prompt = load_prompt("profile_agent")
            self.updater_prompt = load_prompt("profile_updater")
            logger.info("ProfileBuilder initialized successfully with all prompts.")
        except FileNotFoundError as e:
            logger.critical(f"A required prompt file was not found: {e}. ProfileBuilder cannot function.")
            raise

    # Analyzes the user's current profile and determines the next question to ask,.
    def get_next_question(self, current_profile: Dict[str, Any], conversation_history: Dict[str, Any]) -> str:
        """
        Analyzes the user's current profile and determines the next question to ask,
        or signals that the profile is complete.

        Args:
            current_profile: A dictionary representing the user's profile data.

        Returns:
            A string containing the next question for the user, or "PROFILE_COMPLETE".
        """
        logger.debug("Determining the next profile question")
        profile_json_str = json.dumps(current_profile, indent=2)
        conversation_history_str = json.dumps(conversation_history, indent=2)
        full_prompt = f"{self.question_prompt}\n\n### User Profile\n{profile_json_str} \n\n ### Conversation History{conversation_history_str}"

        try:
            response = self.llm.invoke(full_prompt)
            next_step = response.content.strip()
            logger.info("Generated the next profile step")
            return next_step
        except Exception as e:
            logger.error(f"LLM error during question generation: {e}")
            return "I'm sorry, I'm having a little trouble right now. Could we try that again?"

    # Uses an LLM to intelligently update the user's profile with their latest answer.
    def update_profile_with_answer(
        self,
        current_profile: Dict[str, Any],
        last_question: str,
        user_answer: str
    ) -> Dict[str, Any]:
        """
        Uses an LLM to intelligently update the user's profile with their latest answer.

        Args:
            current_profile: The user's profile before the update.
            last_question: The question the user is answering.
            user_answer: The user's free-text answer.

        Returns:
            The updated profile dictionary.
        """
        logger.debug("Updating the profile from a user response")
        profile_json_str = json.dumps(current_profile, indent=2)

        # Construct the prompt for the updater LLM call
        full_prompt = (
            f"{self.updater_prompt}\n\n"
            f"current_profile: {profile_json_str}\n\n"
            f"last_question_asked: \"{last_question}\"\n\n"
            f"user_answer: \"{user_answer}\""
        )

        try:
            response = self.llm.invoke(full_prompt)
            response_content = response.content.strip()

            # Clean the response to ensure it's valid JSON
            # The LLM can sometimes wrap the JSON in markdown
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()

            updated_profile = json.loads(response_content)
            logger.info("Successfully updated profile with user's answer.")
            return updated_profile
        except json.JSONDecodeError as e:
            logger.error("Failed to decode the profile provider response: %s", e)
            # Return the original profile to avoid data corruption
            return current_profile
        except Exception as e:
            logger.error(f"LLM error during profile update: {e}")
            # Return the original profile on error
            return current_profile

    # Executes a full turn of the profile-building conversation.
    def run_conversation_turn(
        self,
        current_profile: Dict[str, Any],
        last_user_answer: str = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Executes a full turn of the profile-building conversation.

        Args:
            current_profile: The current state of the user's profile.
            last_user_answer: The user's answer from the previous turn, if any.

        Returns:
            A tuple containing:
            - The updated profile dictionary.
            - The next question to ask the user (or "PROFILE_COMPLETE").
        """
        # First, determine the question that *would have been asked* for the current state
        # This is the question the user just answered.
        question_that_was_asked = self.get_next_question(current_profile, [])

        profile_after_update = current_profile
        # If there was an answer from the user, update the profile
        if last_user_answer and question_that_was_asked != "PROFILE_COMPLETE":
            profile_after_update = self.update_profile_with_answer(
                current_profile=current_profile,
                last_question=question_that_was_asked,
                user_answer=last_user_answer
            )

        # Now, with the potentially updated profile, get the *next* question
        next_question_to_ask = self.get_next_question(profile_after_update, [])

        return profile_after_update, next_question_to_ask
