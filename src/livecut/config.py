from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    google_genai_use_vertexai: bool = Field(default=False, alias="GOOGLE_GENAI_USE_VERTEXAI")
    google_cloud_project: str | None = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="us-central1", alias="GOOGLE_CLOUD_LOCATION")
    live_model: str = Field(default="gemini-2.5-flash-preview-native-audio-dialog", alias="LIVE_MODEL")

    obs_host: str = Field(default="127.0.0.1", alias="OBS_HOST")
    obs_port: int = Field(default=4455, alias="OBS_PORT")
    obs_password: str = Field(default="", alias="OBS_PASSWORD")
    scene_gameplay_focus: str = Field(default="Gameplay_Focus", alias="SCENE_GAMEPLAY_FOCUS")
    scene_chatting_focus: str = Field(default="Chatting_Focus", alias="SCENE_CHATTING_FOCUS")
    input_host_mic: str = Field(default="Host Mic", alias="INPUT_HOST_MIC")
    source_sfx_airhorn: str = Field(default="SFX_Airhorn", alias="SOURCE_SFX_AIRHORN")
    source_lower_third_text: str = Field(default="LowerThirdText", alias="SOURCE_LOWER_THIRD_TEXT")
    source_host_prompt_text: str = Field(default="HostPromptText", alias="SOURCE_HOST_PROMPT_TEXT")
    source_chat_question_text: str = Field(default="ChatQuestionText", alias="SOURCE_CHAT_QUESTION_TEXT")

    segment_max_minutes: int = Field(default=20, alias="SEGMENT_MAX_MINUTES")
    chat_batch_seconds: int = Field(default=60, alias="CHAT_BATCH_SECONDS")
    cough_recovery_seconds: float = Field(default=1.0, alias="COUGH_RECOVERY_SECONDS")
    host_username: str = Field(default="Tanish", alias="HOST_USERNAME")
    enable_gemini: bool = Field(default=False, alias="ENABLE_GEMINI")


settings = Settings()
