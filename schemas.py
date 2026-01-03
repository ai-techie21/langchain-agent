from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by an agent"""

    url: str = Field(description="Source URL")


class AgentResponse(BaseModel):
    """Schema for agent response with an answer and source"""

    answer: str = Field(description="Response from an agent")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used by an agent"
    )
