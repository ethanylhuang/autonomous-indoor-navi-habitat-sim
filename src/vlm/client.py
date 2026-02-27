"""VLM API client wrapper for Anthropic Claude."""

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.vlm.prompts import (
    CONSTRAINED_SELECTION_SYSTEM_PROMPT,
    NAVIGATION_SYSTEM_PROMPT,
    PIXEL_NAVIGATION_SYSTEM_PROMPT,
    build_confirmation_prompt,
    build_constrained_selection_prompt,
    build_navigation_prompt,
    build_pixel_navigation_prompt,
)

logger = logging.getLogger(__name__)


class Subgoal(Enum):
    """Semantic navigation subgoals output by VLM."""

    EXPLORE_LEFT = "explore_left"
    EXPLORE_RIGHT = "explore_right"
    EXPLORE_FORWARD = "explore_forward"
    TURN_AROUND = "turn_around"
    TARGET_REACHED = "target_reached"


@dataclass
class VLMResponse:
    """Response from VLM navigation query."""

    subgoal: Subgoal
    reasoning: str
    confidence: float
    target_visible: bool


@dataclass
class PixelTarget:
    """Pixel coordinate target from VLM."""

    u: int  # horizontal pixel (0 = left edge)
    v: int  # vertical pixel (0 = top edge)
    reasoning: str
    confidence: float
    is_valid: bool  # False if VLM output was unparseable
    target_visible: bool  # VLM believes destination is in view


@dataclass
class WorldTarget:
    """3D world coordinate derived from pixel target."""

    position: NDArray[np.float64]  # [3] world XYZ
    navmesh_position: NDArray[np.float64]  # [3] snapped NavMesh point
    depth_value: float  # meters
    is_valid: bool  # False if depth invalid or snap failed
    failure_reason: Optional[str]  # "depth_invalid", "snap_failed", "out_of_range"


@dataclass
class VLMPixelResponse:
    """Response from VLM pixel-based navigation query."""

    pixel: PixelTarget
    goal_reached: bool  # VLM says we're at destination


class VLMClient:
    """Wrapper for Anthropic Claude API with vision capabilities."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        """Initialize the VLM client.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            model: Model ID to use for inference.

        Raises:
            ValueError: If no API key is available.
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._model = model
        self._client = None  # Lazy initialization

    def _get_client(self):
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _encode_image(self, image: NDArray[np.uint8]) -> str:
        """Encode RGB image as base64 JPEG.

        Args:
            image: [H, W, 3] or [H, W, 4] uint8 RGB(A) image.

        Returns:
            Base64-encoded JPEG string.
        """
        from PIL import Image
        import io

        # Handle RGBA by dropping alpha channel
        if image.shape[-1] == 4:
            image = image[:, :, :3]

        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_navigation_subgoal(
        self,
        image: NDArray[np.uint8],
        instruction: str,
        context: str = "",
    ) -> VLMResponse:
        """Query VLM for the next navigation subgoal.

        Args:
            image: [H, W, 3] or [H, W, 4] uint8 RGB(A) image from robot camera.
            instruction: Navigation goal (e.g., "go to the bedroom").
            context: Optional additional context about the environment.

        Returns:
            VLMResponse with subgoal and metadata.
        """
        client = self._get_client()
        image_b64 = self._encode_image(image)
        user_prompt = build_navigation_prompt(instruction, context)

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=256,
                system=NAVIGATION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                        ],
                    }
                ],
            )

            # Parse JSON response
            response_text = response.content[0].text.strip()
            logger.debug("VLM raw response: %s", response_text)

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            data = json.loads(response_text)

            # Map string to enum
            subgoal_str = data.get("subgoal", "explore_forward").lower()
            try:
                subgoal = Subgoal(subgoal_str)
            except ValueError:
                logger.warning("Unknown subgoal '%s', defaulting to explore_forward", subgoal_str)
                subgoal = Subgoal.EXPLORE_FORWARD

            return VLMResponse(
                subgoal=subgoal,
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.5)),
                target_visible=bool(data.get("target_visible", False)),
            )

        except json.JSONDecodeError as e:
            logger.error("Failed to parse VLM JSON response: %s", e)
            return VLMResponse(
                subgoal=Subgoal.EXPLORE_FORWARD,
                reasoning="Parse error, defaulting to forward",
                confidence=0.0,
                target_visible=False,
            )
        except Exception as e:
            logger.error("VLM query failed: %s", e)
            return VLMResponse(
                subgoal=Subgoal.EXPLORE_FORWARD,
                reasoning=f"Query error: {e}",
                confidence=0.0,
                target_visible=False,
            )

    def confirm_goal_reached(
        self,
        image: NDArray[np.uint8],
        instruction: str,
    ) -> bool:
        """Ask VLM if the navigation goal has been reached.

        Args:
            image: [H, W, 3] or [H, W, 4] uint8 RGB(A) image from robot camera.
            instruction: The navigation goal to confirm.

        Returns:
            True if VLM confirms the goal is reached, False otherwise.
        """
        client = self._get_client()
        image_b64 = self._encode_image(image)
        prompt = build_confirmation_prompt(instruction)

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=16,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            response_text = response.content[0].text.strip().lower()
            logger.debug("VLM confirmation response: %s", response_text)
            return response_text.startswith("yes")

        except Exception as e:
            logger.error("VLM confirmation query failed: %s", e)
            return False

    def get_pixel_target(
        self,
        image: NDArray[np.uint8],
        instruction: str,
        context: str = "",
    ) -> VLMPixelResponse:
        """Query VLM for pixel coordinate navigation target.

        Args:
            image: [H, W, 3] or [H, W, 4] uint8 RGB(A) image from robot camera.
            instruction: Navigation goal (e.g., "go to the bedroom").
            context: Optional additional context about the environment.

        Returns:
            VLMPixelResponse with pixel target or goal_reached=True.
        """
        client = self._get_client()
        image_b64 = self._encode_image(image)
        height, width = image.shape[:2]
        user_prompt = build_pixel_navigation_prompt(instruction, context, (height, width))

        system_prompt = PIXEL_NAVIGATION_SYSTEM_PROMPT.format(width=width, height=height)

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=256,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                        ],
                    }
                ],
            )

            response_text = response.content[0].text.strip()
            logger.debug("VLM pixel response: %s", response_text)

            # Extract JSON from response
            json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            data = json.loads(json_match.group())

            if data.get("goal_reached", False):
                return VLMPixelResponse(
                    pixel=PixelTarget(
                        0, 0, data.get("reasoning", "At destination"), 1.0, False, True
                    ),
                    goal_reached=True,
                )

            u = int(data.get("u", width // 2))
            v = int(data.get("v", height // 2))

            # Clamp to valid range
            u = max(0, min(u, width - 1))
            v = max(0, min(v, height - 1))

            return VLMPixelResponse(
                pixel=PixelTarget(
                    u=u,
                    v=v,
                    reasoning=data.get("reasoning", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    is_valid=True,
                    target_visible=data.get("target_visible", False),
                ),
                goal_reached=False,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("VLM pixel parse error: %s, defaulting to center", e)
            return VLMPixelResponse(
                pixel=PixelTarget(
                    u=width // 2,
                    v=height // 2,
                    reasoning=f"Parse error: {e}, defaulting to center",
                    confidence=0.0,
                    is_valid=False,
                    target_visible=False,
                ),
                goal_reached=False,
            )

    def select_object_constrained(
        self,
        instruction: str,
        candidates: list,
    ):
        """Query VLM for constrained object selection from candidate list.

        Args:
            instruction: Natural language instruction (e.g., "find something to sit on").
            candidates: List of ObjectCandidate instances.

        Returns:
            ConstrainedVLMResponse with selected object or no_match indicator.
        """
        from src.vlm.constrained import ConstrainedVLMResponse

        client = self._get_client()
        user_prompt = build_constrained_selection_prompt(instruction, candidates)

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=256,
                system=CONSTRAINED_SELECTION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )

            response_text = response.content[0].text.strip()
            logger.debug("VLM constrained selection raw response: %s", response_text)

            # Parse JSON response
            return self._parse_constrained_response(response_text, candidates)

        except Exception as e:
            logger.error("VLM constrained selection query failed: %s", e)
            return ConstrainedVLMResponse(
                selected_object_id=None,
                selected_label="",
                reasoning=f"Query error: {e}",
                confidence=0.0,
                is_valid=False,
                no_match_reason="api_error",
            )

    def _parse_constrained_response(
        self,
        response_text: str,
        candidates: list,
    ):
        """Parse VLM JSON response for constrained object selection.

        Args:
            response_text: Raw VLM response text.
            candidates: List of ObjectCandidate instances.

        Returns:
            ConstrainedVLMResponse.
        """
        from src.vlm.constrained import ConstrainedVLMResponse

        try:
            # Handle markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            data = json.loads(response_text)

            # Check for no_match flag
            if data.get("none_match", False):
                return ConstrainedVLMResponse(
                    selected_object_id=None,
                    selected_label="",
                    reasoning=data.get("reasoning", "No matching object"),
                    confidence=float(data.get("confidence", 0.0)),
                    is_valid=False,
                    no_match_reason="none_match",
                )

            # Extract selected label and instance number
            selected_label = data.get("selected_label", "").lower().strip()
            instance_number = int(data.get("instance_number", 1))
            reasoning = data.get("reasoning", "")
            confidence = float(data.get("confidence", 0.5))

            if not selected_label:
                return ConstrainedVLMResponse(
                    selected_object_id=None,
                    selected_label="",
                    reasoning="Empty label in response",
                    confidence=0.0,
                    is_valid=False,
                    no_match_reason="empty_label",
                )

            # Match label to candidates
            matching_candidates = [
                c for c in candidates if c.label.lower() == selected_label
            ]

            if not matching_candidates:
                logger.warning(
                    "VLM selected label '%s' not in candidate list", selected_label
                )
                return ConstrainedVLMResponse(
                    selected_object_id=None,
                    selected_label=selected_label,
                    reasoning=reasoning,
                    confidence=confidence,
                    is_valid=False,
                    no_match_reason="label_not_found",
                )

            # Select specific instance (clamp to available instances)
            instance_idx = min(instance_number - 1, len(matching_candidates) - 1)
            instance_idx = max(0, instance_idx)
            selected = matching_candidates[instance_idx]

            return ConstrainedVLMResponse(
                selected_object_id=selected.object_id,
                selected_label=selected.label,
                reasoning=reasoning,
                confidence=confidence,
                is_valid=True,
                no_match_reason=None,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse VLM constrained response: %s", e)
            return ConstrainedVLMResponse(
                selected_object_id=None,
                selected_label="",
                reasoning=f"Parse error: {e}",
                confidence=0.0,
                is_valid=False,
                no_match_reason="parse_error",
            )
