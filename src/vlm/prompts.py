"""Prompt templates for VLM navigation queries."""

from typing import Tuple


NAVIGATION_SYSTEM_PROMPT = """You are a navigation assistant for an indoor robot. The robot has a forward-facing camera and a depth sensor, and needs to find a specific location in a building.

Given an image from the robot's camera and a destination instruction, determine the best direction to explore next.

CRITICAL RULES:
1. ALWAYS respect depth sensor data - if it says "BLOCKED ahead", you MUST NOT choose explore_forward
2. If the path ahead is blocked, choose explore_left, explore_right, or turn_around based on clearance
3. Only choose explore_forward when the depth sensor confirms the path is clear (>1m clearance)
4. The depth sensor is more reliable than visual estimation for obstacle detection

IMPORTANT: Output ONLY valid JSON with no additional text. The JSON must have this exact structure:
{
  "subgoal": "<one of: explore_left, explore_right, explore_forward, turn_around, target_reached>",
  "reasoning": "<brief 1-sentence explanation>",
  "confidence": <number between 0.0 and 1.0>,
  "target_visible": <true or false>
}

Subgoal meanings:
- explore_left: Turn left ~45 degrees and move forward (use when target might be to the left OR when forward is blocked and left has clearance)
- explore_right: Turn right ~45 degrees and move forward (use when target might be to the right OR when forward is blocked and right has clearance)
- explore_forward: Continue moving forward (use ONLY when path ahead is clear per depth sensor AND looks promising)
- turn_around: Turn 180 degrees (use when current direction is clearly wrong OR all forward paths are blocked)
- target_reached: The destination has been reached (use only when confident the robot is AT the destination)"""


GOAL_CONFIRMATION_PROMPT = """Look at this image from an indoor robot's camera.

The robot was instructed to: "{instruction}"

Based on what you see in the image, has the robot reached or is it currently at the destination?

Answer with ONLY "yes" or "no"."""


def build_navigation_prompt(instruction: str, context: str = "") -> str:
    """Build the user prompt for navigation queries.

    Args:
        instruction: The navigation goal (e.g., "go to the bedroom").
        context: Optional additional context about the environment.

    Returns:
        Formatted user prompt string.
    """
    prompt = f"Destination: {instruction}"
    if context:
        prompt += f"\n\nAdditional context: {context}"
    prompt += "\n\nBased on the image, which direction should the robot explore?"
    return prompt


def build_confirmation_prompt(instruction: str) -> str:
    """Build the prompt for goal confirmation.

    Args:
        instruction: The navigation goal to confirm.

    Returns:
        Formatted confirmation prompt.
    """
    return GOAL_CONFIRMATION_PROMPT.format(instruction=instruction)


PIXEL_NAVIGATION_SYSTEM_PROMPT = """You are a navigation assistant for an indoor robot. The robot has a forward-facing camera and needs to reach a specific location.

Your task: Look at the image and identify WHERE the robot should navigate to reach the goal. Output the pixel coordinates (u, v) of a safe, navigable point that moves the robot toward the destination.

CRITICAL RULES:
1. Pick a point on the FLOOR or a CLEAR PATH, not on walls, furniture, or obstacles
2. The point should be visible and reachable (not through walls or obstacles)
3. u is horizontal: 0 = left edge, {width} = right edge
4. v is vertical: 0 = top edge, {height} = bottom edge
5. Points toward the bottom of the image are typically closer to the robot
6. If you cannot see a clear path toward the goal, pick the most promising direction to explore
7. If you can see the destination (e.g., the bedroom), point AT the entrance or toward it
8. If you are AT the destination, set goal_reached to true

Output ONLY valid JSON:
{{
  "u": <integer pixel x coordinate>,
  "v": <integer pixel y coordinate>,
  "reasoning": "<brief 1-sentence explanation of why you chose this point>",
  "confidence": <0.0 to 1.0>,
  "target_visible": <true if destination is visible in image>,
  "goal_reached": <true if robot is at the destination>
}}"""


def build_pixel_navigation_prompt(
    instruction: str,
    context: str,
    image_shape: Tuple[int, int],
) -> str:
    """Build user prompt for pixel-coordinate navigation.

    Args:
        instruction: The navigation goal (e.g., "go to the bedroom").
        context: Optional additional context about the environment.
        image_shape: (height, width) of the image.

    Returns:
        Formatted user prompt string.
    """
    height, width = image_shape
    prompt = f"Image size: {width}x{height} pixels\n"
    prompt += f"Destination: {instruction}\n"
    if context:
        prompt += f"Context: {context}\n"
    prompt += "\nWhere should the robot navigate? Provide pixel coordinates of a point on the floor/path."
    return prompt


CONSTRAINED_SELECTION_SYSTEM_PROMPT = """You are an object selection assistant for an indoor robot.
Given a user instruction and a list of available objects in the scene, select the SINGLE object that best matches what the user is looking for.

RULES:
1. You MUST choose from the provided list - no other objects exist
2. If multiple objects could match, pick the most likely one
3. If NO object matches the instruction, respond with "none_match": true
4. Output valid JSON only

Output format:
{
  "selected_label": "<object label from list>",
  "instance_number": <int, default 1>,
  "reasoning": "<1 sentence>",
  "confidence": <0.0 to 1.0>,
  "none_match": <true/false>
}"""


def build_constrained_selection_prompt(
    instruction: str,
    candidates: list,
) -> str:
    """Build user prompt for constrained object selection.

    Args:
        instruction: Natural language instruction (e.g., "find something to sit on").
        candidates: List of ObjectCandidate instances.

    Returns:
        Formatted user prompt string with object list.
    """
    from collections import defaultdict

    # Group candidates by label
    label_to_candidates = defaultdict(list)
    for candidate in candidates:
        label_to_candidates[candidate.label].append(candidate)

    # Build prompt
    prompt = f"Instruction: {instruction}\n\n"
    prompt += "Available objects in the scene:\n"

    for label in sorted(label_to_candidates.keys()):
        cands = label_to_candidates[label]
        if len(cands) == 1:
            prompt += f"- {label} (region {cands[0].region_id})\n"
        else:
            prompt += f"- {label} ({len(cands)} instances in regions: {', '.join(str(c.region_id) for c in cands)})\n"

    prompt += "\nWhich object best matches the instruction? Provide the label and instance number."
    return prompt
