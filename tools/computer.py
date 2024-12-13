from typing import Literal
from dataclasses import dataclass, fields, replace
import pyautogui
from io import BytesIO

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "screenshot",
]

class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message

@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """Represents the result of a tool execution."""

    output: str | None = None
    error: str | None = None
    image: bytes | None = None

    def __bool__(self):
        return any(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            image=combine_fields(self.image, other.image, False),
        )

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        return replace(self, **kwargs)

def get_screenshot():
    """
    Capture a screenshot and return its bytes and dimensions.
    """
    screenshot = pyautogui.screenshot()
    buffer = BytesIO()
    screenshot.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    buffer.close()
    return image_bytes


class ComputerTool:
    def __init__(self):
        pass

    def __call__(
            self,
            action: Action,
            text: str | None = None,
            coordinate: tuple[int, int] | None = None,
    ):
        print(f'Action: {action}')
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            if action == "mouse_move":
                pyautogui.moveTo(coordinate)
            elif action == "left_click_drag":
                pyautogui.dragTo(coordinate, button="left")

            return ToolResult()

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(f"{text} must be a string")

            if action == "key":
                if text.lower() == 'return':
                    text = 'enter'
                pyautogui.press(text)
            elif action == "type":
                pyautogui.write(text)

            return ToolResult()

        if action in (
                "left_click",
                "screenshot",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            screenshot = None
            if action == "screenshot":
                screenshot = get_screenshot()
            elif action == "left_click":
               pyautogui.click()

            return ToolResult(output=None, error=None, image=screenshot)

        raise ToolError(f"Invalid action: {action}")