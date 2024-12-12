import time
import boto3
from botocore.config import Config
import pyautogui
from io import BytesIO
import json
import base64

def get_screenshot(region=None):
    """
    Capture a screenshot and return its base64 encoded string.
    """
    screenshot = pyautogui.screenshot(region=region)
    buffer = BytesIO()
    screenshot.save(buffer, format='PNG')
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    return base64_image

def send_to_bedrock(client, messages):
    response = client.invoke_model(
        modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "anthropic_beta": ["computer-use-2024-10-22"],
            "messages": messages,
            "max_tokens": 4096,
            "system": """You are a helpful AI agent with access to computer control. You can:
                       1. Move the mouse using computer.mouse_move(x, y)
                       2. Click using computer.click()
                       3. Take screenshots using computer.screenshot()

                       Always think step by step and explain your actions.

                       ENVIRONMENT:
                       1. macOS
                       2. Resolution: 3840x1080

                       IMPORTANT: Application is already running and should be visible on a screenshot""",
        })
    )
    return json.loads(response['body'].read())

def parse_tool_calls(response_text):
    """
    Parse tool calls from the model's response text.
    """
    tool_calls = []
    lines = response_text.split('\n')

    for line in lines:
        if "computer.mouse_move" in line:
            try:
                coords = line.split("computer.mouse_move(")[1].split(")")[0]
                x, y = map(int, coords.split(","))
                tool_calls.append(("mouse_move", (x, y)))
            except Exception as e:
                print(f"Error parsing mouse_move command: {e}")
        elif "computer.click()" in line:
            tool_calls.append(("click", None))
        elif "computer.screenshot()" in line:
            tool_calls.append(("screenshot", None))

    return tool_calls


def execute_tool_calls(tool_calls):
    """
    Execute the parsed tool calls.
    """
    took_screenshot = False

    for action, params in tool_calls:
        if action == "mouse_move":
            x, y = params
            print(f"Moving mouse to: {x}, {y}")
            pyautogui.moveTo(x, y)
        elif action == "click":
            print("Clicking")
            pyautogui.click()
        elif action == "screenshot":
            print("Taking screenshot")
            took_screenshot = True

    return took_screenshot


def main():
    client = boto3.client('bedrock-runtime', config=Config(region_name='us-west-2'))
    initial_message = "Calculate 1 + 2 by using calculator app"
    messages = []

    try:
        while True:
            # Get initial screenshot
            screenshot = get_screenshot()

            # If messages is empty, add initial user message with screenshot
            if not messages:
                messages = [{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': initial_message
                        },
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/png',
                                'data': screenshot
                            }
                        }
                    ]
                }]

            # Get model response
            response = send_to_bedrock(client, messages)
            response_text = response['content'][0]['text']
            print("Model response:", response_text)

            # Parse and execute tool calls
            tool_calls = parse_tool_calls(response_text)
            took_screenshot = execute_tool_calls(tool_calls)

            # Add assistant message
            messages.append({
                'role': 'assistant',
                'content': response_text
            })

            # If we took a new screenshot, add it to the next message
            if took_screenshot:
                new_screenshot = get_screenshot()
                messages.append({
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'Here is the current screenshot. Please continue.'
                        },
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/png',
                                'data': new_screenshot
                            }
                        }
                    ]
                })

            # Check if task is complete
            if "task is complete" in response_text.lower():
                print("Task completed!")
                break

    except KeyboardInterrupt:
        print("\nScript terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    print("Starting in 2 seconds...")
    time.sleep(2)
    main()