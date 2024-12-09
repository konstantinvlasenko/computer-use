import asyncio
import time
import boto3
from botocore.config import Config
import pyautogui
from io import BytesIO
from run import run

def get_screenshot(region=None):
    """
    Capture a screenshot and return its bytes and dimensions.
    """
    screenshot = pyautogui.screenshot(region=region)
    width, height = screenshot.size
    buffer = BytesIO()
    screenshot.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    buffer.close()
    return image_bytes, width, height

def send_to_bedrock(client, png, width, height, messages):
    #print(messages)
    return client.converse(
        modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
        messages=messages,
        toolConfig={
            'tools': [
                {
                    'toolSpec': {
                        'name': 'computer_tool',  # Changed name
                        'inputSchema': {
                            'json': {
                                "type": "object"
                            }
                        }
                    }
                }
            ]
        },
        additionalModelRequestFields={
            "tools": [
                {
                    "type": "computer_20241022",
                    "name": "computer",  # Matched name
                    "display_height_px": 1080, #height,
                    "display_width_px": 3840, #width,
                    "display_number": 0
                }
            ],
            "anthropic_beta": ["computer-use-2024-10-22"]
        }
    )

def parse_coordinate(input_data):
    """
    Parse coordinate data from the input, handling different possible formats.
    Returns tuple of (x, y) coordinates or None if parsing fails.
    """
    try:
        if isinstance(input_data.get('coordinate'), list):
            return tuple(input_data['coordinate'])
        elif isinstance(input_data.get('coordinate'), str):
            # Remove brackets and split
            coord_str = input_data['coordinate'].strip('[]')
            x, y = map(int, coord_str.split(','))
            return x, y
        return None
    except (ValueError, KeyError, TypeError):
        print(f"Error parsing coordinates from input: {input_data}")
        return None

def get_answer(tool_use_id):
    new_png, new_width, new_height = get_screenshot()

    return {
        'role': 'user',
        'content': [
            {
                'toolResult': {
                    'toolUseId': tool_use_id,
                    'content': [{
                        'image': {
                            'format': 'png',
                            'source': {
                                'bytes': new_png
                            }
                        }
                    }]
                }
            }
        ]
    }

def get_tool_use(content):
    for item in content:
        if 'toolUse' in item:
            return item['toolUse']

async def main():
    client = boto3.client('bedrock-runtime', config=Config(region_name='us-west-2'))
    initial_message = "Click 1 on calculator"
    messages = []

    try:
        while True:
            png, width, height = get_screenshot()

            # If messages is empty, add initial user message
            if not messages:
                messages = [{
                    'role': 'user',
                    'content': [
                        {
                            'text': initial_message
                        },
                        {
                            'image': {
                                'format': 'png',
                                'source': {
                                    'bytes': png
                                }
                            }
                        }
                    ]
                }]

            response = send_to_bedrock(client, png, width, height, messages)
            message = response['output']['message']
            print("Model response:", message)

            try:
                toolUse = get_tool_use(message['content'])
                tool_use_id = toolUse['toolUseId']
                input_data = toolUse['input']

                # Handle actions
                action = input_data.get('action')
                print(action)

                if action == 'screenshot':
                    pass
                elif action == 'left_click':
                    pyautogui.click()
                elif action == 'mouse_move':
                    _display_prefix = ""
                    xdotool = f"{_display_prefix}xdotool"

                    coordinates = parse_coordinate(input_data)
                    if coordinates:
                        x, y = coordinates
                        # pyautogui.moveTo(x, y)
                        await run(f"{xdotool} mousemove --sync {x} {y}")
                    else:
                        print("Invalid coordinates received")
                        continue
                elif 'coordinate' in input_data:
                    coordinates = parse_coordinate(input_data)
                    if coordinates:
                        x, y = coordinates
                        print(f"Clicking at: {x}, {y}")
                        pyautogui.click(x, y)
                    else:
                        print("Invalid coordinates received")
                        continue
                else:
                    print("Unsupported action received")
                    continue

                # Add assistant message
                messages.append({
                    'role': 'assistant',
                    'content': message['content']
                })

                # Add answer message
                messages.append(get_answer(tool_use_id))

            except (KeyError, IndexError) as e:
                print(f"Error processing model response: {e}")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nScript terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    print("Starting in 2 seconds...")
    time.sleep(2)
    asyncio.run(main())