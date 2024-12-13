import asyncio
import platform
from time import sleep

import boto3
from botocore.config import Config
import pyautogui
from io import BytesIO

def get_screenshot(region=None):
    """
    Capture a screenshot and return its bytes and dimensions.
    """
    screenshot = pyautogui.screenshot(region=region)
    buffer = BytesIO()
    screenshot.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    buffer.close()
    return image_bytes

def send_to_bedrock(client, messages):
    #print(messages)
    return client.converse(
        modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
        messages=messages,
        system=[{'text': f"""You are a helpful AI agent with access to computer control.
                       Always think step by step.

                       ENVIRONMENT:
                       1. {platform.system()}

                       IMPORTANT:
                       1. You don't need to start an application. It is running already."""
        }],
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
                    "display_height_px": 800,  # height,
                    "display_width_px": 1280,  # width,
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

def get_answer(tool_use_id, screenshot):
    sleep(0.1)
    answer = {
        'toolResult': {
            'toolUseId': tool_use_id,
            'content': []
        }
    }

    if screenshot:
        answer['toolResult']['content'].append({
            'image': {
                'format': 'png',
                'source': {
                    'bytes': screenshot
                }
            }
        })
    else:
        answer['toolResult']['content'].append({
            'text': 'OK'
        })

    return answer

def get_tool_use(content):
    for item in content:
        if 'toolUse' in item:
            yield item['toolUse']

async def main():
    done = False
    client = boto3.client('bedrock-runtime', config=Config(region_name='us-west-2'))

    with open('prompt.txt', 'r') as file:
        initial_message = file.read()

    #initial_message = "Calculate 1 + 2 by using calculator app. Combine all instructions"
    messages = []

    try:
        while not done:
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
                                    'bytes': get_screenshot()
                                }
                            }
                        }
                    ]
                }]

            response = send_to_bedrock(client, messages)
            message = response['output']['message']
            print("Model response:", message)

            try:
                content = []

                for toolUse in get_tool_use(message['content']):
                    is_actionable = True
                    tool_use_id = toolUse['toolUseId']
                    input_data = toolUse['input']

                    # Handle actions
                    action = input_data.get('action')
                    print(action)
                    screenshot = None
                    if action == 'screenshot':
                        screenshot = get_screenshot()
                    elif action == 'type':
                        pyautogui.write(input_data.get('text'))
                    elif action == 'key':
                        key = input_data.get('text')
                        if key.lower() == 'return':
                            key = 'enter'
                        pyautogui.press(input_data.get('text'))
                    elif action == 'left_click':
                        pyautogui.click()
                    elif action == 'mouse_move':
                        coordinates = parse_coordinate(input_data)
                        if coordinates:
                            x, y = coordinates
                            pyautogui.moveTo(x, y)
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
                    elif action is None:
                        pass
                    else:
                        print("Unsupported action received")
                        break

                    content.append(get_answer(tool_use_id, screenshot))

                # Add assistant message
                messages.append({
                    'role': 'assistant',
                    'content': message['content']
                })
                # Add answer message
                messages.append({
                    'role': 'user',
                    'content':content
                })
                done = not is_actionable

            except (KeyError, IndexError) as e:
                print(f"Error processing model response: {e}")
                break

    except KeyboardInterrupt:
        print("\nScript terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    print("Starting...")
    asyncio.run(main())