import time
import boto3
from botocore.config import Config
import pyautogui
from io import BytesIO


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
    """Send screenshot and message to Bedrock."""
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
                    "display_height_px": height,
                    "display_width_px": width,
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
            return (x, y)
        return None
    except (ValueError, KeyError, TypeError):
        print(f"Error parsing coordinates from input: {input_data}")
        return None


def main():
    client = boto3.client('bedrock-runtime', config=Config(region_name='us-west-2'))
    initial_message = "Open Amazon Bedrock console by clicking in Recently visited"
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
                toolUse = message['content'][1]['toolUse']
                tool_use_id = toolUse['toolUseId']
                input_data = toolUse['input']

                # Handle actions
                action = input_data.get('action')

                if action == 'screenshot':
                    print("Taking new screenshot...")
                    new_png, new_width, new_height = get_screenshot()

                    # Add response to messages
                    messages.append({
                        'role': 'assistant',
                        'content': message['content']
                    })

                    # Add new screenshot message
                    messages.append({
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
                    })
                    continue

                elif action == 'mouse_move' or 'coordinate' in input_data:
                    coordinates = parse_coordinate(input_data)
                    if coordinates:
                        x, y = coordinates
                        if action == 'mouse_move':
                            print(f"Moving mouse to: {x}, {y}")
                            pyautogui.moveTo(x, y)
                        else:
                            print(f"Clicking at: {x}, {y}")
                            pyautogui.click(x, y)
                    else:
                        print("Invalid coordinates received")
                        continue

                # Add response to messages
                messages.append({
                    'role': 'assistant',
                    'content': message['content']
                })

                # Add empty tool result
                messages.append({
                    'role': 'user',
                    'content': [
                        {
                            'toolUse': {
                                'toolUseId': tool_use_id,
                                'output': ''
                            }
                        }
                    ]
                })

                # # Take new screenshot after action
                # new_png, new_width, new_height = get_screenshot()
                #
                # # Add new screenshot message
                # messages.append({
                #     'role': 'user',
                #     'content': [
                #         {
                #             'text': initial_message
                #         },
                #         {
                #             'image': {
                #                 'format': 'png',
                #                 'source': {
                #                     'bytes': new_png
                #                 }
                #             }
                #         }
                #     ]
                # })

            except (KeyError, IndexError) as e:
                print(f"Error processing model response: {e}")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nScript terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    print("Starting in 3 seconds...")
    time.sleep(3)
    main()