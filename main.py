import asyncio
import platform
from time import sleep
import boto3
from botocore.config import Config
import pyautogui
from io import BytesIO
from tools.computer import ComputerTool, ToolResult

tool = ComputerTool()

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

def _make_api_tool_result(result: ToolResult, tool_use_id: str):
    tool_result_content = []
    if result.output:
        tool_result_content.append({
            "text": result.output,
        })
    if result.image:
        tool_result_content.append({
            'image': {
                'format': 'png',
                'source': {
                    'bytes': result.image
                }
            },
        })
    return {
        'toolResult': {
            'toolUseId': tool_use_id,
            'content': tool_result_content
        }
    }

async def main():
    done = False
    client = boto3.client('bedrock-runtime', config=Config(region_name='us-west-2'))

    with open('prompt.txt', 'r') as file:
        initial_message = file.read()

    #initial_message = "Calculate 1 + 2 by using calculator app. Combine all instructions"
    messages = []

    while True:
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
        text = message['content'][0].get('text')
        if text:
            print(text)
            if 'wait' in text:
                sleep(5)

        # Add assistant message
        messages.append({
            'role': 'assistant',
            'content': message['content']
        })

        tool_result_content = []
        for content_block in message['content']:
            print(content_block)
            tool_use = content_block.get('toolUse')
            if tool_use:
                result = tool.__call__(**tool_use['input'])
                tool_result_content.append(
                    _make_api_tool_result(result, tool_use['toolUseId'])
                )

        if not tool_result_content:
            break

        messages.append({
            'role': 'user',
            'content': tool_result_content
        })


if __name__ == "__main__":
    print("Starting...")
    asyncio.run(main())