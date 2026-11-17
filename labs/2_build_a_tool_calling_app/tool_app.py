import os, json
from openai import OpenAI
from datetime import date as _date

API_KEY:str = os.getenv("OPENAI_API_KEY")
API_URL:str = os.getenv("OPENAI_API_BASE")

client = OpenAI(api_key=API_KEY, base_url=API_URL)

system_prompt:str = f"You are a helpful personal assistant named Zippy. Today is {_date.today().isoformat()}"

messages:list[str] = []

tools:list[str] = [
    {
        "type": "function",
        "function": {
            "name": "check_calendar",
            "description": "Checks the user's calendar for events on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date to check, provided in YYYY-MM-DD format."
                    }
                },
                "required": ["date"]
            }
        }
    }
]

def check_calendar(date) -> str:
    return """8:00 - 9:00 | Dentist Appointment
11:00 - 12:00 | Current Topics in AI Presentation
14:00 - 15:00 | Meeting with Alec
19:00 - 20:00 | Group Project Work Time"""


def execute_tool(name, args) -> str | None:
    if name == "check_calendar":
        return check_calendar(**args)
    else: return f"Tool {name} not implemented."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's on my calendar today?"}
]

response = client.chat.completions.create(
    model="gpt-5.6-luna",
    messages=messages,
    tools=tools
)

assistant_message:str = response.choices[0].message
finish_reason:str = response.choices[0].finish_reason

if finish_reason == "tool_calls":
    messages.append(assistant_message)
    for tool_call in assistant_message.tool_calls:
        tool:str = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        result:str = execute_tool(tool, args)

        messages.append(
            {
                "role": "tool", 
                "tool_call_id": tool_call.id, 
                "content": result
                }
        )

    final_response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=messages,
        tools=tools
    )

    print(final_response.choices[0].message.content)
else:
    print(assistant_message.content)
