from agents import Agent, Runner, ModelSettings
from weather_agent import weather_agent 

manager_agent = Agent(
    name="ManagerAgent",
    instructions="""
    You are a manager agent. If the user input is weather-related, call the WeatherAgent and return its output.
    Otherwise, respond normally.
    """,
    tools=[],  
    model_settings=ModelSettings(tool_choice="auto")
)

def call_manager(user_input: str):
    if "weather" in user_input.lower():
        result = Runner.run_sync(weather_agent, user_input)
        return f"[WeatherAgent]: {result.final_output}"
    else:
        result = Runner.run_sync(manager_agent, user_input)
        return f"[ManagerAgent]: {result.final_output}"

if __name__ == "__main__":
    print(call_manager("Hello, how are you?"))
    print(call_manager("What's the weather in Paris?"))
