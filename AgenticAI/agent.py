from agents import Agent, Runner, function_tool, ModelSettings

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny with a light breeze."

weather_agent = Agent(
    name="WeatherAgent",
    instructions="You are a helpful assistant. If the user asks for weather, use the get_weather tool. Otherwise, respond normally.",
    tools=[get_weather],
    model_settings=ModelSettings(tool_choice="auto")
)

if __name__ == "__main__":
    import os

    result1 = Runner.run_sync(weather_agent, "Hello how are you?")
    print("Agent says:", result1.final_output)

    result2 = Runner.run_sync(weather_agent, "What's the weather in Paris?")
    print("Agent says:", result2.final_output)





                