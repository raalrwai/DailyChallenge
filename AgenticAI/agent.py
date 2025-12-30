from agents import Agent, Runner, function_tool, ModelSettings
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY_HERE")

@function_tool
def get_weather(city: str) -> str:
    weather_info = f"The weather in {city} is sunny with a light breeze."
    prompt = f"Given this weather info: '{weather_info}', respond to the user in a friendly way."
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

weather_agent = Agent(
    name="WeatherGPTAgent",
    instructions=(
        "You are a helpful assistant. "
        "If the user asks for weather, use the get_weather tool. "
        "Otherwise, respond normally using GPT."
    ),
    tools=[get_weather],
    model_settings=ModelSettings(tool_choice="auto")
)

if __name__ == "__main__":
    result = Runner.run_sync(weather_agent, "What's the weather in Paris?")
    print("Agent says:", result.final_output)
    
    result2 = Runner.run_sync(weather_agent, "Explain what an API is.")
    print("Agent says:", result2.final_output)



                