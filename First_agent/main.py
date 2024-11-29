# main2.py content

import os
from crewai import LLM, Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Set the API key environment variable
os.environ['SERPER_API_KEY'] = "Your_API_Key"

# Initialize the tool for internet searching capabilities
tool = SerperDevTool()

# Initialize the language model
llm = LLM(
    model="groq/llama3-8b-8192",
    verbose=True,
    temperature=0.8,
    api_key='Your_API_Key'
)

# Creating a senior researcher agent with dynamic goals based on user inputs
news_researcher = Agent(
    role="Senior News Researcher",
    goal='Discover the latest news in the chosen topic of {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "As a senior news analyst, you excel at uncovering the latest and most accurate information"
        " on a specified topic. You always ensure the information is authentic and relevant."
    ),
    tools=[tool],  # Tool is assigned to the researcher
    llm=llm,
    allow_delegation=True
)

# Creating a writer agent with dynamic goals based on user inputs
news_writer = Agent(
    role='News Content Writer',
    goal='Create a {output_format} based on the latest findings about {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "You are skilled at extracting essential information and transforming it into"
        " various formats like blogs, articles, reports, or even plays, depending on the audience's needs."
    ),
    tools=[],  # No tools assigned to the writer
    llm=llm,
    allow_delegation=False
)

# Research task with dynamic topic input
research_task = Task(
    description=(
        "Investigate the latest and most relevant news on {topic}. Ensure the sources are credible."
        " Summarize the key findings with accuracy and include quantitative data where applicable."
    ),
    expected_output='A detailed report on the latest developments in {topic}',
    tools=[tool],
    agent=news_researcher,
    on_execute=lambda agent, task, context: agent.tools[0].search_internet(context['topic'] + " latest news")
)

# Writing task that adapts to user-chosen output format
write_task = Task(
    description=(
        "Craft a {output_format} on {topic}, highlighting the current trends and their implications."
        " The content should be clear, engaging, and tailored to the target audience."
    ),
    expected_output='A formatted {output_format} on {topic}, ready for publication.',
    tools=[],  # No tools are assigned here; the writer uses information provided by the researcher
    agent=news_writer,
    async_execution=False,
    output_file='{output_format}-on-{topic}.md'
)

# Define the crew with configurable tasks and agents
crew = Crew(
    agents=[news_researcher, news_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
)

# Start the crew process with user-specified topic and desired output format
inputs = {
    'topic': input("Enter the research topic: "),
    'output_format': input("Enter the desired output format (e.g., blog, article, report): ")
}
result = crew.kickoff(inputs=inputs)
print(result)
