from llm import get_open_ai_model
from  langchain_core.prompts import ChatPromptTemplate

llm=get_open_ai_model()

prompt=ChatPromptTemplate.from_messages([
   ( "system","you are a helpful assistant"),
    ("human","Explain the concept of {topic}"),
])

chain=prompt | llm
response=chain.invoke({"topic":"Newton's Laws of Motion"})
print(response.content)
