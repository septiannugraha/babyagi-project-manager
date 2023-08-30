import os
from collections import deque
from typing import Dict, List, Optional
from dotenv import load_dotenv

import pinecone
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores import Pinecone
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field

load_dotenv()

# refer to this link
# https://github.com/hinthornw/lckg/blob/57d63b5f6697cf8908c9bc51df075232967d0678/docs/modules/chains/examples/baby_agi.ipynb

# Define the parameters for this run.
index_name = "langchain-baby-agi"
OBJECTIVE = "Building support-oriented Bard in Baldur's Gate 3" # Ultimate goal
YOUR_FIRST_TASK = "create my character, choosing wood elf as the race" # Where to start

llm = OpenAI() # The underlying LLM

# Get Pinecone environment
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"], # find at app.pinecone.io
    environment=os.environ["PINECONE_ENVIRONMENT"] # next to api key in console
)

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

# Create Pinecone index if it doesn't exist
dimension = 1536 # dimension of the embedding space
metric = "cosine" # distance function
pod_type = "p1" # performance-optimized
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=dimension, metric=metric, pod_type=pod_type)

# Fetch the index
text_key = "page_content"
if index_name in pinecone.list_indexes():
    index = pinecone.Index(index_name)
else:
    raise ValueError(f"Index '{index_name}' not found in your Pinecone project.")

vectorstore = Pinecone(index, embeddings_model.embed_query, text_key)

class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            partial_variables={"objective": objective},
            input_variables=["result", "task_description", "incomplete_tasks"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
    def get_next_task(self, result: Dict, task_description: str, task_list: List[str]) -> List[Dict]:
        """Get the next task."""
        incomplete_tasks = ", ".join(task_list)
        response = self.run(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks)
        new_tasks = response.split('\n')
        return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]

class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            partial_variables={"objective": objective},
            input_variables=["task_names", "next_task_id"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def prioritize_tasks(self, this_task_id: int, task_list: List[Dict]) -> List[Dict]:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in task_list]
        next_task_id = int(this_task_id) + 1
        response = self.run(task_names=task_names, next_task_id=next_task_id)
        new_tasks = response.split('\n')
        prioritized_task_list = []
        for task_string in new_tasks:
            if not task_string.strip():
                continue
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
        return prioritized_task_list

class ExecutionChain(LLMChain):
    """Chain to execute tasks."""
    
    vectorstore: VectorStore = Field(init=False)

    @classmethod
    def from_llm(cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "You are an AI who performs one task based on the following objective: {objective}."
            " Take into account these previously completed tasks: {context}."
            " Your task: {task}."
            " Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose, vectorstore=vectorstore)
    
    def _get_top_tasks(self, query: str, k: int) -> List[str]:
        """Get the top k tasks based on the query."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return []
        sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
        return [str(item.metadata['task']) for item in sorted_results]
    
    def execute_task(self, objective: str, task: str, k: int = 5) -> str:
        """Execute a task."""
        context = self._get_top_tasks(query=objective, k=k)
        return self.run(objective=objective, context=context, task=task)

class BabyAGI(BaseModel):
    """Controller model for the BabyAGI agent."""

    objective: str = Field(alias="objective")
    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: ExecutionChain = Field(...)
    task_id_counter: int = Field(1)

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    def run(self, max_iterations: Optional[int] = None):
        """Run the agent."""
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = self.execution_chain.execute_task(
                    self.objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.execution_chain.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = self.task_creation_chain.get_next_task(
                    result, task["task_name"], [t["task_name"] for t in self.task_list]
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    self.task_prioritization_chain.prioritize_tasks(
                        this_task_id, list(self.task_list)
                    )
                )
            num_iters += 1
            if max_iterations is not None and num_iters == max_iterations:
                print("\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m")
                break

    @classmethod
    def from_llm_and_objectives(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        objective: str,
        first_task: str,
        verbose: bool = False,
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, objective, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, objective, verbose=verbose
        )
        execution_chain = ExecutionChain.from_llm(llm, vectorstore, verbose=verbose)
        controller =  cls(
            objective=objective,
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
        )
        controller.add_task({"task_id": 1, "task_name": first_task})
        return controller

# Run the agent
verbose=False

baby_agi = BabyAGI.from_llm_and_objectives(
    llm=llm,
    vectorstore=vectorstore,
    objective=OBJECTIVE,
    first_task=YOUR_FIRST_TASK,
    verbose=verbose
)

max_iterations: Optional[int] = 3 # If None, agent will run until interrupted
baby_agi.run(max_iterations=max_iterations)