from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing import Callable
from typing import Literal
from typing_extensions import NotRequired

from langchain.agents import AgentState
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage

model = init_chat_model("anthropic:claude-3-5-sonnet-latest")


# Define the possible agent types
AgentType = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]


class SupportState(AgentState):
    """State for customer support workflow with handoffs."""

    active_agent: NotRequired[AgentType]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]



@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's warranty status and transition to issue classification."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded as: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "active_agent": "issue_classifier",
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the type of issue and transition to resolution specialist."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "active_agent": "resolution_specialist",
        }
    )


@tool
def escalate_to_human(reason: str, runtime: ToolRuntime[None, SupportState]) -> str:
    """Escalate the case to a human support specialist."""
    # In a real system, this would create a ticket, notify staff, etc.
    return f"Escalating to human support. Reason: {reason}"


@tool
def provide_solution(solution: str, runtime: ToolRuntime[None, SupportState]) -> str:
    """Provide a solution to the customer's issue."""
    return f"Solution provided: {solution}"


# Define prompts as constants for lazy interpolation
WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Warranty verification

At this stage, you need to:
1. Greet the customer warmly
2. Ask if their device is under warranty
3. Use record_warranty_status to record their response and move to the next stage

Be conversational and friendly. Don't ask multiple questions at once."""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Issue classification
CUSTOMER INFO: Warranty status is {warranty_status}

At this stage, you need to:
1. Ask the customer to describe their issue
2. Determine if it's a hardware issue (physical damage, broken parts) or software issue (app crashes, performance)
3. Use record_issue_type to record the classification and move to the next stage

If unclear, ask clarifying questions before classifying."""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Resolution
CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

At this stage, you need to:
1. For SOFTWARE issues: provide troubleshooting steps using provide_solution
2. For HARDWARE issues:
   - If IN WARRANTY: explain warranty repair process using provide_solution
   - If OUT OF WARRANTY: escalate_to_human for paid repair options

Be specific and helpful in your solutions."""



@wrap_model_call
async def warranty_collector_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent as warranty collector."""
    # Only apply if this is the active agent
    if request.state.get("active_agent", "warranty_collector") != "warranty_collector":
        return await handler(request)

    # Inject system prompt
    request = request.override(
        messages=[SystemMessage(content=WARRANTY_COLLECTOR_PROMPT)] + request.messages,
        tools=[record_warranty_status],
    )

    return await handler(request)


@wrap_model_call
async def issue_classifier_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent as issue classifier."""
    # Only apply if this is the active agent
    if request.state.get("active_agent") != "issue_classifier":
        return await handler(request)

    # Materialize system prompt with warranty status
    warranty_status = request.state.get("warranty_status")
    if warranty_status is None:
        raise ValueError("warranty_status must be set before reaching issue_classifier")

    system_prompt = ISSUE_CLASSIFIER_PROMPT.format(warranty_status=warranty_status)

    # Inject system prompt
    request = request.override(
        messages=[SystemMessage(content=system_prompt)] + request.messages,
        tools=[record_issue_type],
    )

    return await handler(request)


@wrap_model_call
async def resolution_specialist_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent as resolution specialist."""
    # Only apply if this is the active agent
    if request.state.get("active_agent") != "resolution_specialist":
        return await handler(request)

    # Materialize system prompt with lazy interpolation
    # These should always exist when we reach resolution specialist
    warranty_status = request.state.get("warranty_status")
    issue_type = request.state.get("issue_type")

    if warranty_status is None:
        raise ValueError("warranty_status must be set before reaching resolution_specialist")
    if issue_type is None:
        raise ValueError("issue_type must be set before reaching resolution_specialist")

    system_prompt = RESOLUTION_SPECIALIST_PROMPT.format(
        warranty_status=warranty_status,
        issue_type=issue_type
    )

    # Inject system prompt
    request = request.override(
        messages=[SystemMessage(content=system_prompt)] + request.messages,
        tools=[provide_solution, escalate_to_human],
    )

    return await handler(request)

# Collect all tools from all agent configurations
all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
]

# Create the agent with configuration functions
agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[
        warranty_collector_config,
        issue_classifier_config,
        resolution_specialist_config,
    ],
    checkpointer=InMemorySaver(),  # Required for state persistence across turns
)

