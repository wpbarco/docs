"""Link mapping for cross-reference resolution across different scopes.

This module provides link mappings for different language/framework scopes
to resolve @[link_name] references to actual URLs.

Link maps are composed of:
1. MANUAL_LINK_MAPS: Curated, high-quality mappings that override auto-generated ones
2. AUTO_GENERATED_LINK_MAPS: Generated from objects.inv files via scripts/generate_link_maps.py

Manual maps take precedence over auto-generated ones when merging.
"""

from collections.abc import Mapping
from typing import TypedDict


class LinkMap(TypedDict):
    """Typed mapping describing each link map entry."""

    host: str
    scope: str
    links: Mapping[str, str]


# Manual link maps - these override auto-generated links
# Use these for:
# - Custom aliases or shortcuts
# - Links that need special handling
# - External documentation not in objects.inv
MANUAL_LINK_MAPS: list[LinkMap] = [
    {
        # Python LangGraph reference
        "host": "https://langchain-ai.github.io/langgraph/",
        "scope": "python",
        "links": {
            "StateGraph": "reference/graphs/#langgraph.graph.StateGraph",
            "add_conditional_edges": "reference/graphs/#langgraph.graph.state.StateGraph.add_conditional_edges",
            "add_edge": "reference/graphs/#langgraph.graph.state.StateGraph.add_edge",
            "add_node": "reference/graphs/#langgraph.graph.state.StateGraph.add_node",
            "add_messages": "reference/graphs/#langgraph.graph.message.add_messages",
            "astream_events": "https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html#langchain_core.language_models.chat_models.BaseChatModel.astream_events",
            "ToolNode": "reference/agents/#langgraph.prebuilt.tool_node.ToolNode",
            "CompiledStateGraph.astream": "reference/graphs/#langgraph.graph.state.CompiledStateGraph.astream",
            "Pregel.astream": "reference/pregel/#langgraph.pregel.Pregel.astream",
            "AsyncPostgresSaver": "reference/checkpoints/#langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
            "AsyncSqliteSaver": "reference/checkpoints/#langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver",
            "BaseCheckpointSaver": "reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver",
            "BaseLoader": "https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.base.BaseLoader.html",
            "BaseStore": "reference/store/#langgraph.store.base.BaseStore",
            "BaseStore.put": "reference/store/#langgraph.store.base.BaseStore.put",
            "BinaryOperatorAggregate": "reference/pregel/#langgraph.pregel.Pregel--advanced-channels-context-and-binaryoperatoraggregate",
            "CipherProtocol": "reference/checkpoints/#langgraph.checkpoint.serde.base.CipherProtocol",
            "client.runs.stream": "cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.RunsClient.stream",
            "client.runs.wait": "cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.RunsClient.wait",
            "client.threads.get_history": "cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.ThreadsClient.get_history",
            "client.threads.update_state": "cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.ThreadsClient.update_state",
            "Command": "reference/types/#langgraph.types.Command",
            "CompiledStateGraph": "reference/graphs/#langgraph.graph.state.CompiledStateGraph",
            "create_react_agent": "reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent",
            "get_runtime": "reference/runtime/#langgraph.runtime.get_runtime",
            "create_supervisor": "reference/supervisor/#langgraph_supervisor.supervisor.create_supervisor",
            "EncryptedSerializer": "reference/checkpoints/#langgraph.checkpoint.serde.encrypted.EncryptedSerializer",
            "entrypoint.final": "reference/func/#langgraph.func.entrypoint.final",
            "entrypoint": "reference/func/#langgraph.func.entrypoint",
            "from_pycryptodome_aes": "reference/checkpoints/#langgraph.checkpoint.serde.encrypted.EncryptedSerializer.from_pycryptodome_aes",
            "get_state_history": "reference/graphs/#langgraph.graph.state.CompiledStateGraph.get_state_history",
            "get_stream_writer": "reference/config/#langgraph.config.get_stream_writer",
            "HumanInterrupt": "reference/prebuilt/#langgraph.prebuilt.interrupt.HumanInterrupt",
            "InjectedState": "reference/agents/#langgraph.prebuilt.tool_node.InjectedState",
            "InMemorySaver": "reference/checkpoints/#langgraph.checkpoint.memory.InMemorySaver",
            "init_chat_model": "https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html",
            "interrupt": "reference/types/#langgraph.types.interrupt",
            "CompiledStateGraph.invoke": "reference/graphs/#langgraph.graph.state.CompiledStateGraph.invoke",
            "JsonPlusSerializer": "reference/checkpoints/#langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer",
            "langgraph.json": "cloud/reference/cli/#configuration-file",
            "LastValue": "reference/channels/#langgraph.channels.LastValue",
            "PostgresSaver": "reference/checkpoints/#langgraph.checkpoint.postgres.PostgresSaver",
            "Pregel": "reference/pregel/",
            "Pregel.stream": "reference/pregel/#langgraph.pregel.Pregel.stream",
            "pre_model_hook": "reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent",
            "protocol": "reference/checkpoints/#langgraph.checkpoint.serde.base.SerializerProtocol",
            "Reference": "https://python.langchain.com/api_reference/",
            "Runtime": "reference/runtime/#langgraph.runtime.Runtime",
            "Send": "reference/types/#langgraph.types.Send",
            "SerializerProtocol": "reference/checkpoints/#langgraph.checkpoint.serde.base.SerializerProtocol",
            "SqliteSaver": "reference/checkpoints/#langgraph.checkpoint.sqlite.SqliteSaver",
            "START": "reference/constants/#langgraph.constants.START",
            "CompiledStateGraph.stream": "reference/graphs/#langgraph.graph.state.CompiledStateGraph.stream",
            "task": "reference/func/#langgraph.func.task",
            "Topic": "reference/channels/#langgraph.channels.Topic",
            "update_state": "reference/graphs/#langgraph.graph.state.CompiledStateGraph.update_state",
        },
    },
    {
        # JS LangGraph reference
        "host": "https://langchain-ai.github.io/langgraphjs/",
        "scope": "js",
        "links": {
            "Auth": "reference/classes/sdk_auth.Auth.html",
            "StateGraph": "reference/classes/langgraph.StateGraph.html",
            "add_conditional_edges": "/reference/classes/langgraph.StateGraph.html#addConditionalEdges",
            "add_edge": "reference/classes/langgraph.StateGraph.html#addEdge",
            "add_node": "reference/classes/langgraph.StateGraph.html#addNode",
            "add_messages": "reference/modules/langgraph.html#addMessages",
            "astream_events": "https://v03.api.js.langchain.com/types/_langchain_core.tracers_log_stream.StreamEvent.html",
            "BaseCheckpointSaver": "reference/classes/checkpoint.BaseCheckpointSaver.html",
            "BaseLoader": "https://v03.api.js.langchain.com/classes/_langchain_core.document_loaders_base.BaseDocumentLoader.html",
            "BaseStore": "reference/classes/checkpoint.BaseStore.html",
            "BaseStore.put": "reference/classes/checkpoint.BaseStore.html#put",
            "BinaryOperatorAggregate": "reference/classes/langgraph.BinaryOperatorAggregate.html",
            "client.runs.stream": "reference/classes/sdk_client.RunsClient.html#stream",
            "client.runs.wait": "reference/classes/sdk_client.RunsClient.html#wait",
            "client.threads.get_history": "reference/classes/sdk_client.ThreadsClient.html#getHistory",
            "client.threads.update_state": "reference/classes/sdk_client.ThreadsClient.html#updateState",
            "Command": "reference/classes/langgraph.Command.html",
            "CompiledStateGraph": "reference/classes/langgraph.CompiledStateGraph.html",
            "create_react_agent": "reference/functions/langgraph_prebuilt.createReactAgent.html",
            "create_supervisor": "reference/functions/langgraph_supervisor.createSupervisor.html",
            "entrypoint.final": "reference/functions/langgraph.entrypoint.html#final",
            "entrypoint": "reference/functions/langgraph.entrypoint.html",
            "getContextVariable": "https://v03.api.js.langchain.com/functions/_langchain_core.context.getContextVariable.html",
            "get_state_history": "reference/classes/langgraph.CompiledStateGraph.html#getStateHistory",
            "HumanInterrupt": "reference/interfaces/langgraph_prebuilt.HumanInterrupt.html",
            "init_chat_model": "https://v03.api.js.langchain.com/functions/langchain.chat_models_universal.initChatModel.html",
            "interrupt": "reference/functions/langgraph.interrupt-2.html",
            "CompiledStateGraph.invoke": "reference/classes/langgraph.CompiledStateGraph.html#invoke",
            "langgraph.json": "cloud/reference/cli/#configuration-file",
            "MemorySaver": "reference/classes/checkpoint.MemorySaver.html",
            "messagesStateReducer": "reference/functions/langgraph.messagesStateReducer.html",
            "PostgresSaver": "reference/classes/checkpoint_postgres.PostgresSaver.html",
            "Pregel": "reference/classes/langgraph.Pregel.html",
            "Pregel.stream": "reference/classes/langgraph.Pregel.html#stream",
            "pre_model_hook": "reference/functions/langgraph_prebuilt.createReactAgent.html",
            "protocol": "reference/interfaces/checkpoint.SerializerProtocol.html",
            "Send": "reference/classes/langgraph.Send.html",
            "SerializerProtocol": "reference/interfaces/checkpoint.SerializerProtocol.html",
            "SqliteSaver": "reference/classes/checkpoint_sqlite.SqliteSaver.html",
            "START": "reference/variables/langgraph.START.html",
            "CompiledStateGraph.stream": "reference/classes/langgraph.CompiledStateGraph.html#stream",
            "task": "reference/functions/langgraph.task.html",
            "update_state": "reference/classes/langgraph.CompiledStateGraph.html#updateState",
        },
    },
    {
        "host": "https://v03.api.js.langchain.com/",
        "scope": "js",
        "links": {
            "AIMessage": "classes/_langchain_core.messages_ai_message.AIMessage.html",
            "AIMessageChunk": "classes/_langchain_core.messages_ai_message.AIMessageChunk.html",
            "BaseChatModel.invoke": "classes/_langchain_core.language_models_chat_models.BaseChatModel.html#invoke",
            "BaseChatModel.stream": "classes/_langchain_core.language_models_chat_models.BaseChatModel.html#stream",
            "BaseChatModel.streamEvents": "classes/_langchain_core.language_models_chat_models.BaseChatModel.html#streamEvents",
            "BaseChatModel.batch": "classes/_langchain_core.language_models_chat_models.BaseChatModel.html#batch",
            "BaseChatModel.bindTools": "classes/langchain.chat_models_universal.ConfigurableModel.html#bindTools",
            "Document": "classes/_langchain_core.documents.Document.html",
            "initChatModel": "functions/langchain.chat_models_universal.initChatModel.html",
            "RunnableConfig": "interfaces/_langchain_core.runnables.RunnableConfig.html",
            "Reference": "index.html",
            "Embeddings": "classes/_langchain_core.embeddings.Embeddings.html",
        },
    },
    {
        "host": "https://reference.langchain.com/python/",
        "scope": "python",
        "links": {
            # Module pages
            "langchain": "langchain/langchain",
            "langchain.agents": "langchain/agents",
            "langchain.messages": "langchain/messages",
            "langchain.tools": "langchain/tools",
            "langchain.chat_models": "langchain/models",
            "langchain.embeddings": "langchain/embeddings",
            "langchain_core": "langchain_core/",
            # Agents
            "create_agent": "langchain/agents/#langchain.agents.create_agent",
            "create_agent(tools)": "langchain/agents/#langchain.agents.create_agent(tools)",
            "system_prompt": "langchain/agents/#langchain.agents.create_agent(system_prompt)",
            "AgentState": "langchain/agents/#langchain.agents.AgentState",
            # Middleware
            "AgentMiddleware": "langchain/middleware/#langchain.agents.middleware.AgentMiddleware",
            "state_schema": "langchain/middleware/#langchain.agents.middleware.AgentMiddleware.state_schema",
            "PIIMiddleware": "langchain/middleware/#langchain.agents.middleware.PIIMiddleware",
            "SummarizationMiddleware": "langchain/middleware/#langchain.agents.middleware.SummarizationMiddleware",
            "HumanInTheLoopMiddleware": "langchain/middleware/#langchain.agents.middleware.HumanInTheLoopMiddleware",
            # Messages
            "AIMessage": "langchain/messages/#langchain.messages.AIMessage",
            "AIMessageChunk": "langchain/messages/#langchain.messages.AIMessageChunk",
            "ToolMessage": "langchain/messages/#langchain.messages.ToolMessage",
            "SystemMessage": "langchain/messages/#langchain.messages.SystemMessage",
            "trim_messages": "langchain/messages/#langchain.messages.trim_messages",
            # Content blocks
            "BaseMessage": "langchain_core/language_models/#langchain_core.messages.BaseMessage",
            "BaseMessage(content)": "langchain_core/language_models/#langchain_core.messages.BaseMessage.content",
            "BaseMessage(content_blocks)": "langchain_core/language_models/#langchain_core.messages.BaseMessage.content_blocks",
            "ContentBlock": "langchain/messages/#langchain.messages.ContentBlock",
            "TextContentBlock": "langchain/messages/#langchain.messages.TextContentBlock",
            "ReasoningContentBlock": "langchain/messages/#langchain.messages.ReasoningContentBlock",
            "NonStandardContentBlock": "langchain/messages/#langchain.messages.NonStandardContentBlock",
            "ImageContentBlock": "langchain/messages/#langchain.messages.ImageContentBlock",
            "VideoContentBlock": "langchain/messages/#langchain.messages.VideoContentBlock",
            "AudioContentBlock": "langchain/messages/#langchain.messages.AudioContentBlock",
            "PlainTextContentBlock": "langchain/messages/#langchain.messages.PlainTextContentBlock",
            "FileContentBlock": "langchain/messages/#langchain.messages.FileContentBlock",
            "ToolCall": "langchain/messages/#langchain.messages.ToolCall",
            "ToolCallChunk": "langchain/messages/#langchain.messages.ToolCallChunk",
            "ServerToolCall": "langchain/messages/#langchain.messages.ServerToolCall",
            "ServerToolCallChunk": "langchain/messages/#langchain.messages.ServerToolCallChunk",
            "ServerToolResult": "langchain/messages/#langchain.messages.ServerToolResult",
            # Integrations
            "langchain-openai": "integrations/langchain_openai",
            "ChatOpenAI": "integrations/langchain_openai/#langchain_openai.ChatOpenAI",
            "AzureChatOpenAI": "integrations/langchain_openai/#langchain_openai.AzureChatOpenAI",
            # Models
            "init_chat_model": "langchain/models/#langchain.chat_models.init_chat_model",
            "init_chat_model(model_provider)": "langchain/models/#langchain.chat_models.init_chat_model(model_provider)",
            "BaseChatModel": "langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel",
            "BaseChatModel.invoke": "langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.invoke",
            "BaseChatModel.stream": "langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.stream",
            "BaseChatModel.astream_events": "langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.astream_events",
            "BaseChatModel.batch": "langchain_core.language_models.chat_models.BaseChatModel.batch",
            "BaseChatModel.batch_as_completed": "langchain_core.language_models.chat_models.BaseChatModel.batch_as_completed",
            "BaseChatModel.bind_tools": "langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.bind_tools",
            # Tools
            "@tool": "langchain/tools/#langchain.tools.tool",
            "BaseTool": "langchain/tools/#langchain.tools.BaseTool",
            # Embeddings
            "init_embeddings": "langchain_core/embeddings/#langchain_core.embeddings.embeddings.Embeddings",
            "Embeddings": "langchain_core/embeddings/#langchain_core.embeddings.embeddings.Embeddings",
            # Documents
            "Document": "langchain_core/documents/#langchain_core.documents.base.Document",
            # Runnables
            "RunnableConfig": "langchain_core/runnables/#langchain_core.runnables.RunnableConfig",
        },
    },
    {
        "host": "https://reference.langchain.com/javascript/",
        "scope": "js",
        "links": {
            "Runtime": "interfaces/_langchain_langgraph.index.Runtime.html",
            "tool": "functions/_langchain_core.tools.tool.html",
            "ToolNode": "classes/langchain.index.ToolNode.html",
            "UsageMetadata": "types/_langchain_core.messages.UsageMetadata.html",
        },
    },
]


def _merge_link_maps(
    manual_maps: list[LinkMap],
    auto_maps: list[LinkMap],
) -> list[LinkMap]:
    """Merge manual and auto-generated link maps.

    Manual maps take precedence over auto-generated maps for the same keys.

    Args:
        manual_maps: Manually curated link maps.
        auto_maps: Auto-generated link maps from objects.inv.

    Returns:
        Merged list of link maps with manual entries taking precedence.
    """
    # Group by (host, scope) for merging
    merged: dict[tuple[str, str], dict[str, str]] = {}

    # First add auto-generated links
    for link_map in auto_maps:
        key = (link_map["host"], link_map["scope"])
        if key not in merged:
            merged[key] = {}
        merged[key].update(link_map["links"])

    # Then overlay manual links (these take precedence)
    for link_map in manual_maps:
        key = (link_map["host"], link_map["scope"])
        if key not in merged:
            merged[key] = {}
        merged[key].update(link_map["links"])

    # Convert back to list of LinkMap
    result: list[LinkMap] = []
    for (host, scope), links in merged.items():
        result.append({"host": host, "scope": scope, "links": links})

    return result


# Import auto-generated link maps
try:
    from pipeline.preprocessors.link_map_generated import (
        AUTO_GENERATED_LINK_MAPS,
    )
except ImportError:
    # If not generated yet, use empty list
    AUTO_GENERATED_LINK_MAPS = []

# Merge manual and auto-generated link maps
LINK_MAPS: list[LinkMap] = _merge_link_maps(MANUAL_LINK_MAPS, AUTO_GENERATED_LINK_MAPS)


def _enumerate_links(scope: str) -> dict[str, str]:
    """Enumerate all links for a given scope.

    Args:
        scope: The scope to enumerate links for (e.g., 'python', 'js').

    Returns:
        Dictionary mapping link names to full URLs.
    """
    result = {}
    for link_map in LINK_MAPS:
        if link_map["scope"] == scope:
            links = link_map["links"]
            for key, value in links.items():
                if not value.startswith("http"):
                    result[key] = f"{link_map['host']}{value}"
                else:
                    result[key] = value
    return result


# Global scope is assembled from the Python and JS mappings
# Combined mapping by scope
SCOPE_LINK_MAPS = {
    "python": _enumerate_links("python"),
    "js": _enumerate_links("js"),
}
