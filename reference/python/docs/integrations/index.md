---
title: Integrations overview
hide:
  - toc
---

Welcome! These pages include reference documentation for all `langchain-*` Python integration packages.

To learn more about integrations in LangChain, visit the [Integrations overview](https://docs.langchain.com/oss/python/integrations/providers/overview).

!!! tip "Model Context Protocol (MCP) support"
    To use MCP tools within LangChain and LangGraph applications, refer to [`langchain-mcp-adapters`](../langchain_mcp_adapters/index.md).

## Popular providers

<div class="grid cards" markdown>

- :fontawesome-brands-openai:{ .lg .middle } __`langchain-openai`__

    ---

    Interact with OpenAI (completions, responses) and OpenAI compatible APIs.

    [:octicons-arrow-right-24: Reference](./langchain_openai/index.md)

- :simple-claude:{ .lg .middle } __`langchain-anthropic`__

    ---

    Interact with Claude (Anthropic) APIs.

    [:octicons-arrow-right-24: Reference](./langchain_anthropic/index.md)

- :simple-googlegemini:{ .lg .middle } __`langchain-google-genai`__

    ---

    Access Google Gemini models via the Google Gen AI SDK.

    [:octicons-arrow-right-24: Reference](./langchain_google_genai.md)

- :simple-googlecloud:{ .lg .middle } __`langchain-google-vertexai`__

    ---

    Use Google's Vertex AI model platform.

    [:octicons-arrow-right-24: Reference](./langchain_google_vertexai.md)

- :material-aws:{ .lg .middle } __`langchain-aws`__

    ---

    Use integrations related to the AWS platform such as Bedrock, S3, and more.

    [:octicons-arrow-right-24: Reference](./langchain_aws.md)

- :simple-huggingface:{ .lg .middle } __`langchain-huggingface`__

    ---

    Access HuggingFace-hosted models in LangChain.

    [:octicons-arrow-right-24: Reference](./langchain_huggingface.md)

- :material-message:{ .lg .middle } __`langchain-groq`__

    ---

    Interface to Groq Cloud.

    [:octicons-arrow-right-24: Reference](./langchain_groq.md)

- :simple-ollama:{ .lg .middle } __`langchain-ollama`__

    ---

    Use locally hosted models via Ollama.

    [:octicons-arrow-right-24: Reference](./langchain_ollama.md)

</div>

Other providers are listed in the section navigation (left sidebar).

!!! question ""I don't see the integration I'm looking for?""
    LangChain has hundreds of integrations, but not all are documented on this site. If you don't see the integration you're looking for, refer to their [provider page in the LangChain docs](https://docs.langchain.com/oss/python/integrations/providers/all_providers) or check their respective GitHub repository for more information.

!!! note "Create new integrations"
    For information on contributing new integrations, see [the guide](https://docs.langchain.com/oss/python/contributing/integrations-langchain).
