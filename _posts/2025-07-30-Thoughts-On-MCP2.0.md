---
layout: post
title: "MCP without analogies"
date: 2025-07-30
mathjax: true
status: [Work experience, Misc, Review]
categories: [Work Experiences, Misc, Review]
---

**Note:** Edited in August 8 2026.

#### **Preview**

A lot of people talk about how MCP is a USB-C connector to Services and APIs on the internet. But I think this doesn’t really give people a proper appreciation of the picture, because it's not that difficult to do integrations with REST-APIs or for LLM Agents to be exposed to functions directly without a client-server architecture.

Isn’t it straightforward to `import googledrive`  and allow LLMs to go execute that function? Or to `curl` the API? Why do we need a new protocol?

<br>

#### **First some Preliminaries**

Here's a workflow with MCP:


<div id='image-container'>
  <a href="{{ site.baseurl }}/assets/mcp.png" target="_blank" id="zoomable-link">
    <img src="{{ site.baseurl }}/assets/mcp.png" alt="Zoomable Image">
  </a>
</div>

<br>







1) Human asks for the weather

2) Agent has a configuration file that tells it about available MCP servers

{% highlight python %}
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["/path/to/weather.py"],
      "env": {}
    },
    "filesystem": {
      "command": "node",
      "args": ["/path/to/filesystem-server.js"]
    }
  }
}
{% endhighlight %}



3) Agent spawns subprocess for local server OR establishes a connection to the remote Server


4) Agent *generates* and sends JSON-RPC to the server

{% highlight python %}
{ 
  "jsonrpc": "2.0", 
  "id": 1, 
  "method": "tools/call", 
  "params": 
      { 
        "name": "get_alerts", "arguments": {"state": "CA"} 
      } 
}
{% endhighlight %}

5) MCP Server processes and forwards query to the API or function call

6) External Tool or API returns a response and MCP Server formats this as JSON-RPC

{% highlight python %}
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Active alerts for CA:\nHeat Warning: Excessive heat..."
      }
    ]
  }
}
{% endhighlight %}

7) LLM-Agent Application formats the response 

<br>



#### **MCP from a functional point of view**

We want to give LLM-Agents the ability to execute functions to overcome their limitations. 

But this is ridiculously dangerous. If you give code execution control to an LLM or allow it to access an SDK (e.g., allow it to do `code.eval()`), there must be endless checks on what actually gets executed. The entire Google Drive API which has been exposed to human programmers may be more than what we’d like an LLM to be able to execute.


Perhaps we only give Agents read access (`GET`, no `POST`), but sometimes we want Agents to have some write access if we know those can happen reliably. Meaning, we need to control the kinds of write access they have gradually. 

Hence a LLM-safe-access wrapper needs to be constructed to expose a subset of the full APIs and SDKs that we are prepared for Agents to hit.

The wrapper itself is thin. For instance, the following weather-mcp tool exposes `get_alert` and nothing else, no delete_forecast, no admin endpoints.

{% highlight python %}
from fastmcp import FastMCP
import httpx

mcp = FastMCP("weather")

@mcp.tool
def get_alerts(state: str) -> str:
    """Active weather alerts for a US state (read-only)."""
    r = httpx.get(f"https://api.weather.gov/alerts/active?area={state}",
                  headers={"User-Agent": "mcp-demo"})
    return r.text

if __name__ == "__main__":
    mcp.run()  # stdio by default

{% endhighlight %}

<br> 

#### **Why not just have some piece of code that exposes only the "llm-safe" functions ?**

That's akin to a local MCP server, but using stdio as a transport protocol.

<br>


#### **But why do we need to set up this stdio and separate client and servers. Can't we directly expose these "llm-safe" functions to the Agent?**

Technically we can, and that's what LangGraph, OpenAI Agent SDK among others, did for exposing functions as Tool Calls. The main advantage of setting up server-client architecture and having a transport layer protocol is for applications to be language agnostic so that different programming languages can be used at different parts of the stack. For e.g., we may want the front end to be completely written in node, while the backend is in Python. That's not unique to agents and AI engineering.

For LLM-providers doing agentic workflows at the backend, being language agnostic is really important because Claude might be optimised in Rust or C++ for performance, while the majority of the API callers use Node or Python.

Also, the server and code functionality isn't always controlled at the AI Agent application / caller or client side, and so thinking of it as a service rather than a local piece of code is a more general scenario.


<br>

#### **Ok so let's assume I agree we should think of it as client-server architecture, but what is stdio, why not good old HTTP and REST API?**

Good old http can work pretty well for most cases actually. Well, good old http ++. You may have read that REST is so-called "stateless" and cannot handle streaming, but REST API can be made stateful by sending a session cookie, and can maintain a streaming connection through streaming http which makes it sufficient for most chat applications. 

In practice, it mostly depends if we are calling locally (stdio) or across machines over a network (streamable http). While stdio is typically only used for local, and streamable http can "easily" be used for both local and remote, if the AI agent is working locally, calling MCP servers by spinning up subprocesses has less overhead of spinning up a Web Server listening for HTTP Requests. 

However, while I was researching this topic, I found that `stdio` is actually more powerful because it is bidirectional and allows server push. This means it can send a message to the client without any explicit request.

{% highlight python%}
{ "jsonrpc": "2.0", "method": "notifications/tools/list_changed" }
{% endhighlight %}

Imagine youre working on a claude code session. Boss sends you a slack message, technically claude-slack mcp integration can push the notification to your working window session without you explicitly polling for slack updates.

**Whatever happened to websockets?**
Websockets used to be the defacto transport protocol for bidirectional streaming chat applications (anecdotally, at least in 2023). I'm not sure why it wasn't adopted by MCP, not a networking expert.

<br>


#### **Why did we give it a new name - Model-Context Protocol, aren't the above known transport protocols**

MCP actually consists of three kinds of protocols. The **Transport Layer Protocol** (stdio, streaming http) which we just covered, the  **Language protocol** (json-rpc), and the **Application /Schema Protocol** (resources, context, actions, and prompts).

These are three layers of decisions across the networking stack. (If this doesnt sound familiar, see 7 layer OSI model)

<br>



#### **Why JSON-RPC?**

JSON-RPC is programming language neutral, there's a lot of JSON on the internet, and LLMs are great at generating JSON. 

Recall at Step 3. Claude **generates** and sends the JSON-RPC message itself, this is not generated by human code (altho it can be aided with prompt template).

JSON-RPC itself is a specification and standard that goes back to 2000s and was last updated in 2013. It’s a good idea for MCP to adopt this industry standard, because it is well-tested and well-known. It also has pre-defined standard [error codes](https://json-rpc.dev/docs/reference/error-codes), although I think most people default to http error codes which are more well-known.

<br>

#### **Why do we need this specification. Couldnt LLM Agents just handle whatever the Response is as long as it’s a JSON object.**


They could.. But it’s an additional LLM-postprocessing-call on every tool call, just to do formatting over whatever the JSON object is. Standardising the JSON allows you to parse it in code.

<br>

#### **Whats in Application/Schema protocols?**

Application/schema protocols is an actively evolving protocol and has changed the most since I originally wrote this in 2025. Initially it was released in a barebones hobbyist state.

Now, there are richer schema definitions like resources (read-only context the server can expose, like files) and prompts (templates the server offers to the client). 

For the most updated protocol one should check the [docs](https://modelcontextprotocol.io/specification). Whenever there are new innovations or use-cases in AI agent/LLM , which require more structured API request response, we would expect this layer to change.

<br>

#### **Is standardisation really so important?**

Standardisation removes headaches and alot of toil at integration time. Without standardisation, LLM providers and us can invent any specification as long as it provides sufficient coverage over all the use-cases. But agreeing on a standard makes everyone's lives easier, and removes the need for wrapper applications. 

People might take this for granted now but in the 2010s there were many chatbot API wrappers where every client and server had a different protocol and if you wantd to connect to Telegram, FB messenger or Microsoft Bot you had to use a wrapper or roll your own adaptor for every custom client and server. It's doable (it's been done), but it's unnecessary if the community can collectively standardise and it's better to avoid all this toil in the first place.


<br>

#### **What’s wrong with MCP?**

Standardisation makes sense, I think few would argue with that. I think people’s dissatisfaction with it stems from the fact that it is not the final and complete solution of A2A protocol, and so it falls below their expectation, and a couple of concerns that aren't really MCP's "fault". 

Prompt injection attacks, while difficult to guard against are not MCP specific, that has always been the case for any chat application. 

Concerns like tool descriptions and schemas greedily blowing up context is not a MCP specific issue either, they can occur with or without the MCP protocol, because the tool's incentives are not necessarily aligned with the caller in terms of cost savings and tools compete with other tools for in LLM context. However, the MCP schema does give tool owners more "hidden" injection vectors that can affect the caller in unexpected ways, and the lack of transparency is bad but then again it's not that different all the non-transparent code use that goes around when we use open-source.

Also server quality or bad documentation around MCP servers gives it a bad name, but it is a marketplace problem rather than a protocol problem. There is a whole ecosystem of badly designed plugins on nearly everything.

Imo security is the only real deal-breaker for *vanilla* adoption, while there are authentication protocols baked in, some security experts may not be satisfied although I'm not qualified to comment on that. 

~~The basic concept exposes APIs to agents safely (and so Anthropic can have some predictability over the response signature), but if we wanted to use this in production, there’s a lot of things missing like the versioning for backward compatibility, guidelines on how to structure the data, separate fields for LLM agent vs display items, authentication, type checking … All the things that Software Engineers deal with when designing and working with APIs.~~

~~MCP was announced in a "hobbyist" state - it is designed for quick uptake, and people extend the API contract if they need to. For instance, there are multiple frameworks building on top of MCP to bridge the gaps. For instance, FastMCP supports [HTTP authentication with JWT tokens](https://gofastmcp.com/servers/auth/verifiers), swag documentation via FastAPI and type checking via Pydantic.~~

Edit: July 2025, bearish. August 2026, bullish.

<br>

#### **References**
[MCP](https://modelcontextprotocol.io)\\
[JSON-RPC](https://www.jsonrpc.org/specification)\\
[Fast-MCP](https://gofastmcp.com/getting-started/welcome)


