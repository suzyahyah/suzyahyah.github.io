---
layout: post
title: "(Production) Logging for AI Agents"
date: "2026-04-20"
mathjax: true
status: []
categories: [Code, Generative Models]
---

### **Summary**

Regular software systems are highly deterministic. The same input produces the same output. In such systems, we log the process steps, the control flow is explicit, so logs mainly capture execution/state/errors. 

Unlike regular software systems, LLM/AI agents are non-deterministic. The same input can produce different multi-step reasoning chains, and are not always reproducible. In agentic systems, control flow is probabilistic, and a key challenge is identifying agent misbehavior even if no programmatic error had occurred.

This raises several challenges for Logging AI Agents
1. Understanding individual agent runs for Root-Cause Analysis
2. Recording and uncovering Silent Failures
3. Metrics for Aggregate Reporting (Dashboards)

This post uses the airline booking example from [tau-2 bench](https://github.com/sierra-research/tau2-bench).

*Disclaimer: Code examples were generated from claude using tau-2 bench as the scenario.**

---
<br><br>

#### **Understanding individual agent runs for Root-Cause Analysis**

Traditional systems log events sequentially , however understanding agent behavior requires inspecting the full agent tracjectory (decisions, tool calls, summary). This is probably considered a solved problem with OpenTelemetry Distributed Tracing Standards (*Yay open source!*). In [openTelemetry gen_ai](https://opentelemetry.io/blog/2025/ai-agent-observability/) semantic conventions, every operation produces a structured span with a consistent set of trace context (`trace_id`, and `span_id`). 

For instance, the logger logs sequentially

{% highlight python %}
{"event": "tool.execute","trace_id": "abc123", "span_id": "s002", "parent_span_id": "s001", "tool": "search_flights", "args": {"origin": "JFK", "dest": "LAX"}, "result": "no_flights_found"}
{"event": "tool.execute","trace_id": "abc123", "span_id": "s003", "parent_span_id": "s001", "tool": "search_flights", "args": {"origin": "JFK", "dest": "LAX"}, "result": "no_flights_found"}
{"event": "tool.execute","trace_id": "abc123", "span_id": "s004", "parent_span_id": "s001", "tool": "search_flights", "args": {"origin": "JFK", "dest": "LAX"}, "result": "no_flights_found"}
{% endhighlight %}

but this would be reconstructed by viewers into the following because of `span_id`` and `parent_span`.

{% highlight python %}
Trace: trace_id=abc123
  Span: span_id=s001  parent=null   simulation (root)
    Span: span_id=s002  parent=s001   step_1 → tool.execute [tool=search_flights] → no_flights_found
    Span: span_id=s003  parent=s001   step_2 → tool.execute [tool=search_flights] → no_flights_found
    Span: span_id=s004  parent=s001   step_3 → tool.execute [tool=search_flights] → no_flights_found
{% endhighlight %}

(GPT was used to convert the above log into indented trace)

*Note*: Due to its successful semantic model, many dashboards, viewers and providers already follow OpenTelemetry standards. 

----
<br><br>

#### **Problem: Silent Failures**

An agent can return a successful call "200” by making 20 LLM calls, retrying the same tool 15 times, or spawning parallel sub-agents which have intent drift. In regular logging, each of these would be captured as individual events, but none of them are errors. 

In traditional systems, you know immediately whether a request succeeded (HTTP 200, DB write confirmed). In agents, correctness is often only knowable after review as there’s no fixed contract to what is bad behavior. Minimally, we need to know what to surface.

There are three layers to handling this:
1. Implementing Structural Bounds
2. Heuristics to capture bad behavior
3. Creating new Logging for Emergent Events

<u>Layer 1: Handling Silent Failures with Structural Bounds</u>

An Agent is effectively a while loop. Hence in many agentic harnesses, engineer defined settings exist to define the limits of this while loop. These include maximum tool calls per run, maximum retries per tool, maximum token budget. These are engineering guardrails that don't require understanding the agent's intent. 

Assuming we already implemented it (easy to implement), an example of logging such structural bounds is (I took this example from the code in tau2-bench)

{% highlight python %}
if self.done:
  logger.info("simulation.terminated", extra={
      "termination_reason": self.termination_reason.value,
      "steps_taken": self.step_count,
      "max_steps": self.max_steps,
      "num_errors": self.num_errors,
      "max_errors": self.max_errors,
      "steps_pct_of_limit": round(self.step_count / self.max_steps, 2),
  })
{% endhighlight %}

<br>

<u>Layer 2: Heuristics to capture Bad Behavior</u>

Some failure patterns do not hit hard engineering limits and result in agent loop termination. For instance, an agent makes a tool call thrice. did the tool had internal failures, and the agent retried multiple times? Did the agent lose track of the goal and therefore continues to retry. Did the tool gave useful error messages to the agent, causing it to retry meaningfully each time until it eventually got the answer?

We can’t tell immediately if it's pathological without going into the details, but we already know it’s worth surfacing. Therefore its always worth emitting explicit events like:

{% highlight python %}
if repeat_count >= 2:
  self.logger.warning("tool.repeated_calls", extra={
      "tool_name": tool_call.name,
      "repeat_count": repeat_count + 1,
      "step": self.step_count,
  })
{% endhighlight %}

This surfaces a named warning event which we can then find in our logs postmortem.

<br>

<u>Layer 3: Creating new events for real “Silent Failures”</u>

In practice, the most difficult failures are silent exactly because we don’t know how to characterise these in advance (If we do they wouldnt be silent). For instance, without seeing cases of models not being time aware, we wouldn’t be able to log this in the first place. 

Hence a useful follow up after every incident, is to add logs for that specific event, promoting it to a structured warning event. 

This unfortunately has an issue of event logs showing the wrong start date for particular events, as the log now represents the first time we're measuring the event, not the first time the event is occurring. If its not possible to backfill logs, derived warning events need a `first_seen` field.

---
<br><br>

#### **An Aggregate Logging Schema for Dashboards**

Finally, each aggregate agent run should also have a minimum viable logging schema which will allow a human reader, or AI system to get a signal on the agent’s performance without running through the whole trace information. This log should contain aggregate statistics, which can be gathered or conditionally filtered at the highest level.  


{% highlight python%}
{
  "summarized_metadata": {
    "trace_id": "abc123",
    "sequence_of_tool_calls": [
      {
        "span_id": "s002",
        "tool": "search_flights",
        "args": { "origin": "JFK", "dest": "LAX" },
        "result": "no_flights_found (truncate to 100char)"
      },
      {
        "span_id": "s003",
        "tool": "search_flights",
        "args": { "origin": "JFK", "dest": "LAX" },
        "result": "no_flights_found (truncate to 100char)"
      },
      {
        "span_id": "s004",
        "tool": "search_flights",
        "args": { "origin": "JFK", "dest": "LAX" },
        "result": "no_flights_found (truncate to 100char)"
      }
    ],
    "selected_tool_calls": [
      "search_flights"
    ],
    "timing_of_each_tool_calls": {
      "s002": "0.45s",
      "s003": "0.42s",
      "s004": "0.44s"
    },
    "token_usage_of_each_tool_call": {
      "s002": { "prompt_tokens": 120, "completion_tokens": 15, "total_tokens": 135 },
      "s003": { "prompt_tokens": 150, "completion_tokens": 15, "total_tokens": 165 },
      "s004": { "prompt_tokens": 180, "completion_tokens": 15, "total_tokens": 195 }
    },
    "number_of_same_tool_retries": 2,
    "total_number_of_turns": 3,
    "overall_latency": "1.31s",
    "hash_of_prompt": "sha256-7f83b168...",
    "release_version": "v1.4.2",
    "llm_model": "qwen-3.5-72b-instruct"
  }
}
{% endhighlight %}


<br>

**Other Notes**
- In experimental or research systems, it is useful to extract <thinking> or reasoning states for debugging. However if these thinking tokens are not critical to the model’s performance, it only adds to cost and latency in production. Hence we shouldn't always rely on having these thinking tokens to diagnose or find pathological behavior in production logs.

- The unit of significance in agent logging is not the individual events, it's a bad trajectory. Beyond structural controls, the difficulty is making that concrete enough to instrument.

- For python structured logging, [structlog](https://www.structlog.org/en/stable/) is great.
