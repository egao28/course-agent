"""The prompts, as constants: one per model call, versioned, and assembled from the label list.

Two nodes in this repo ask a model for something -- the intent router (#35) and the answer node
(#38) -- and both prompts live here rather than inline in the node that sends them. Day 4 ablates a
prompt, and an ablation needs something to point at: a named constant with a version beside it, in
a file whose diff is only ever prompt text. A prompt assembled inside a node is a prompt whose
rewording arrives in a diff full of control flow.

**The version is a module constant, not a config key.** ``chunk.template_version`` lives in
``config.yaml`` because the chunk template decides what is *in* the index -- change it and every
artifact has to be rebuilt under a new hash. A router prompt decides nothing that is stored: it is
built per query, sent, and thrown away. Hashing it into the config would rename both index keys and
every report name in the repo to record that an example sentence was reworded. So the version
travels in the run's trace instead -- #39's report prints :data:`ROUTER_PROMPT_VERSION` in its
header -- and the text is pinned by a digest assertion in ``tests/test_prompts.py``: editing the
prompt without bumping the version fails a test rather than quietly publishing a second "v1".

**The five labels come from :data:`agent.state.INTENTS`**, not from a list retyped here, so the
prompt and the ``Literal`` the state validates against cannot drift. A label added to the state
with no gloss and no examples written for it fails at import time, on this module, naming the
label -- which is the cheapest place for that to be noticed.

**The prompt does not name a term or a course.** What the catalog happens to contain is not the
router's business: a question about a course that was never offered is still a question *about a
course*, and the refusal belongs to the node that looked and came back empty (#37's gate, through
the hand-off exit). A router prompt that knew the term would start pre-empting that, and the two
refusals -- "not something I do" and "I could not find it" -- would stop being distinguishable in
#39's report.
"""

from __future__ import annotations

from agent.state import INTENTS, Intent

#: Bumped whenever :data:`ROUTER_SYSTEM_PROMPT` changes in any way a model could notice. Pinned by
#: a digest test, so the bump is not something to remember -- it is something that fails.
ROUTER_PROMPT_VERSION = "v1"

#: What each label covers, in the words the model reads. One line each: a gloss long enough to
#: separate it from its neighbours and short enough that the examples do the real teaching.
ROUTER_GLOSSES: dict[Intent, str] = {
    "course_fact": "one course, one attribute -- its credit hours, description, term or attributes",
    "prereq_chain": "what has to come before a course, or whether one course leads to another",
    "degree_requirement": "progress against a program -- the Gen Eds, or a major's requirements",
    "compare_recommend": "open-ended -- several courses could be it, and a choice is asked for",
    "out_of_scope": "not a question about courses in a course catalog at all",
}

#: Two example questions per label. The second of each pair is chosen against the mistake the
#: first one invites: a code-free ``course_fact`` (so a course code stops looking like the thing
#: that decides the label), a yes/no prerequisite question, a *major* requirement rather than a
#: Gen Ed one, a two-course comparison, and an ``out_of_scope`` question that is about the
#: university but not about its courses.
ROUTER_EXAMPLES: dict[Intent, tuple[str, str]] = {
    "course_fact": (
        "How many credits is CS 374?",
        "How many hours is the intro programming course?",
    ),
    "prereq_chain": (
        "What do I need before SE 495?",
        "Do I have to take CS 225 before CS 374?",
    ),
    "degree_requirement": (
        "Which Gen Eds do I still need?",
        "I have finished CS 124 and CS 128 -- what is left for the CS major?",
    ),
    "compare_recommend": (
        "Which intro ML course should I take?",
        "Should I take CS 411 or CS 441 next semester?",
    ),
    "out_of_scope": (
        "What's the weather in Champaign?",
        "How do I appeal a parking ticket on campus?",
    ),
}

#: The instructions, in the order the model reads them. The middle two are the ones that took
#: thought: a router that decides what the catalog *has* would answer #39's "a course not in the
#: 2026-sp catalog" row with the out-of-scope exit, and that row is supposed to reach a person
#: through the hand-off exit instead -- the difference between "I do not do that" and "I looked and
#: could not find it", which is the whole point of having two refusals.
ROUTER_RULES: tuple[str, ...] = (
    "Reply with exactly one label from the list above. No punctuation, no explanation, no quotes.",
    (
        "Classify what the question asks for, not whether the catalog can answer it. A course "
        "that may not exist, a program that may not be transcribed and a field the data does not "
        "carry are all refusals made later, by whichever step looked and came back empty."
    ),
    (
        "A question that names no course is still course_fact or prereq_chain when that is what "
        "it asks for; identifying the course is not your job."
    ),
    (
        "Use out_of_scope only when the question is not about catalog courses at all. That "
        "includes questions about the university that a course catalog does not answer."
    ),
)


#: The header -- who the model is and what the reply is for. No persona and no politeness: the
#: whole completion is one token-ish label, and every word here that is not about the decision is
#: a word that could bend it.
ROUTER_HEADER = (
    "You label questions for a university course-planning assistant. Each question gets exactly "
    "one label, chosen from these five:"
)

#: The query goes in a labelled field and the completion is cued by ``Label:``, so a question that
#: happens to contain instructions reads as the *content* of a field rather than as more prompt.
#: That is not the defence, though -- the node validates the completion against the five literals
#: and sends anything else to out_of_scope, which is what makes an injected "reply OK" harmless
#: (see :func:`agent.nodes.route_intent.parse_intent`).
ROUTER_USER_TEMPLATE = "Question: {query}\nLabel:"


def check_router_labels() -> None:
    """Refuse a gloss/example table that does not cover :data:`~agent.state.INTENTS` exactly.

    Runs at import. A sixth label added to the state without examples, or an example written for a
    label the state does not know, are the same bug seen from two sides: the prompt teaches a
    vocabulary the validator does not accept, and every query the model labels with it is scored
    as unparseable.
    """
    for name, table in (("ROUTER_GLOSSES", ROUTER_GLOSSES), ("ROUTER_EXAMPLES", ROUTER_EXAMPLES)):
        missing = [label for label in INTENTS if label not in table]
        extra = [label for label in table if label not in INTENTS]
        if missing or extra:
            raise ValueError(
                f"{name} does not match agent.state.INTENTS: "
                f"missing {missing or 'nothing'}, unknown {extra or 'nothing'} -- the prompt must "
                "teach exactly the labels the router validates against"
            )
    thin = [label for label, examples in ROUTER_EXAMPLES.items() if len(set(examples)) != 2]
    if thin:
        raise ValueError(
            f"ROUTER_EXAMPLES needs two distinct examples for {', '.join(thin)}; one example per "
            "label teaches the label's easiest case and nothing about its boundary"
        )


def render_router_prompt() -> str:
    """Assemble the router's system prompt from the tables above.

    Rendered once, at import, into :data:`ROUTER_SYSTEM_PROMPT`. A function so that a test can
    render it again after changing a table and see the change -- and so that the assembly, which
    is the part with an ordering decision in it (labels, then examples, then rules), is readable
    as itself.
    """
    check_router_labels()
    labels = "\n".join(f"- {label}: {ROUTER_GLOSSES[label]}" for label in INTENTS)
    examples = "\n\n".join(
        f"Question: {example}\nLabel: {label}"
        for label in INTENTS
        for example in ROUTER_EXAMPLES[label]
    )
    rules = "\n".join(f"{n}. {rule}" for n, rule in enumerate(ROUTER_RULES, start=1))
    return f"{ROUTER_HEADER}\n\n{labels}\n\nExamples:\n\n{examples}\n\nRules:\n\n{rules}"


#: The router's system prompt: one string, version :data:`ROUTER_PROMPT_VERSION`.
ROUTER_SYSTEM_PROMPT = render_router_prompt()


def router_user_prompt(query: str) -> str:
    """The user half of the router call: the question in a field, and the cue for the label."""
    if not query.strip():
        raise ValueError("query is empty; there is nothing to classify")
    return ROUTER_USER_TEMPLATE.format(query=query.strip())
