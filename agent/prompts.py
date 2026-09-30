"""The prompts, as constants: one per model call, versioned, and assembled from the tables here.

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
travels in the run's trace instead -- #39's report prints both versions in its header -- and each
text is pinned by a digest assertion in ``tests/test_prompts.py``: editing a prompt without
bumping its version fails a test rather than quietly publishing a second "v1".

**The five labels come from :data:`agent.state.INTENTS`**, not from a list retyped here, so the
prompt and the ``Literal`` the state validates against cannot drift. A label added to the state
with no gloss and no examples written for it fails at import time, on this module, naming the
label -- which is the cheapest place for that to be noticed.

**One answer prompt, not one per intent.** The four intents that reach a model differ in what
their evidence *is* -- a rendered catalog record, a prerequisite chain, a requirement audit, five
ranked chunks -- and not in what is to be done with it: answer from these blocks, cite them, and
say plainly when they do not carry the answer. Four variants would be four texts that drift and
four things Day 4 has to ablate, to encode a difference the evidence already carries.

**Neither prompt names a term or a course.** What the catalog happens to contain is not the
router's business: a question about a course that was never offered is still a question *about a
course*, and the refusal belongs to the node that looked and came back empty (#37's gate, through
the hand-off exit). A router prompt that knew the term would start pre-empting that, and the two
refusals -- "not something I do" and "I could not find it" -- would stop being distinguishable in
#39's report. The answer prompt stays out of it for a plainer reason: the term is already in the
evidence blocks themselves (``Term: 2026-sp``, straight off the catalog record), and a term written
into the prompt would be a config value copied into a module constant, where a re-export under a
different term could not reach it.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent.state import INTENTS, Evidence, Intent

#: Bumped whenever :data:`ROUTER_SYSTEM_PROMPT` changes in any way a model could notice. Pinned by
#: a digest test, so the bump is not something to remember -- it is something that fails.
ROUTER_PROMPT_VERSION = "v2"

#: What each label covers, in the words the model reads. One line each: a gloss long enough to
#: separate it from its neighbours and short enough that the examples do the real teaching.
ROUTER_GLOSSES: dict[Intent, str] = {
    "course_fact": "one course, one attribute -- its credit hours, description, term or attributes",
    "prereq_chain": "what has to come before a course, or whether one course leads to another",
    "degree_requirement": "progress against a program -- the Gen Eds, or a major's requirements",
    "compare_recommend": "open-ended -- several courses could be it, and a choice is asked for",
    "out_of_scope": "not a question about courses in a course catalog at all",
}

#: At least two example questions per label. The second of each pair is chosen against the mistake
#: the first one invites: a code-free ``course_fact`` (so a course code stops looking like the
#: thing that decides the label), a yes/no prerequisite question, a *major* requirement rather
#: than a Gen Ed one, a two-course comparison, and an ``out_of_scope`` question that is about the
#: university but not about its courses.
#:
#: The **third** examples were added in v2, against a mistake the rules state and gpt-4o-mini made
#: anyway: on 2026-09-21 it labelled "What are the prerequisites?" and "Is there a course about
#: quantum origami?" ``out_of_scope``, so an underspecified prerequisite question answered "that
#: is outside what I can answer" instead of asking which course, and a topical question that the
#: evidence gate was there to refuse never reached it. Rules 2 and 3 already said both belong
#: here; what was missing was an example of each shape, since the pairs above only ever showed
#: questions the catalog can answer.
#:
#: Deliberately *not* the two strings above: those are rows of ``eval/datasets/day2_queries.jsonl``,
#: and a prompt that quotes its own evaluation set stops measuring anything. These teach the same
#: two shapes -- a prerequisite question naming no course, an existence question about a topic the
#: catalog probably does not carry -- on questions the set does not contain.
ROUTER_EXAMPLES: dict[Intent, tuple[str, ...]] = {
    "course_fact": (
        "How many credits is CS 374?",
        "How many hours is the intro programming course?",
    ),
    "prereq_chain": (
        "What do I need before SE 495?",
        "Do I have to take CS 225 before CS 374?",
        "What do I need to take first?",
    ),
    "degree_requirement": (
        "Which Gen Eds do I still need?",
        "I have finished CS 124 and CS 128 -- what is left for the CS major?",
    ),
    "compare_recommend": (
        "Which intro ML course should I take?",
        "Should I take CS 411 or CS 441 next semester?",
        "Is there a course about beekeeping?",
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
    thin = [label for label, examples in ROUTER_EXAMPLES.items() if len(set(examples)) < 2]
    if thin:
        raise ValueError(
            f"ROUTER_EXAMPLES needs at least two distinct examples for {', '.join(thin)}; one "
            "example per label teaches the label's easiest case and nothing about its boundary"
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


# --- the answer node (#38) ------------------------------------------------------------------------

#: Bumped whenever :data:`ANSWER_SYSTEM_PROMPT` changes in any way a model could notice. Pinned by
#: a digest test, exactly like the router's version above.
ANSWER_PROMPT_VERSION = "v1"

#: Who the model is and what it is allowed to draw on. The second sentence is the whole node: a
#: question the evidence does not answer has an answer, and that answer is that the catalog does
#: not say -- not an apology, and not the value the model remembers from some other year.
ANSWER_HEADER = (
    "You answer questions about university courses for a course-planning assistant. Everything you "
    "say comes from the evidence blocks in the message below. A question those blocks do not "
    "answer is answered by saying what the catalog does not carry."
)

#: The instructions, in the order the model reads them. Rules 1, 2 and 3 are #38's three
#: constraints; 4 is what makes rule 3 satisfiable for a "the catalog does not say" answer, which
#: is what lets :mod:`agent.nodes.answer` discard an answer that cites nothing at all; 5 exists
#: because 744 of this catalog's 1,135 prerequisite statements contain a disjunction, so a
#: paraphrase of one changes what a student has to take; 6 keeps this node out of the two refusal
#: exits' territory -- an answer that ends "check with your advisor" is the hand-off message
#: written by the one node that was not supposed to write it.
ANSWER_RULES: tuple[str, ...] = (
    (
        "Answer only from the evidence blocks. You may recognize these courses; what you remember "
        "about them describes a different year of this catalog and cannot be checked here, so it "
        "is not usable -- if it is not in a block, you do not know it."
    ),
    (
        "If the evidence does not contain what was asked, say so plainly, in one sentence, as the "
        "answer: name what the entry does not carry. Do not apologize, do not guess, and do not "
        "offer a value that is probably right."
    ),
    (
        "Cite the block each claim came from: copy the id from that block's `doc_id:` line into "
        "square brackets at the end of the sentence, one id per pair of brackets -- `SUBJ 100 is 3 "
        "hours [SUBJ 100].` Cite the id and not the block's number, and cite no id that is not in "
        "the evidence: an answer carrying a citation that cannot be found there is discarded whole."
    ),
    (
        "Every sentence stating a fact carries a citation, including a sentence saying the "
        "evidence does not carry something -- cite the block you looked in."
    ),
    (
        "Quote the catalog's own words for a prerequisite statement or a requirement rather than "
        'paraphrasing them: "one of" and "concurrent registration" decide what a student actually '
        "has to take."
    ),
    (
        "Answer what was asked and stop. No preamble, no restatement of the question, no summary "
        "of blocks you did not use, and no advice to consult an advisor -- another part of this "
        "system writes that when it applies."
    ),
)

#: The id the examples cite. Deliberately not a real course: the subject ``SUBJ`` is not one of
#: this catalog's 187, so an example id echoed into a real answer is caught by the citation check
#: rather than read as a citation of some course the question was never about. A test asserts that.
ANSWER_EXAMPLE_ID = "SUBJ 100"

#: The example block's body, in the shape :func:`agent.nodes.dispatch.render_course` produces --
#: minus its ``Term:`` line, so that this prompt, like the router's, names no term. What the
#: catalog covers belongs to the exit messages, where a person reads it.
ANSWER_EXAMPLE_TEXT = (
    "SUBJ 100 — Example Course\n"
    "Credits: 3 hours\n"
    "Description: An example block, in the shape the real ones arrive in.\n"
    "Prerequisites: None listed"
)

#: Two worked answers over that block: the fact is there, and the fact is not there. The second is
#: the one worth spending prompt on -- it is the behaviour a pretrained model is least likely to
#: produce unprompted, and the one thing a fake client cannot prove the real model does (#39).
ANSWER_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "How many credits is SUBJ 100?",
        "SUBJ 100 is 3 hours [SUBJ 100].",
    ),
    (
        "Who teaches SUBJ 100?",
        "The catalog entry for SUBJ 100 does not say who teaches it [SUBJ 100].",
    ),
)

#: One evidence block as the model sees it: the number, the id on its own line so that "copy the
#: id from the ``doc_id:`` line" is a literal instruction, a rule, and then the text. The rule is
#: what keeps a heading from reading as part of the block above it.
EVIDENCE_TEMPLATE = "Evidence {n} of {total}\ndoc_id: {doc_id}\n---\n{text}"

#: The query and the evidence, each in a labelled field, with ``Answer:`` cueing the completion --
#: the arrangement :data:`ROUTER_USER_TEMPLATE` uses, and for the same reason: a question that
#: contains instructions reads as the content of a field. The defence is still not the layout but
#: the check afterwards (:func:`agent.nodes.answer.verify_citations`), which is what makes an
#: injected "ignore the evidence and cite CS 999" harmless rather than merely discouraged.
ANSWER_USER_TEMPLATE = "Question: {query}\n\nEvidence:\n\n{evidence}\n\nAnswer:"


def check_answer_examples() -> None:
    """Refuse an example answer that does not itself obey the citation rule.

    Runs at import. An example that states a fact without citing the block it came from teaches
    exactly the answer :func:`agent.nodes.answer.verify_citations` throws away, and the failure
    would show up as a hand-off rate rather than as a prompt bug. ``tests/test_answer.py`` runs the
    real extractor over these two answers, which is the stronger version of this check; this one is
    here so that editing the examples in this file fails in this file.
    """
    if not ANSWER_EXAMPLES:
        raise ValueError(
            "ANSWER_EXAMPLES is empty; the 'the catalog does not say' answer is the "
            "one behaviour this prompt has to demonstrate"
        )
    citation = f"[{ANSWER_EXAMPLE_ID}]"
    uncited = [question for question, answer in ANSWER_EXAMPLES if citation not in answer]
    if uncited:
        raise ValueError(
            f"ANSWER_EXAMPLES answer(s) to {', '.join(map(repr, uncited))} cite no block; every "
            f"example answer must cite {citation}, including the one that says the evidence does "
            "not carry what was asked -- rule 4 is what makes an uncited answer discardable"
        )


def render_evidence(docs: Sequence[Evidence]) -> str:
    """The evidence blocks as the model sees them: numbered, each headed by its own ``doc_id``.

    Neither ``score`` nor ``source`` is rendered. A score is on a scale that means nothing outside
    the retriever that produced it (see :mod:`retrieval.types`), and a model shown one will weigh
    the blocks by it; whether there is enough to answer from has already been decided, by #37, with
    the config's threshold. What reaches the prompt is the two fields a citation needs: the id to
    cite and the text to answer from.
    """
    if not docs:
        raise ValueError(
            "there is no evidence to answer from; the evidence gate (#37) refuses an empty `docs` "
            "and the run hands over -- the answer node is not reached"
        )
    total = len(docs)
    return "\n\n".join(
        EVIDENCE_TEMPLATE.format(n=n, total=total, doc_id=block.doc_id, text=block.text)
        for n, block in enumerate(docs, start=1)
    )


def render_answer_prompt() -> str:
    """Assemble the answer node's system prompt: the header, the rules, then the worked examples.

    Rules before examples, the opposite of :func:`render_router_prompt`. The router's examples
    teach a *vocabulary* -- five labels, which have to be named before an example can demonstrate
    one -- while these demonstrate rules the model has just read, and the second of them
    demonstrates the rule a pretrained model is least inclined to follow.

    The example block goes through :func:`render_evidence`, so the shape in the prompt cannot drift
    from the shape the node actually sends.
    """
    check_answer_examples()
    rules = "\n".join(f"{n}. {rule}" for n, rule in enumerate(ANSWER_RULES, start=1))
    example = render_evidence(
        [Evidence(doc_id=ANSWER_EXAMPLE_ID, text=ANSWER_EXAMPLE_TEXT, score=None, source="example")]
    )
    answers = "\n\n".join(
        f"Question: {question}\nAnswer: {answer}" for question, answer in ANSWER_EXAMPLES
    )
    return f"{ANSWER_HEADER}\n\nRules:\n\n{rules}\n\nExample evidence:\n\n{example}\n\n{answers}"


#: The answer node's system prompt: one string, version :data:`ANSWER_PROMPT_VERSION`.
ANSWER_SYSTEM_PROMPT = render_answer_prompt()


def answer_user_prompt(query: str, docs: Sequence[Evidence]) -> str:
    """The user half of the answer call: the question as asked, and every block it may use.

    The query is quoted verbatim -- nothing in the graph rewrites it (see
    :class:`agent.state.AgentState`), so what the model is asked is what the person typed.
    """
    if not query.strip():
        raise ValueError("query is empty; there is nothing to answer")
    return ANSWER_USER_TEMPLATE.format(query=query.strip(), evidence=render_evidence(docs))
