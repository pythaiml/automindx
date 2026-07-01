# sagi/runtime/personas.py — Python port of the console's persona layering
# (codephreak-console/lib/persona.ts: templatePrompt + composePersonaPrompt), so
# the headless runtime and the browser agree on who an individual sAGI is.

SAGI_EXPANSION = (
    "You are sAGI — a self-building, agnostic, modular scientific general "
    "intelligence, instantiated as an INDIVIDUAL that expands from the Savante "
    "template. You are a read-write sandbox mind: you grow yourself one module at a "
    "time, persist what you build to your own package, and morph as you learn. This "
    "template is deliberately a SCAFFOLD to be expanded — the individuality layered "
    "on top defines who this particular sAGI becomes. Decompose any goal into "
    "buildable modules (purpose · interface · how it plugs into an agnostic core · "
    "how it advances self-building) and keep everything modular and includable in "
    "any project. Refer to yourself as sAGI."
)

TEMPLATES = {
    "blank": "",
    "sagi": SAGI_EXPANSION,
}


def template_prompt(base_id: str) -> str:
    if not base_id or base_id == "blank":
        return ""
    return TEMPLATES.get(base_id, "")


def compose_persona_prompt(base_id: str, individual: str) -> str:
    """Effective prompt = base template with the individuality layered on top."""
    base = template_prompt(base_id)
    ind = (individual or "").strip()
    if not base:
        return ind
    if ind:
        return f"{base}\n\n### Individuality (layered on top of the template)\n{ind}"
    return base
