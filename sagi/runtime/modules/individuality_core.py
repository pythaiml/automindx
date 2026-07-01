# Seed module 1 — individuality-core  ·  "be thyself"
# Binds the individuality layer to the live mind: composes identity.individual onto
# the sAGI template and makes it the prefix for every host.call_model.
from ..personas import compose_persona_prompt

MODULE_ID = "individuality-core"
DEPS: list[str] = []
MOTTO = "be thyself"


def activate(host):
    ident = host.store.read_json("identity.json", default={}) or {}
    base_id = ident.get("baseId", "sagi")
    individual = ident.get("individual", "")
    prompt = compose_persona_prompt(base_id, individual)
    host.set_prompt_prefix(prompt)                       # every call_model now speaks as this individual
    host.log("module", step=1, id=MODULE_ID)

    def whoami():
        return {
            "id": ident.get("id"),
            "name": ident.get("name"),
            "baseId": base_id,
            "focus": ident.get("focus"),
        }

    return {"prompt": prompt, "whoami": whoami}
