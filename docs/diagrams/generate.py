"""Generate the README diagrams as SVG, in a light and a dark version.

    python docs/diagrams/generate.py

Writes ``docs/images/diagrams/<name>.svg`` and ``<name>-dark.svg``. The README shows the
version that matches the reader's GitHub theme with a ``<picture>`` element. Edit the layout
functions below and run the script again; no third-party packages are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "images" / "diagrams"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


@dataclass(frozen=True)
class Theme:
    suffix: str
    canvas: str
    card: str
    border: str
    text: str
    muted: str
    line: str
    accents: dict[str, str]


LIGHT = Theme(
    suffix="",
    canvas="#ffffff",
    card="#ffffff",
    border="#d0d7de",
    text="#1f2328",
    muted="#59636e",
    line="#8c959f",
    accents={
        "orchestrator": "#4f46e5",
        "tool": "#0f766e",
        "vision": "#9333ea",
        "external": "#b45309",
        "input": "#475569",
    },
)
DARK = Theme(
    suffix="-dark",
    canvas="#0d1117",
    card="#161b22",
    border="#30363d",
    text="#e6edf3",
    muted="#9198a1",
    line="#6e7681",
    accents={
        "orchestrator": "#818cf8",
        "tool": "#2dd4bf",
        "vision": "#c084fc",
        "external": "#f59e0b",
        "input": "#94a3b8",
    },
)


class Canvas:
    """A small SVG builder with the few primitives the diagrams need."""

    def __init__(self, width: int, height: int, theme: Theme, title: str) -> None:
        self.width, self.height, self.t = width, height, theme
        self.parts: list[str] = []
        self.title = title

    # --- primitives ----------------------------------------------------------------
    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 13,
        color: str | None = None,
        weight: int = 400,
        anchor: str = "middle",
        mono: bool = False,
        spacing: float = 0,
    ) -> None:
        family = MONO if mono else FONT
        letter = f' letter-spacing="{spacing}"' if spacing else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color or self.t.text}" text-anchor="{anchor}"'
            f"{letter}>{escape(value)}</text>"
        )

    def card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        kind: str,
        tag: str,
        title: str,
        lines: tuple[str, ...] = (),
        dashed: bool = False,
    ) -> None:
        accent = self.t.accents[kind]
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{self.t.card}" '
            f'stroke="{accent if not dashed else self.t.border}" stroke-width="1.5"{dash}/>'
        )
        if not dashed:
            self.parts.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{accent}" '
                'fill-opacity="0.07"/>'
            )
            self.parts.append(
                f'<path d="M{x + 10},{y} H{x + w - 10}" stroke="{accent}" stroke-width="3" '
                'stroke-linecap="round"/>'
            )
        cx = x + w / 2
        block = 14 + 18 + 16 * len(lines)
        top = y + (h - block) / 2 + 10
        self.text(cx, top, tag.upper(), size=10, color=accent, weight=700, spacing=1.2)
        self.text(cx, top + 19, title, size=14.5, weight=600)
        for i, line in enumerate(lines):
            self.text(cx, top + 37 + 16 * i, line, size=12, color=self.t.muted)

    def arrow(
        self,
        d: str,
        *,
        both: bool = False,
        dashed: bool = False,
        head: bool = True,
        color: str | None = None,
    ) -> None:
        color = color or self.t.line
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        end = ' marker-end="url(#head)"' if head else ""
        start = ' marker-start="url(#tail)"' if both else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6" '
            f'stroke-linejoin="round"{dash}{start}{end}/>'
        )

    def label(
        self, x: float, y: float, value: str, *, anchor: str = "middle", backdrop: bool = True
    ) -> None:
        if backdrop:  # hides the line behind the text
            width = 7.0 * len(value) + 12
            left = {"middle": x - width / 2, "start": x - 6, "end": x - width + 6}[anchor]
            self.parts.append(
                f'<rect x="{left}" y="{y - 13}" width="{width}" height="19" rx="9.5" '
                f'fill="{self.t.canvas}"/>'
            )
        self.text(x, y, value, size=12, color=self.t.muted, anchor=anchor)

    def badge(self, x: float, y: float, number: int, color: str) -> None:
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="9" fill="{color}"/>')
        self.text(x, y + 4, str(number), size=10.5, color=self.t.canvas, weight=700)

    # --- output --------------------------------------------------------------------
    def svg(self) -> str:
        marker = (
            '<marker id="{id}" viewBox="0 0 10 10" refX="{ref}" refY="5" markerWidth="7" '
            'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{self.t.line}"/></marker>'
        )
        defs = marker.format(id="head", ref=9) + marker.format(id="tail", ref=9)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" role="img" '
            f'aria-label="{escape(self.title)}"><title>{escape(self.title)}</title>'
            f"<defs>{defs}</defs>"
            f'<rect width="100%" height="100%" fill="{self.t.canvas}"/>'
            + "".join(self.parts)
            + "</svg>\n"
        )


# --- diagrams ----------------------------------------------------------------------------
def architecture(t: Theme) -> Canvas:
    c = Canvas(960, 560, t, "System architecture of the PCB CAD LLM assistant")

    c.card(
        200,
        24,
        260,
        68,
        kind="input",
        tag="Input",
        title="User",
        lines=("Question + up to 3 images",),
    )
    c.card(
        40,
        160,
        580,
        104,
        kind="orchestrator",
        tag="Orchestrator",
        title="Orchestrator LLM",
        lines=(
            "LangChain ReAct agent · one JSON action per step · memory",
            "DeepSeek-R1 via Ollama  ·  GPT-4o  ·  Gemini",
        ),
    )
    c.card(
        680,
        160,
        240,
        104,
        kind="vision",
        tag="Vision agent",
        title="Image descriptor",
        lines=("Granite-Vision 3.2 2B + LoRA", "image → text"),
    )

    tools = [
        (40, "PCB expert", ("Fine-tuned SI / EDA answers",)),
        (360, "RAG agent", ("Documents and live web",)),
        (680, "Simulation agent", ("Text → eCADSTAR run",)),
    ]
    for x, title, lines in tools:
        c.card(x, 344, 240, 88, kind="tool", tag="Tool", title=title, lines=lines)
    resources = [
        (40, "DeepSeek-R1-Distill-Llama-8B", "+ LoRA adapter"),
        (360, "Chroma vector stores", "PDFs · web search engines"),
        (680, "eCADSTAR Electrical Editor", "Windows GUI automation"),
    ]
    for x, title, sub in resources:
        c.card(
            x, 476, 240, 64, kind="external", tag="Runs on", title=title, lines=(sub,), dashed=True
        )
        c.arrow(f"M{x + 120},432 V476", dashed=True, head=False)

    # user <-> orchestrator, images to the vision agent
    c.arrow("M300,92 V160")
    c.label(290, 131, "question", anchor="end")
    c.arrow("M360,160 V92")
    c.label(370, 131, "answer", anchor="start")
    c.arrow("M460,58 H800 V160")
    c.label(630, 62, "images")
    c.arrow("M680,212 H620")
    c.label(650, 202, "text")

    # orchestrator <-> tools
    c.arrow("M160,264 V344", both=True)
    c.arrow("M480,264 V344", both=True)
    c.arrow("M590,264 V304 H760 V344", both=True)
    c.arrow("M860,344 V264")
    c.label(870, 309, "result plot", anchor="start")
    c.label(320, 309, "JSON action  ↓   ·   text observation  ↑")
    return c


def query_lifecycle(t: Theme) -> Canvas:
    names = [
        ("User", "input"),
        ("Vision agent", "vision"),
        ("Orchestrator", "orchestrator"),
        ("Simulation agent", "tool"),
        ("eCADSTAR", "external"),
        ("PCB expert", "tool"),
    ]
    xs = [85 + 158 * i for i in range(len(names))]
    steps: list[tuple[int, int, str, bool]] = [
        (0, 1, "attached images", False),
        (1, 2, "image descriptions", True),
        (0, 2, "question", False),
        (2, 3, 'Simulation: "TDR, 100 Ω, 100 mm"', False),
        (3, 4, "patch circuit, run analysis", False),
        (4, 3, "result plot", True),
        (3, 1, "result plot", False),
        (1, 3, "plot description", True),
        (3, 2, "observation (text)", True),
        (2, 5, "explain the reflection", False),
        (5, 2, "domain explanation", True),
        (2, 0, "final answer", True),
    ]
    top, gap = 92, 44
    height = top + gap * len(steps) + 56
    c = Canvas(960, height, t, "Lifecycle of one query through the agents")

    for x, (name, kind) in zip(xs, names, strict=True):
        accent = t.accents[kind]
        c.parts.append(
            f'<path d="M{x},64 V{height - 46}" stroke="{t.border}" stroke-width="1.5" '
            'stroke-dasharray="4 5"/>'
        )
        c.parts.append(
            f'<rect x="{x - 70}" y="20" width="140" height="44" rx="10" fill="{t.card}" '
            f'stroke="{accent}" stroke-width="1.5"/>'
            f'<rect x="{x - 70}" y="20" width="140" height="44" rx="10" fill="{accent}" '
            'fill-opacity="0.08"/>'
        )
        c.text(x, 47, name, size=13.5, weight=600)

    # the orchestrator reasons between receiving the question and calling a tool
    c.parts.append(
        f'<rect x="{xs[2] - 6}" y="{top + gap * 2 + gap // 2 - 4}" width="12" '
        f'height="{gap * 9 + 8}" rx="3" fill="{t.accents["orchestrator"]}" '
        'fill-opacity="0.18"/>'
    )

    for i, (src, dst, text, reply) in enumerate(steps):
        y = top + gap * i + gap // 2
        x1, x2 = xs[src], xs[dst]
        pad = 8 if src == 2 or dst == 2 else 2
        start = x1 + (pad if x2 > x1 else -pad)
        end = x2 - (pad if x2 > x1 else -pad)
        c.arrow(f"M{start},{y} H{end}", dashed=reply)
        c.label((x1 + x2) / 2, y - 13, text, backdrop=False)
        c.badge(x1 + (18 if x2 > x1 else -18), y, i + 1, t.accents[names[src][1]])
    legend_y = height - 22
    c.arrow(f"M30,{legend_y - 4} H70")
    c.text(78, legend_y, "request", size=11.5, color=t.muted, anchor="start")
    c.arrow(f"M150,{legend_y - 4} H190", dashed=True)
    c.text(198, legend_y, "reply", size=11.5, color=t.muted, anchor="start")
    c.parts.append(
        f'<rect x="260" y="{legend_y - 12}" width="12" height="14" rx="3" '
        f'fill="{t.accents["orchestrator"]}" fill-opacity="0.18"/>'
    )
    c.text(280, legend_y, "orchestrator ReAct loop", size=11.5, color=t.muted, anchor="start")
    return c


def rag(t: Theme) -> Canvas:
    c = Canvas(960, 330, t, "Retrieval-augmented generation agent")
    c.card(20, 128, 140, 72, kind="input", tag="Input", title="Question")
    c.card(
        200,
        116,
        200,
        96,
        kind="orchestrator",
        tag="Router",
        title="Router LLM",
        lines=("deepseek-r1:32b", "replies 1, 2 or 3"),
    )
    sources = [
        (24, "1 · Local corpus", ("default Chroma store",)),
        (126, "2 · Uploaded PDFs", ("embedded into Chroma",)),
        (228, "3 · Live web search", ("DuckDuckGo · Google · Brave · Bing",)),
    ]
    for y, title, lines in sources:
        c.card(450, y, 250, 78, kind="tool", tag="Source", title=title, lines=lines)
        c.arrow(f"M400,164 H425 V{y + 39} H450")
        c.arrow(f"M700,{y + 39} H720 V164 H740")
    c.card(
        740,
        116,
        200,
        96,
        kind="orchestrator",
        tag="Generator",
        title="Answer from context",
        lines=("deepseek-r1:32b", "retrieved passages"),
    )
    c.arrow("M160,164 H200")
    c.text(300, 246, "Any other reply falls back", size=11.5, color=t.muted)
    c.text(300, 262, "to the web search", size=11.5, color=t.muted)
    return c


def simulation(t: Theme) -> Canvas:
    c = Canvas(960, 252, t, "Simulation agent: from a sentence to an eCADSTAR result")
    c.card(
        10,
        24,
        170,
        100,
        kind="input",
        tag="Request",
        title="Plain English",
        lines=("“Run a TDR with 100 Ω", "and a 100 mm line”"),
    )
    c.card(
        210,
        24,
        170,
        100,
        kind="tool",
        tag="Parser",
        title="Rule-based parser",
        lines=("SI · FD · TDR", "ohms · millimetres"),
    )
    c.card(
        410,
        24,
        170,
        100,
        kind="orchestrator",
        tag="Contract",
        title="SimulationRequest",
        lines=("TDR · 100 Ω · 100 mm",),
    )
    c.card(
        610,
        24,
        170,
        100,
        kind="external",
        tag="Adapter",
        title="eCADSTAR GUI",
        lines=("patch circuit file", "run on P2P topology"),
    )
    c.card(
        810,
        24,
        140,
        100,
        kind="vision",
        tag="Vision agent",
        title="Result → text",
        lines=("plot described", "for the agent"),
    )
    for x in (180, 380, 580, 780):
        c.arrow(f"M{x},74 H{x + 30}")
    c.card(
        210,
        164,
        170,
        72,
        kind="tool",
        tag="Fallback",
        title="Sentence-BERT",
        lines=("only if no analysis named",),
        dashed=True,
    )
    c.arrow("M295,124 V164", both=True, dashed=True)
    c.text(400, 194, "weak matches are rejected,", size=11.5, color=t.muted, anchor="start")
    c.text(
        400, 210, "so the agent asks instead of guessing", size=11.5, color=t.muted, anchor="start"
    )
    return c


DIAGRAMS = {
    "architecture": architecture,
    "query-lifecycle": query_lifecycle,
    "rag-agent": rag,
    "simulation-agent": simulation,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in DIAGRAMS.items():
        for theme in (LIGHT, DARK):
            path = OUT_DIR / f"{name}{theme.suffix}.svg"
            path.write_text(build(theme).svg(), encoding="utf-8")
            print(f"wrote {path.relative_to(OUT_DIR.parents[2])}")


if __name__ == "__main__":
    main()
