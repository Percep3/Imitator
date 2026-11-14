"""Build the WAIMLAp 2026 talk for the Imitator paper from the official template.

Every number comes from the camera-ready paper (springer_submission/, sha256 6f93...896d7):
results_macros.tex (folds 7-8) and sections/07_results.tex (folds 4-6, Gemma audit table).

Run:  ~/miniconda3/envs/deck-tools/bin/python build_deck.py
"""
import copy
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = HERE / "figures"
TEMPLATE = HERE / "WAIMLAp2026_Plantilla_Presentacion.pptx"
OUT = HERE / "imitator_waimlap2026.pptx"

# WAIMLAp palette (template theme)
PURPLE, TEAL, NAVY = "5D239C", "40A6B8", "0E285C"
TEXT, GREY = "464F5E", "6B7380"
LT_TEAL, TEAL_LINE, LAVENDER = "EAF8F9", "C9ECF0", "D9C8EE"
CONTEXT = "B4BAC4"  # neutral grey for comparison / context marks
CARD_GREY = "F1F2F4"


def rgb(h):
    return RGBColor.from_string(h)


# ---------------------------------------------------------------- figures
for _f in Path("/usr/share/fonts/truetype/lato").glob("Lato-*.ttf"):  # template font
    font_manager.fontManager.addfont(str(_f))
plt.rcParams.update({
    "font.family": ["Lato", "DejaVu Sans"], "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#" + GREY, "axes.labelcolor": "#" + TEXT,
    "xtick.color": "#" + TEXT, "ytick.color": "#" + TEXT,
})
FIGSIZE = (8.6, 5.0)  # matches the template figure box aspect (7863840 x 4572000 EMU)


def bar_pair(ax, labels, base, ours, base_name, ours_name):
    x = range(len(labels))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], base, w, color="#" + CONTEXT, label=base_name)
    b2 = ax.bar([i + w / 2 for i in x], ours, w, color="#" + PURPLE, label=ours_name)
    for bars, col in ((b1, GREY), (b2, PURPLE)):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        color="#" + col, fontweight="bold")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 25)  # same 0-25 % scale on both accuracy slides; bars start at 0
    ax.set_ylabel("Strict sequence accuracy (%)")
    ax.yaxis.grid(True, color="#E3E5E9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")


def make_figures():
    FIG.mkdir(exist_ok=True)

    # Development folds 4-6 (07_results.tex, first paragraph)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bar_pair(ax, ["Signer 4", "Signer 5", "Signer 6"], [2.2, 2.3, 1.8], [18.1, 19.8, 16.9],
             "Initial autoregressive decoder", "imitator_e1")
    ax.set_xlabel("Development folds (held-out test signer)")
    fig.tight_layout()
    fig.savefig(FIG / "dev_accuracy.png", dpi=220)
    plt.close(fig)

    # Prospective confirmation folds 7-8 (results_macros.tex)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bar_pair(ax, ["Signer 7", "Signer 8"], [1.8, 2.7], [12.5, 14.1],
             "Historical affine CIF", "imitator_e1")
    ax.axhline(11.9, color="#" + TEAL, linestyle="--", linewidth=2, label="H2 threshold (11.9%)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-0.6, 1.6)
    ax.set_xlabel("Prospective confirmation folds (held-out test signer)")
    fig.tight_layout()
    fig.savefig(FIG / "confirmation_accuracy.png", dpi=220)
    plt.close(fig)

    # Gemma 3n correction audit (07_results.tex, Table gemma-tolerance)
    rates = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
    exact_after = [95.0, 0, 0, 0, 0, 0]
    chrf_before = [100.00, 36.67, 36.67, 36.67, 34.29, 23.30]
    chrf_after = [99.16, 34.58, 34.58, 34.58, 32.35, 21.85]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE)
    bars = a1.bar(rates, exact_after, color="#" + PURPLE, width=0.6)
    for b, v in zip(bars, exact_after):
        a1.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", color="#" + PURPLE, fontweight="bold")
    a1.set_ylim(0, 105)
    a1.set_title("Exact match after Gemma (%)", color="#" + TEXT, fontsize=15)
    a1.set_xlabel("Substitution rate")
    a2.plot(rates, chrf_before, "o-", color="#" + CONTEXT, linewidth=2.5, label="Before Gemma")
    a2.plot(rates, chrf_after, "o-", color="#" + PURPLE, linewidth=2.5, label="After Gemma")
    a2.set_ylim(0, 105)
    a2.set_title("chrF (0–100)", color="#" + TEXT, fontsize=15)
    a2.set_xlabel("Substitution rate")
    a2.legend(frameon=False, loc="upper right")
    for a in (a1, a2):
        a.yaxis.grid(True, color="#E3E5E9", linewidth=0.8)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "gemma_audit.png", dpi=220)
    plt.close(fig)

    # Paper figure for the backup slide
    subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile",
                    str(REPO / "springer_submission/figures/keypoint_interpretability.pdf"),
                    str(FIG / "keypoint_interpretability")], check=True)


# ---------------------------------------------------------------- pptx helpers
def fill_text(shape, paras, size=20, color=TEXT, bullets=False, space_after=10):
    """paras: list of str or (str, dict(bold,color,size)) ; '-' prefix -> bullet."""
    tf = shape.text_frame
    tf.word_wrap = True
    for i, p in enumerate(paras):
        text, opt = (p, {}) if isinstance(p, str) else p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        bullet = bullets or opt.get("bullet", False)
        pPr = para._p.get_or_add_pPr()
        for child in list(pPr):
            pPr.remove(child)
        if bullet:
            pPr.set("marL", "342900"), pPr.set("indent", "-342900")
            bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar",
                                 {"char": "•"})
            pPr.append(bu)
        else:
            pPr.set("marL", "0"), pPr.set("indent", "0")
            pPr.append(pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buNone", {}))
        para.space_after = Pt(opt.get("space_after", space_after))
        para.alignment = opt.get("align", PP_ALIGN.LEFT)
        run = para.add_run()
        run.text = text
        f = run.font
        f.size = Pt(opt.get("size", size))
        f.bold = opt.get("bold", False)
        f.italic = opt.get("italic", False)
        f.color.rgb = rgb(opt.get("color", color))


def ph(slide, idx):
    for s in slide.placeholders:
        if s.placeholder_format.idx == idx:
            return s
    raise KeyError(idx)


def title(slide, text, idx=103):
    fill_text(ph(slide, idx), [(text, {"bold": True, "size": 30, "color": PURPLE})], space_after=0)


def box(slide, x, y, w, h, fill=LT_TEAL, line=TEAL_LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1.25)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    tf = s.text_frame
    tf.margin_left = tf.margin_right = Inches(0.15)
    tf.margin_top = tf.margin_bottom = Inches(0.1)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return s


def arrow(slide, x1, y1, x2, y2, color=GREY):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(2.25)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd",
                             {"type": "triangle", "w": "med", "len": "med"}))
    return c


def picture_in(slide, idx, path):
    """Replace a layout picture placeholder by the image, fitted and centred in its box."""
    p = ph(slide, idx)
    x, y, w, h = p.left, p.top, p.width, p.height
    p._element.getparent().remove(p._element)
    pic = slide.shapes.add_picture(str(path), x, y)
    scale = min(w / pic.width, h / pic.height)
    pic.width, pic.height = int(pic.width * scale), int(pic.height * scale)
    pic.left, pic.top = x + (w - pic.width) // 2, y + (h - pic.height) // 2
    return pic


def drop(slide, idx):
    p = ph(slide, idx)
    p._element.getparent().remove(p._element)


def flow(slide, stages, y0, h, x0=0.6, total=12.1, gap=0.45, sub_size=15):
    """Row of boxes joined by arrows. stages: (head, sub, fill, line[, dashed]).
    Returns shapes in reveal order: box0, arrow1, box1, arrow2, box2, ..."""
    w = (total - gap * (len(stages) - 1)) / len(stages)
    seq = []
    for i, (head, sub, fill, line, *dashed) in enumerate(stages):
        b = box(slide, x0 + i * (w + gap), y0, w, h, fill, line)
        if dashed:
            b.line.dash_style, b.line.width = MSO_LINE.DASH, Pt(2.25)
        on_dark = fill == PURPLE
        fill_text(b, [(head, {"bold": True, "size": 18, "color": "FFFFFF" if on_dark else NAVY,
                              "align": PP_ALIGN.CENTER, "space_after": 6}),
                      (sub, {"size": sub_size, "color": "FFFFFF" if on_dark else TEXT, "align": PP_ALIGN.CENTER})])
        if i:
            xa = x0 + i * (w + gap)
            seq.append(arrow(slide, xa - gap + 0.06, y0 + h / 2, xa - 0.06, y0 + h / 2))
        seq.append(b)
    return seq


# ---------------------------------------------------------------- animations
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def title_intro(slide, prs, idx=103):
    """Centred copy of the slide title for the opening of a slide that starts empty.
    The real title stays at its layout position (so edit view and PDF are unchanged); see animate(intro=...)."""
    t = ph(slide, idx)
    run = t.text_frame.paragraphs[0].runs[0]
    c = slide.shapes.add_textbox(t.left, (prs.slide_height - t.height) // 2, t.width, t.height)
    c.name = "Title intro (animation)"
    c.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    c.text_frame.margin_left = c.text_frame.margin_right = 0
    fill_text(c, [(run.text, {"bold": True, "size": run.font.size.pt, "color": PURPLE,
                              "align": PP_ALIGN.CENTER})], space_after=0)
    return c, t


def animate(slide, clicks, intro=None):
    """Fade-in entrance animations. clicks: list of (targets, stagger_ms), one entry per click.
    A target is a shape, or (shape, first_par, last_par) to reveal only those paragraphs.
    Within a click, targets fade in one after another, `stagger_ms` apart.
    intro=(centred_copy, title) from title_intro(): on the first click the centred copy flies out
    upwards while the real title fades in at the top, then the first click's targets appear.
    Only fade and fly effects are used, because Google Slides drops motion paths."""
    ids = iter(range(3, 100000))
    build = {}
    FLY_MS = 500

    def fly_out_top(spid):
        tgt = f'<p:spTgt spid="{spid}"/>'
        anim = lambda attr, end: (
            f'<p:anim calcmode="lin" valueType="num"><p:cBhvr additive="base"><p:cTn id="{next(ids)}" '
            f'dur="{FLY_MS}"/><p:tgtEl>{tgt}</p:tgtEl><p:attrNameLst><p:attrName>{attr}</p:attrName>'
            f'</p:attrNameLst></p:cBhvr><p:tavLst><p:tav tm="0"><p:val><p:strVal val="{attr}"/></p:val>'
            f'</p:tav><p:tav tm="100000"><p:val><p:strVal val="{end}"/></p:val></p:tav></p:tavLst></p:anim>')
        return (f'<p:par><p:cTn id="{next(ids)}" presetID="2" presetClass="exit" presetSubtype="1" '
                f'fill="hold" grpId="0" nodeType="clickEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
                f'<p:childTnLst>{anim("ppt_x", "ppt_x")}{anim("ppt_y", "0-ppt_h/2")}<p:set><p:cBhvr>'
                f'<p:cTn id="{next(ids)}" dur="1" fill="hold"><p:stCondLst><p:cond delay="{FLY_MS - 1}"/>'
                f'</p:stCondLst></p:cTn><p:tgtEl>{tgt}</p:tgtEl><p:attrNameLst><p:attrName>style.visibility'
                f'</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val="hidden"/></p:to></p:set>'
                f'</p:childTnLst></p:cTn></p:par>')

    def effect(spid, pars, delay, node):
        tgt = (f'<p:spTgt spid="{spid}">'
               + (f'<p:txEl><p:pRg st="{pars[0]}" end="{pars[1]}"/></p:txEl>' if pars else "")
               + "</p:spTgt>")
        return (f'<p:par><p:cTn id="{next(ids)}" presetID="10" presetClass="entr" presetSubtype="0" '
                f'fill="hold" grpId="0" nodeType="{node}"><p:stCondLst><p:cond delay="{delay}"/></p:stCondLst>'
                f'<p:childTnLst><p:set><p:cBhvr><p:cTn id="{next(ids)}" dur="1" fill="hold"><p:stCondLst>'
                f'<p:cond delay="0"/></p:stCondLst></p:cTn><p:tgtEl>{tgt}</p:tgtEl><p:attrNameLst>'
                f'<p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to>'
                f'<p:strVal val="visible"/></p:to></p:set><p:animEffect transition="in" filter="fade">'
                f'<p:cBhvr><p:cTn id="{next(ids)}" dur="500"/><p:tgtEl>{tgt}</p:tgtEl></p:cBhvr>'
                f'</p:animEffect></p:childTnLst></p:cTn></p:par>')

    seq = []
    for c, (targets, stagger) in enumerate(clicks):
        effs, base = [], 0
        if intro and c == 0:
            copy_, title_ = intro
            effs.append(fly_out_top(copy_.shape_id))
            effs.append(effect(title_.shape_id, None, 200, "withEffect"))
            build[copy_.shape_id] = build[title_.shape_id] = 'animBg="1"'
            base = FLY_MS + 100
        for k, t in enumerate(targets):
            shp, pars = (t[0], t[1:]) if isinstance(t, tuple) else (t, None)
            node = "clickEffect" if k == 0 and not effs else "withEffect"
            effs.append(effect(shp.shape_id, pars, base + k * stagger, node))
            if shp.has_text_frame:  # shapes with text need a build entry; connectors/pictures don't
                build[shp.shape_id] = 'build="p"' if pars else 'animBg="1"'
        seq.append(f'<p:par><p:cTn id="{next(ids)}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/>'
                   f'</p:stCondLst><p:childTnLst><p:par><p:cTn id="{next(ids)}" fill="hold"><p:stCondLst>'
                   f'<p:cond delay="0"/></p:stCondLst><p:childTnLst>{"".join(effs)}</p:childTnLst></p:cTn>'
                   f'</p:par></p:childTnLst></p:cTn></p:par>')
    bld = "".join(f'<p:bldP spid="{i}" grpId="0" {a}/>' for i, a in build.items())
    xml = (f'<p:timing xmlns:p="{P_NS}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" '
           f'nodeType="tmRoot"><p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" '
           f'dur="indefinite" nodeType="mainSeq"><p:childTnLst>{"".join(seq)}</p:childTnLst></p:cTn>'
           f'<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>'
           f'</p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl>'
           f'</p:cond></p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst>'
           f'<p:bldLst>{bld}</p:bldLst></p:timing>')
    slide._element.append(etree.fromstring(xml))


# ---------------------------------------------------------------- slides
def build():
    prs = Presentation(str(TEMPLATE))
    L = {l.name: l for l in prs.slide_layouts}
    sldnum = copy.deepcopy([s for s in prs.slides[2].shapes if s.name.startswith("Slide Number")][0]._element)

    # remove the template's example slides
    lst = prs.slides._sldIdLst
    for sid in list(lst):
        prs.part.drop_rel(sid.rId)
        lst.remove(sid)

    def new(layout, notes, number=True):
        s = prs.slides.add_slide(L[layout])
        if number:
            s.shapes._spTree.append(copy.deepcopy(sldnum))
        s.notes_slide.notes_text_frame.text = notes
        return s

    # 1 · Title ---------------------------------------------------------------
    s = new("WAIMLAp Title", NOTES[1], number=False)
    fill_text(ph(s, 106), [("Imitator: Direct Prediction of Gemma Token IDs for "
                            "Language-Model-Assisted Sign Transcription",
                            {"bold": True, "size": 36, "color": PURPLE})], space_after=0)
    fill_text(ph(s, 107), [("Christian Velasquez Borasino, Giorgio Mancusi Barreda, Rody Vilchez Marin",
                            {"bold": True, "size": 20})])
    fill_text(ph(s, 108), [("Universidad Peruana de Ciencias Aplicadas (UPC), Lima, Peru",
                            {"size": 16, "color": TEAL})])
    fill_text(ph(s, 105), [("Oral Session 10  ·  Paper ID 58",
                            {"bold": True, "size": 14, "color": TEAL, "align": PP_ALIGN.RIGHT})])

    # 2 · Context: the interface problem ---------------------------------------
    s = new("WAIMLAp Content", NOTES["context"])
    title(s, "Sign-to-text needs a bridge between vision and language")
    drop(s, 104)
    f = flow(s, [
        ("See the signs", "motion → linguistic evidence\n(video → 2D keypoints)", CARD_GREY, None),
        ("Interface ?", "what should the visual model\nhand to the language model?", "FFFFFF", PURPLE, True),
        ("Write the language", "pretrained LLMs already\nturn tokens into fluent text", CARD_GREY, None),
    ], y0=2.0, h=1.9, gap=0.9)
    t = s.shapes.add_textbox(Inches(0.6), Inches(4.4), Inches(12.1), Inches(2.0))
    fill_text(t, [
        ("Bridges explored so far", {"bold": True, "size": 20, "color": PURPLE, "space_after": 8}),
        ("Pseudo-gloss pretraining (Sign2GPT) · hierarchical sign tokens (SignLLM)", {"bullet": True}),
        ("Separate visual and language learning (FLa-LLM)", {"bullet": True}),
        ("In speech: continuous prompts and CTC token embeddings (Wav2Prompt, LegoSLM)", {"bullet": True}),
    ], size=18, space_after=6)
    animate(s, [([f[0]], 0), (f[1:3], 250), (f[3:5], 250), ([t], 0)], intro=title_intro(s, prs))

    # 3 · History: the original Imitator ---------------------------------------
    s = new("WAIMLAp Content", NOTES["history"])
    title(s, "Our first Imitator imitated the LLM's input embeddings")
    drop(s, 104)
    f = flow(s, [
        ("2D keypoints", "sign video", CARD_GREY, None),
        ("Small visual model", "lightweight\nST-GCN encoder", LT_TEAL, TEAL_LINE),
        ("Vectors ≈ embeddings", "close to the frozen\nLLM's embeddings", PURPLE, None),
        ("Frozen LLM", "reads the vectors\nas its input", CARD_GREY, None),
    ], y0=2.0, h=1.9)
    t = s.shapes.add_textbox(Inches(0.6), Inches(4.4), Inches(12.1), Inches(2.0))
    fill_text(t, [
        ("Premise: a compact model learns the input language of a frozen LLM", {"bullet": True}),
        ("Trained with MSE + cosine distance; reported loss ≈ 8×10⁻⁴ (preliminary)", {"bullet": True}),
        ("Open question: does the LLM read those vectors as the intended words?",
         {"bullet": True, "bold": True, "color": PURPLE}),
    ], size=20, space_after=10)
    animate(s, [(f[0:3], 300), (f[3:5], 250), (f[5:7], 250), ([(t, 0, 1)], 0), ([(t, 2, 2)], 0)],
            intro=title_intro(s, prs))

    # 4 · Turning point --------------------------------------------------------
    s = new("WAIMLAp Two Columns", NOTES[2])
    title(s, "Close to the right embedding is not the right token")
    for idx, fill, line, head, head_col, lines in (
        (104, CARD_GREY, None, "Limits of embedding imitation", GREY,
         ["A nearby vector may not be the intended token, especially in anisotropic spaces",
          "No exact way to check what the LLM receives",
          "Splits were not signer-exclusive"]),
        (105, LT_TEAL, TEAL_LINE, "This work: predict the token IDs", PURPLE,
         ["Keypoints → exact Gemma tokenizer IDs + EOS",
          "Correct only if every ID matches: measurable, falsifiable",
          "Gemma 3n receives ordinary text tokens"]),
    ):
        p = ph(s, idx)
        p.fill.solid()
        p.fill.fore_color.rgb = rgb(fill)
        if line:
            p.line.color.rgb = rgb(line)
            p.line.width = Pt(1.25)
        tf = p.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.3)
        fill_text(p, [(head, {"bold": True, "size": 22, "color": head_col, "space_after": 16})]
                  + [(t, {"bullet": True}) for t in lines], size=20, space_after=14)
    animate(s, [([ph(s, 105)], 0)])

    # 3 · Pipeline ------------------------------------------------------------
    s = new("WAIMLAp Content", NOTES[3])
    title(s, "Imitator writes exact Gemma IDs; Gemma only reads text")
    drop(s, 104)
    stages = [
        ("2D keypoints", "111 per frame\n(pose, face, hands)", CARD_GREY, None),
        ("Visual encoder", "ST-GCN + temporal conv\n+ Transformer\n+ positional encoding", LT_TEAL, TEAL_LINE),
        ("Token decoder", "2-layer causal\nTransformer\n121-row softmax", LT_TEAL, TEAL_LINE),
        ("Exact Gemma IDs", "content IDs\n+ EOS", PURPLE, None),
        ("Gemma 3n", "E2B, frozen;\nreads IDs as text", CARD_GREY, None),
    ]
    x0, y0, w, h, gap = 0.6, 2.35, 2.05, 1.9, 0.45
    f = flow(s, stages, y0, h, x0=x0, total=5 * w + 4 * gap, gap=gap, sub_size=14)
    # brackets: what is scored where
    brackets = []
    for xs, xe, label, col in ((x0, x0 + 4 * w + 3 * gap, "Imitator · scored on its own: strict accuracy, edit similarity", PURPLE),
                               (x0 + 4 * (w + gap), x0 + 5 * w + 4 * gap, "Audited separately", GREY)):
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(xs), Inches(y0 - 0.25), Inches(xe), Inches(y0 - 0.25))
        ln.line.color.rgb, ln.line.width = rgb(col), Pt(2)
        t = s.shapes.add_textbox(Inches(xs), Inches(y0 - 0.72), Inches(xe - xs), Inches(0.4))
        fill_text(t, [(label, {"bold": True, "size": 15, "color": col, "align": PP_ALIGN.CENTER})])
        brackets += [ln, t]
    t = s.shapes.add_textbox(Inches(0.6), Inches(4.5), Inches(12.1), Inches(1.6))
    fill_text(t, ["No 64-class gloss classifier at inference (targets still come from LSA64 labels)",
                  "121 output rows map one-to-one back to the original Gemma IDs",
                  "Transcription is scored before the LLM, so fluent output cannot hide visual errors"],
              size=18, bullets=True, space_after=8)
    animate(s, [(f, 250), (brackets, 0), ([t], 0)], intro=title_intro(s, prs))

    # 4 · Protocol ------------------------------------------------------------
    s = new("WAIMLAp Two Columns", NOTES[4])
    title(s, "Test signers are unseen; confirmation was preregistered")
    fill_text(ph(s, 104), [
        ("Data: LSA64", {"bold": True, "size": 22, "color": PURPLE, "space_after": 12}),
        ("3,200 videos · 64 signs · 10 signers", {"bullet": True}),
        ("Streams of 2–8 isolated clips, random order", {"bullet": True}),
        ("Target: Gemma tokens of the Spanish labels", {"bullet": True}),
        ("Random order blocks a language shortcut — but there is no syntax to exploit either",
         {"bullet": True, "color": GREY}),
    ], size=20, space_after=12)
    drop(s, 105)
    xr = 6.7
    n0 = len(s.shapes)
    hd = s.shapes.add_textbox(Inches(xr), Inches(1.6), Inches(6), Inches(0.5))
    fill_text(hd, [("Leave-one-signer-out folds", {"bold": True, "size": 22, "color": PURPLE})])
    sq, gp = 0.5, 0.08
    for k in range(10):
        fill, line, fc = ((CARD_GREY, None, TEXT) if k < 6 else
                          (PURPLE, None, "FFFFFF") if k < 8 else ("FFFFFF", CONTEXT, GREY))
        b = box(s, xr + k * (sq + gp), 2.3, sq, sq, fill, line, shape=MSO_SHAPE.RECTANGLE)
        b.text_frame.margin_left = b.text_frame.margin_right = 0
        fill_text(b, [(str(k + 1), {"bold": True, "size": 16, "color": fc, "align": PP_ALIGN.CENTER})], space_after=0)
    for x, wdt, lab, col in ((xr, 6 * sq + 5 * gp, "development", GREY),
                             (xr + 6 * (sq + gp), 2 * sq + gp, "confirmation", PURPLE),
                             (xr + 8 * (sq + gp), 2 * sq + gp, "reserved", GREY)):
        t = s.shapes.add_textbox(Inches(x - 0.3), Inches(2.85), Inches(wdt + 0.6), Inches(0.35))
        fill_text(t, [(lab, {"size": 14, "bold": col == PURPLE, "color": col, "align": PP_ALIGN.CENTER})])
    folds = list(s.shapes)[n0:]
    t = s.shapes.add_textbox(Inches(xr), Inches(3.45), Inches(6), Inches(2.6))
    fill_text(t, [
        ("Preregistered, on each confirmation signer:", {"bold": True, "size": 18, "color": NAVY}),
        ("H1  beat historical CIF (sign test p < .01)", {"bullet": True}),
        ("H2  strict accuracy ≥ 11.9%", {"bullet": True}),
        ("H3  label order ≥ .90 and permutation drop ≥ .15", {"bullet": True}),
    ], size=18, space_after=8)
    animate(s, [(folds, 0), ([t], 0)])

    # 5 · Result: development -------------------------------------------------
    s = new("WAIMLAp Figure", NOTES[5])
    title(s, "Token IDs raise strict accuracy ~10× on development signers")
    picture_in(s, 104, FIG / "dev_accuracy.png")
    fill_text(ph(s, 105), [
        ("Key result", {"bold": True, "size": 20, "color": TEAL}),
        ("16.9–19.8%", {"bold": True, "size": 32, "color": PURPLE, "space_after": 2}),
        ("strict accuracy", {"size": 18, "space_after": 2}),
        ("(initial decoder: 1.8–2.3%)", {"size": 18, "space_after": 18}),
        ("Positional encoding: label order goes from chance to ≈ 0.99", {"size": 18, "space_after": 18}),
        ("Viable, not solved: most sequences still contain ≥ 1 error", {"size": 18, "color": GREY}),
    ])
    side = ph(s, 105)
    animate(s, [([(side, 0, 3)], 0), ([(side, 4, 4)], 0), ([(side, 5, 5)], 0)])

    # 6 · Result: confirmation ------------------------------------------------
    s = new("WAIMLAp Figure", NOTES[6])
    title(s, "Both held-out signers pass all three preregistered criteria")
    picture_in(s, 104, FIG / "confirmation_accuracy.png")
    fill_text(ph(s, 105), [
        ("H1 ✓  beats CIF", {"bold": True, "size": 20, "color": PURPLE, "space_after": 2}),
        ("wins/losses 99/3 and 108/6", {"size": 16, "space_after": 14}),
        ("H2 ✓  ≥ 11.9%", {"bold": True, "size": 20, "color": PURPLE, "space_after": 2}),
        ("12.5% and 14.1% strict accuracy", {"size": 16, "space_after": 14}),
        ("H3 ✓  visually ordered", {"bold": True, "size": 20, "color": PURPLE, "space_after": 2}),
        ("label order 0.98–0.99; shuffling segments drops edit similarity by 0.34–0.38",
         {"size": 16}),
    ])
    side = ph(s, 105)
    animate(s, [([(side, 0, 1)], 0), ([(side, 2, 3)], 0), ([(side, 4, 5)], 0)])

    # 7 · Result: Gemma audit -------------------------------------------------
    s = new("WAIMLAp Figure", NOTES[7])
    title(s, "Without sentence context, Gemma cannot repair substitutions")
    picture_in(s, 104, FIG / "gemma_audit.png")
    fill_text(ph(s, 105), [
        ("Negative result", {"bold": True, "size": 20, "color": TEAL}),
        ("Clean input: 95% kept exact", {"size": 18, "space_after": 12}),
        ("Any substitution: 0% exact, before and after Gemma", {"bold": True, "size": 18, "color": PURPLE,
                                                               "space_after": 12}),
        ("chrF drops 0.8–2.1 points", {"size": 18, "space_after": 12}),
        ("“photo” → “green”: isolated labels leave nothing to infer from", {"size": 18, "color": GREY}),
    ])
    side = ph(s, 105)
    animate(s, [([(side, 0, 1)], 0), ([(side, 2, 2)], 0), ([(side, 3, 4)], 0)])

    # 8 · Conclusions ---------------------------------------------------------
    s = new("WAIMLAp Two Columns", NOTES[8])
    title(s, "Exact IDs make the bridge measurable; correction needs context")
    for idx, fill, line, head, lines in (
        (104, LT_TEAL, TEAL_LINE, "What we showed",
         ["Exact Gemma IDs replace approximate embeddings with a checkable contract",
          "Unseen signers: 12.5–14.1% strict accuracy, ordered by the video",
          "Isolated labels give Gemma nothing to correct"]),
        (105, CARD_GREY, None, "Limits → next step",
         ["One seed (23), synthetic LSA64 streams, two confirmation signers",
          "Next: preregistered sentence corpus with real context",
          "Pass top-k / posterior alternatives, still scoring the transcript"]),
    ):
        p = ph(s, idx)
        p.fill.solid()
        p.fill.fore_color.rgb = rgb(fill)
        if line:
            p.line.color.rgb, p.line.width = rgb(line), Pt(1.25)
        tf = p.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = Inches(0.3)
        fill_text(p, [(head, {"bold": True, "size": 22, "color": PURPLE if fill == LT_TEAL else GREY,
                              "space_after": 16})]
                  + [(t, {"bullet": True}) for t in lines], size=20, space_after=14)
    animate(s, [([ph(s, 105)], 0)])

    # 9 · Closing -------------------------------------------------------------
    s = new("WAIMLAp Closing", NOTES[9], number=False)
    fill_text(ph(s, 105), [("Thank you · Questions?", {"bold": True, "size": 54, "color": "FFFFFF"})])
    fill_text(ph(s, 106), [
        ("Christian Velasquez Borasino", {"bold": True, "size": 20, "color": "FFFFFF"}),
        ("Universidad Peruana de Ciencias Aplicadas (UPC)  ·  Paper ID 58", {"size": 18, "color": LAVENDER}),
    ])

    # 10 · Backup -------------------------------------------------------------
    s = new("WAIMLAp Figure", NOTES[10])
    title(s, "Backup · Left hand drives the central subtokens (one example)")
    picture_in(s, 104, FIG / "keypoint_interpretability.png")
    fill_text(ph(s, 105), [
        ("Signer 7, “A tierra / Perfume”", {"bold": True, "size": 18, "color": TEAL}),
        ("Left hand: 28.9–41.6% of attribution", {"size": 18, "space_after": 12}),
        ("Largest ablation loss for “a” and “perf” (7.92, 15.77 log-prob)", {"size": 18, "space_after": 12}),
        ("One example: descriptive, not population-level", {"size": 18, "color": GREY}),
    ])

    prs.save(str(OUT))
    print("wrote", OUT)


# ---------------------------------------------------------------- speaker notes (≈10 min)
NOTES = {
    1: "Good morning. I am Christian Velasquez, from UPC, presenting joint work with Giorgio Mancusi "
       "and Rody Vilchez. The question behind this paper is simple: how should a small sign-language "
       "model talk to a large language model? Let me start with why that is the hard part.",
    "context": "Turning signing into text needs two skills. First, seeing: extracting linguistic evidence "
       "from motion. Second, writing: turning that evidence into language. Large language models are "
       "already very good at the second part. The open question is the bridge: what exactly should a "
       "visual model hand to the language model? Several answers exist: pseudo-glosses as in Sign2GPT, "
       "hierarchical sign tokens as in SignLLM, or keeping visual and language learning separate. In "
       "speech, systems pass continuous prompts or token embeddings. [pause] Our own first answer was this.",
    "history": "In our earlier, published Imitator, a small visual model read 2D keypoints and produced "
       "continuous vectors meant to sit close to the frozen LLM's own input embeddings. The idea is that a "
       "compact model can learn the input language of a large one. Training used mean-squared and cosine "
       "distance, and the preliminary loss was very low, about 8 times 10 to the minus 4. But a low loss "
       "leaves one question open: does the LLM actually read those vectors as the intended words?",
    2: "And the answer is: not necessarily. Being geometrically close does not guarantee the intended "
       "token, especially because these embedding spaces are anisotropic. There was also no exact way to "
       "check what the LLM received, and the splits were not signer-exclusive. So in this work we keep the "
       "premise, but we predict the exact Gemma token IDs. A prediction is right only if every ID matches, "
       "and Gemma receives ordinary text. [pause] Let me show the pipeline.",
    3: "From 2D keypoints, 111 per frame, a spatial-temporal graph encoder with positional encoding builds "
       "visual memory. A small causal Transformer decoder emits Gemma IDs until end-of-sequence. The task "
       "uses 118 Gemma content IDs, so the output layer has 121 rows that map one-to-one back to the real "
       "IDs. Important: we score this transcription on its own, before Gemma, so a fluent LLM cannot hide "
       "a visual error. Gemma 3n is frozen and audited separately. Next, how we evaluated it.",
    4: "We use LSA64: 64 Argentinian signs, ten signers. We build streams of two to eight clips in random "
       "order, and the target is the Gemma tokenization of their Spanish labels. Random order prevents a "
       "language shortcut, but it also means there is no syntax. Every fold leaves one signer out. Folds 1 "
       "to 6 were for development, 7 and 8 were held back for confirmation, and before testing them we "
       "registered three criteria: beat the historical CIF model, reach 11.9 percent strict accuracy, and "
       "show that output order follows the video. First, development.",
    5: "On the development signers, strict accuracy, meaning the whole sequence exactly right, goes from about "
       "2 percent with our initial autoregressive decoder to 17 to 20 percent. Roughly ten times. Adding "
       "positional encoding moved pairwise label order from chance to about 0.99. But to be clear: most "
       "sequences still have at least one error. It is viable, not solved. [pause] Does it hold on new signers?",
    6: "On the two signers we had never touched, imitator_e1 reaches 12.5 and 14.1 percent, above the "
       "pre-registered 11.9 threshold, against 1.8 and 2.7 for CIF. Paired, it wins 99 times and loses 3 on "
       "signer 7, and 108 to 6 on signer 8. Edit similarity is 53 and 59 percent. And if we shuffle the "
       "sign segments while keeping the target, edit similarity drops by about 0.35: the order really comes "
       "from the video. All three criteria pass. Now the second half: can Gemma fix the remaining errors?",
    7: "We corrupted gold token sequences by replacing whole labels and asked Gemma 3n to correct them, "
       "120 rows per rate. With clean input Gemma keeps 95 percent exact. With any substitution, exact "
       "match is zero before and zero after, and chrF actually goes down slightly. The reason is "
       "structural: if 'photo' became 'green' in a list of random signs, nothing tells you it was 'photo'. "
       "So this is not evidence that LLMs cannot correct signs; it shows our data has no context to correct with.",
    8: "To summarise: predicting exact token IDs turns the bridge into something we can measure. On unseen "
       "signers it gives 12 to 14 percent fully correct, visually ordered sequences. And correction needs "
       "context that isolated labels do not have. Limits: one training seed, synthetic streams and two "
       "confirmation signers. Next, we want a pre-registered sentence corpus, and passing top-k "
       "alternatives to the LLM while still scoring the transcript.",
    9: "Thank you. I am happy to take questions.",
    10: "Backup for questions on interpretability. For one unseen-signer sequence, input-times-gradient and "
        "region ablation both point to the left hand for the central subtokens. It is a single example, so "
        "it is descriptive, not a dataset-wide claim.",
}

if __name__ == "__main__":
    make_figures()
    build()
