"""Build the WAIMLAp 2026 talk for the Imitator paper from the official template.

Follows the draft in paper/content/full, including the exploratory
isolated-sign results from the retrained folds 7–8.

Run:  ~/miniconda3/envs/deck-tools/bin/python build_deck.py
"""
import copy
import subprocess
from pathlib import Path

import matplotlib
import segno

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

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
LINKEDIN = "https://www.linkedin.com/in/christian-velasquez-borasino/"


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
             "Our initial decoder (no positional encoding)", "Imitator (this work)")
    ax.set_xlabel("Development folds (held-out test signer)")
    fig.tight_layout()
    fig.savefig(FIG / "dev_accuracy.png", dpi=220)
    plt.close(fig)

    # Prospective confirmation folds 7-8 (results_macros.tex)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bar_pair(ax, ["Signer 7", "Signer 8"], [1.8, 2.7], [12.5, 14.1],
             "Our earlier model (CIF)", "Imitator (this work)")
    ax.axhline(11.9, color="#" + TEAL, linestyle="--", linewidth=2, label="Pass mark fixed in advance (11.9%)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-0.6, 1.6)
    ax.set_xlabel("Prospective confirmation folds (held-out test signer)")
    fig.tight_layout()
    fig.savefig(FIG / "confirmation_accuracy.png", dpi=220)
    plt.close(fig)

    # Same diagnostic on both replicas: the complete reference is an output prefix.
    # Percentile intervals: 10,000 bootstrap resamples of 64 glosses, seed 23.
    values = [71.5625, 68.75]
    bounds = [(60.9375, 81.2578125), (58.75, 78.75)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(["Signer 7", "Signer 8"], values, width=0.48, color="#" + PURPLE,
                  yerr=[[v - lo for v, (lo, _) in zip(values, bounds)],
                        [hi - v for v, (_, hi) in zip(values, bounds)]],
                  error_kw={"ecolor": "#" + NAVY, "capsize": 7, "elinewidth": 2, "capthick": 2})
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, 45, f"{value:.1f}%", ha="center",
                color="white", fontweight="bold", fontsize=22)
    ax.set(ylim=(0, 100), ylabel="Output starts with the right label (%)",
           xlabel="Error bars: 95% bootstrap intervals over the 64 signs")
    ax.yaxis.grid(True, color="#E3E5E9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "isolated_prefix.png", dpi=220)
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
    a1.set_xlabel("Nominal substitution rate")
    a2.plot(rates, chrf_before, "o-", color="#" + CONTEXT, linewidth=2.5, label="Before Gemma")
    a2.plot(rates, chrf_after, "o-", color="#" + PURPLE, linewidth=2.5, label="After Gemma")
    a2.set_ylim(0, 105)
    a2.set_title("chrF (0–100)", color="#" + TEXT, fontsize=15)
    a2.set_xlabel("Nominal substitution rate")
    a2.legend(frameon=False, loc="upper right")
    for a in (a1, a2):
        a.yaxis.grid(True, color="#E3E5E9", linewidth=0.8)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "gemma_audit.png", dpi=220)
    plt.close(fig)

    # QR for the closing slide (white quiet zone so it scans on the dark background)
    segno.make(LINKEDIN, error="m").save(str(FIG / "linkedin_qr.png"), scale=20, border=3)

    # Paper figure for the backup slide
    subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile",
                    str(REPO / "paper/figures/keypoint_interpretability.pdf"),
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
    """Place the sole title in the centre and record how far it must move up."""
    t = ph(slide, idx)
    top = t.top
    t.left = (prs.slide_width - t.width) // 2
    t.top = (prs.slide_height - t.height) // 2
    t.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    return t, (top - t.top) / prs.slide_height


def animate(slide, clicks, intro=None):
    """Fade-in entrance animations. clicks: list of (targets, stagger_ms), one entry per click.
    A target is a shape, or (shape, first_par, last_par) to reveal only those paragraphs.
    Within a click, targets fade in one after another, `stagger_ms` apart.
    intro=(title, dy) moves the single title vertically on the first click.
    The first content reveal starts after the title reaches the header."""
    ids = iter(range(3, 100000))
    build = {}
    MOVE_MS = 1000

    def glide(spid, dy):
        tgt = f'<p:spTgt spid="{spid}"/>'
        return (f'<p:par><p:cTn id="{next(ids)}" presetID="0" presetClass="path" presetSubtype="0" '
                f'accel="50000" decel="50000" fill="hold" grpId="0" nodeType="clickEffect"><p:stCondLst>'
                f'<p:cond delay="0"/></p:stCondLst><p:childTnLst><p:animMotion origin="layout" '
                f'path="M 0 0 L 0 {dy:.5f} E" pathEditMode="relative" ptsTypes="AA"><p:cBhvr>'
                f'<p:cTn id="{next(ids)}" dur="{MOVE_MS}" fill="hold"/><p:tgtEl>{tgt}</p:tgtEl>'
                f'<p:attrNameLst><p:attrName>ppt_x</p:attrName><p:attrName>ppt_y</p:attrName>'
                f'</p:attrNameLst></p:cBhvr><p:rCtr x="0" y="0"/></p:animMotion></p:childTnLst>'
                f'</p:cTn></p:par>')

    def effect(spid, pars, delay, node, dur=500):
        tgt = (f'<p:spTgt spid="{spid}">'
               + (f'<p:txEl><p:pRg st="{pars[0]}" end="{pars[1]}"/></p:txEl>' if pars else "")
               + "</p:spTgt>")
        return (f'<p:par><p:cTn id="{next(ids)}" presetID="10" presetClass="entr" presetSubtype="0" '
                f'fill="hold" grpId="0" nodeType="{node}"><p:stCondLst><p:cond delay="{delay}"/></p:stCondLst>'
                f'<p:childTnLst><p:set><p:cBhvr><p:cTn id="{next(ids)}" dur="1" fill="hold"><p:stCondLst>'
                f'<p:cond delay="0"/></p:stCondLst></p:cTn><p:tgtEl>{tgt}</p:tgtEl><p:attrNameLst>'
                f'<p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to>'
                f'<p:strVal val="visible"/></p:to></p:set><p:animEffect transition="in" filter="fade">'
                f'<p:cBhvr><p:cTn id="{next(ids)}" dur="{dur}"/><p:tgtEl>{tgt}</p:tgtEl></p:cBhvr>'
                f'</p:animEffect></p:childTnLst></p:cTn></p:par>')

    seq = []
    for c, (targets, stagger) in enumerate(clicks):
        effs, base = [], 0
        if intro and c == 0:
            title_, dy = intro
            effs.append(glide(title_.shape_id, dy))
            base = MOVE_MS
        # One effect per paragraph: a multi-paragraph range hides only its first paragraph
        # in some viewers, which left a bullet on screen before its click.
        targets = [x for t in targets for x in
                   ([(t[0], i, i) for i in range(t[1], t[2] + 1)] if isinstance(t, tuple) else [t])]
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
    old = slide._element.find(f"{{{P_NS}}}timing")  # add_movie() already wrote a timing tree
    videos = [] if old is None else old.findall(f".//{{{P_NS}}}video")
    if old is not None:
        slide._element.remove(old)
    xml = (f'<p:timing xmlns:p="{P_NS}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" '
           f'nodeType="tmRoot"><p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" '
           f'dur="indefinite" nodeType="mainSeq"><p:childTnLst>{"".join(seq)}</p:childTnLst></p:cTn>'
           f'<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>'
           f'</p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl>'
           f'</p:cond></p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst>'
           f'<p:bldLst>{bld}</p:bldLst></p:timing>')
    timing = etree.fromstring(xml)
    for v in videos:  # click the clip to play; it loops until the slide advances
        v.find(f".//{{{P_NS}}}cTn").attrib.update({"id": str(next(ids)), "repeatCount": "indefinite"})
        timing.find(f".//{{{P_NS}}}childTnLst").append(v)
    slide._element.append(timing)


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

    def new(layout, notes, number=True, section=None):
        s = prs.slides.add_slide(L[layout])
        if section:  # small tag above the title: which part of the talk we are in
            tag = s.shapes.add_textbox(Inches(0.6), Inches(0.14), Inches(6), Inches(0.32))
            fill_text(tag, [(section.upper(), {"bold": True, "size": 13, "color": TEAL})], space_after=0)
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

    # 2 · Hook: a sign the audience cannot read --------------------------------
    s = new("WAIMLAp Content", NOTES["hook"], section="Motivation")
    title(s, "What did this person just say?")
    drop(s, 104)
    vw, vh = Inches(6.6), Inches(6.6 * 9 / 16)
    s.shapes.add_movie(str(FIG / "hook_sign.mp4"), Inches(0.6), Inches(1.7), vw, vh,
                       poster_frame_image=str(FIG / "hook_sign.png"), mime_type="video/mp4")
    cap = s.shapes.add_textbox(Inches(0.6), Inches(5.5), Inches(6.6), Inches(0.4))
    fill_text(cap, [("Clip: LSA64 dataset (Ronchetti et al., 2016) · click to play",
                     {"size": 13, "color": GREY})])
    t = s.shapes.add_textbox(Inches(7.6), Inches(1.9), Inches(5.1), Inches(3.6))
    fill_text(t, [
        ("Have you ever asked an AI model to describe a video?", {"size": 24, "space_after": 26}),
        ("Could it understand this one?", {"size": 24, "bold": True, "color": PURPLE, "space_after": 26}),
        ("“Perfume”", {"size": 30, "bold": True, "color": NAVY, "space_after": 2}),
        ("in Argentinian Sign Language", {"size": 20, "color": GREY}),
    ])
    animate(s, [([(t, 1, 1)], 0), ([(t, 2, 3)], 0)])

    # 3 · The LLM knows language; do we need a translator from scratch? ---------
    s = new("WAIMLAp Content", NOTES["context"], section="Motivation")
    title(s, "The LLM knows language. It does not know sign language.")
    drop(s, 104)
    f = flow(s, [
        ("A signer moves", "video → body, face\nand hand keypoints", CARD_GREY, None),
        ("A small adapter?", "learns to write in\nthe LLM’s input format", LT_TEAL, TEAL_LINE),
        ("The LLM writes", "fluent Spanish;\nit stays frozen", CARD_GREY, None),
    ], y0=2.0, h=1.9, gap=0.9)
    q = s.shapes.add_textbox(Inches(0.6), Inches(4.5), Inches(12.1), Inches(0.8))
    fill_text(q, [("Do we have to train a sign translator from scratch?",
                   {"bold": True, "size": 24, "color": PURPLE, "align": PP_ALIGN.CENTER})])
    animate(s, [([f[0]], 0), ([f[4]], 0), ([q], 0), ([f[1], f[3], f[2]], 250)], intro=title_intro(s, prs))

    # 4 · Previous work: embeddings; this work: token IDs ----------------------
    s = new("WAIMLAp Content", NOTES["history"], section="Previous work")
    title(s, "Before: imitate embeddings. Now: write token IDs.")
    drop(s, 104)
    x0, gap, total, y0, h = 0.6, 0.45, 12.1, 1.7, 1.3
    f = flow(s, [
        ("Text", "“perfume”", CARD_GREY, None),
        ("Token IDs", "whole numbers,\none per word piece", CARD_GREY, None),
        ("Embeddings", "one vector\nper token ID", CARD_GREY, None),
        ("Frozen LLM", "writes the text", CARD_GREY, None),
    ], y0=y0, h=h, x0=x0, total=total, gap=gap)
    w = (total - 3 * gap) / 4
    cards = []
    for col, x, fill, line, head, head_col, lines in (
        (2, 6.8, CARD_GREY, None, "Previous work: first Imitator (2025)", GREY,
         ["Vectors close to a frozen LLM’s embeddings", "Loss ≈ 8×10⁻⁴ (preliminary)",
          "Open: is a close vector the intended word? No exact check."]),
        (1, 0.6, LT_TEAL, TEAL_LINE, "This work: write the token IDs", PURPLE,
         ["Exact Gemma token IDs", "Each ID is right or wrong",
          "Scored before the LLM reads anything"]),
    ):
        c = box(s, x, 3.75, 5.9, 2.6, fill, line)
        c.text_frame.vertical_anchor = MSO_ANCHOR.TOP
        fill_text(c, [(head, {"bold": True, "size": 20, "color": head_col, "space_after": 10})]
                  + [(t, {"bullet": True}) for t in lines], size=18, space_after=8)
        xa = x0 + col * (w + gap) + w / 2
        cards.append([arrow(s, xa, 3.69, xa, y0 + h + 0.06, PURPLE if col == 1 else GREY), c])
    animate(s, [(f, 200), (cards[0], 0), (cards[1], 0)], intro=title_intro(s, prs))

    # 6 · Pipeline ------------------------------------------------------------
    s = new("WAIMLAp Content", NOTES[3], section="Methodology · model")
    title(s, "Imitator now writes Gemma’s own token IDs")
    drop(s, 104)
    stages = [
        ("Sign video", "111 keypoints per frame\n(pose, face, hands)", CARD_GREY, None),
        ("Encoder", "reads the movement\nand keeps the order", LT_TEAL, TEAL_LINE),
        ("Decoder", "writes one token ID\nat a time, then stops", LT_TEAL, TEAL_LINE),
        ("Gemma token IDs", "the label’s IDs\n+ end marker", PURPLE, None),
        ("Gemma 3n", "frozen LLM; reads\nthe IDs as text", CARD_GREY, None),
    ]
    x0, y0, w, h, gap = 0.6, 2.35, 2.05, 1.9, 0.45
    f = flow(s, stages, y0, h, x0=x0, total=5 * w + 4 * gap, gap=gap, sub_size=14)
    brackets = []
    for xs, xe, label, col in ((x0, x0 + 4 * w + 3 * gap, "Imitator · scored on its own: every ID must match", PURPLE),
                               (x0 + 4 * (w + gap), x0 + 5 * w + 4 * gap, "Audited separately", GREY)):
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(xs), Inches(y0 - 0.25), Inches(xe), Inches(y0 - 0.25))
        ln.line.color.rgb, ln.line.width = rgb(col), Pt(2)
        t = s.shapes.add_textbox(Inches(xs), Inches(y0 - 0.72), Inches(xe - xs), Inches(0.4))
        fill_text(t, [(label, {"bold": True, "size": 15, "color": col, "align": PP_ALIGN.CENTER})])
        brackets += [ln, t]
    t = s.shapes.add_textbox(Inches(0.6), Inches(4.6), Inches(12.1), Inches(1.2))
    fill_text(t, [
        ("We score the draft before any LLM touches it, so fluent text cannot hide a visual error.",
         {"size": 20}),
        ("In this paper Gemma is tested apart, on labels with synthetic errors. "
         "It has not yet read Imitator’s own drafts.", {"size": 18, "color": GREY}),
    ], size=18)
    animate(s, [(f, 250), (brackets, 0), ([t], 0)], intro=title_intro(s, prs))

    # 7 · Protocol ------------------------------------------------------------
    s = new("WAIMLAp Two Columns", NOTES[4], section="Methodology · evaluation")
    title(s, "One sign first. Then several signs in order.")
    fill_text(ph(s, 104), [
        ("Data: LSA64", {"bold": True, "size": 22, "color": PURPLE, "space_after": 12}),
        ("3,200 videos · 64 signs · 10 signers", {"bullet": True}),
        ("One sign: one clip → its Spanish label", {"bullet": True, "bold": True}),
        ("Several signs: 2–8 clips joined in random order → the labels, in order",
         {"bullet": True, "bold": True}),
        ("Joined clips, not real sentences", {"bullet": True, "color": GREY}),
    ], size=20, space_after=12)
    drop(s, 105)
    xr = 6.7
    n0 = len(s.shapes)
    hd = s.shapes.add_textbox(Inches(xr), Inches(1.6), Inches(6), Inches(0.5))
    fill_text(hd, [("Each fold tests on one unseen signer", {"bold": True, "size": 22, "color": PURPLE})])
    sq, gp = 0.5, 0.08
    for k in range(8):  # folds 9-10 are reserved and produced no result, so they are not drawn
        fill, line, fc = (CARD_GREY, None, TEXT) if k < 6 else (PURPLE, None, "FFFFFF")
        b = box(s, xr + k * (sq + gp), 2.3, sq, sq, fill, line, shape=MSO_SHAPE.RECTANGLE)
        b.text_frame.margin_left = b.text_frame.margin_right = 0
        fill_text(b, [(str(k + 1), {"bold": True, "size": 16, "color": fc, "align": PP_ALIGN.CENTER})], space_after=0)
    for x, wdt, lab, col in ((xr, 6 * sq + 5 * gp, "development (results: 4–6)", GREY),
                             (xr + 6 * (sq + gp), 2 * sq + gp, "confirmation", PURPLE)):
        t = s.shapes.add_textbox(Inches(x - 0.3), Inches(2.85), Inches(wdt + 0.6), Inches(0.35))
        fill_text(t, [(lab, {"size": 14, "bold": col == PURPLE, "color": col, "align": PP_ALIGN.CENTER})])
    folds = list(s.shapes)[n0:]
    t = s.shapes.add_textbox(Inches(xr), Inches(3.45), Inches(6), Inches(2.6))
    fill_text(t, [
        ("The test signer never appears in training", {"bullet": True}),
        ("Signers 7–8: pass marks written down before testing", {"bullet": True}),
        ("A sequence counts only if every token ID is right", {"bullet": True}),
    ], size=18, space_after=8)
    animate(s, [(folds, 0), ([t], 0)])

    # Isolated signs ---------------------------------------------------------
    s = new("WAIMLAp Content", NOTES["isolated"], section="Results · one sign")
    title(s, "On one sign, does the output start with the right label?")
    drop(s, 104)
    width = Inches(8.6)
    s.shapes.add_picture(str(FIG / "isolated_prefix.png"), (prs.slide_width - width) // 2,
                         Inches(1.45), width=width)

    # 5 · Result: development -------------------------------------------------
    s = new("WAIMLAp Figure", NOTES[5], section="Results · several signs, development")
    title(s, "On 2–8 signs, strict accuracy rises about tenfold")
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
    s = new("WAIMLAp Figure", NOTES[6], section="Results · several signs, unseen signers")
    title(s, "Both held-out signers pass all three preregistered criteria")
    picture_in(s, 104, FIG / "confirmation_accuracy.png")
    fill_text(ph(s, 105), [
        ("H1 ✓  beats our earlier model", {"bold": True, "size": 20, "color": PURPLE, "space_after": 2}),
        ("wins/losses 99/3 and 108/6", {"size": 16, "space_after": 14}),
        ("H2 ✓  ≥ 11.9%", {"bold": True, "size": 20, "color": PURPLE, "space_after": 2}),
        ("12.5% and 14.1% strict accuracy", {"size": 16, "space_after": 14}),
        ("H3 ✓  visually ordered", {"bold": True, "size": 20, "color": PURPLE, "space_after": 2}),
        ("98–99% of sign pairs in the right order; shuffling the video segments costs 34–38 points of edit similarity",
         {"size": 16}),
    ])
    side = ph(s, 105)
    animate(s, [([(side, 0, 1)], 0), ([(side, 2, 3)], 0), ([(side, 4, 5)], 0)])

    # 7 · Result: Gemma audit -------------------------------------------------
    s = new("WAIMLAp Figure", NOTES[7], section="Results · can the LLM repair a draft?")
    title(s, "Gemma recovered no exact label at tested nonzero rates")
    picture_in(s, 104, FIG / "gemma_audit.png")
    fill_text(ph(s, 105), [
        ("Exploratory audit", {"bold": True, "size": 20, "color": TEAL}),
        ("Clean input: 95% kept exact", {"size": 18, "space_after": 12}),
        ("Tested nonzero rates: 0% exact before and after", {"bold": True, "size": 18, "color": PURPLE,
                                                             "space_after": 12}),
        ("Character overlap (chrF) drops 0.8–2.1 points", {"size": 18, "space_after": 12}),
        ("Illustration: per / ma / me → perfume?", {"size": 18, "color": GREY}),
    ])
    side = ph(s, 105)
    animate(s, [([(side, 0, 1)], 0), ([(side, 2, 2)], 0), ([(side, 3, 4)], 0)])

    # Conclusions -------------------------------------------------------------
    s = new("WAIMLAp Two Columns", NOTES[8], section="Conclusions")
    title(s, "A small adapter can write a draft in Gemma’s own token IDs")
    for idx, fill, line, head, lines in (
        (104, LT_TEAL, TEAL_LINE, "What we showed",
         ["One sign: right label first on 69–72% of clips; exact and stopped on 11–14%",
          "2–8 signs: 12.5–14.1% strict accuracy on unseen signers, in visual order",
          "Gemma: zero exact recoveries at tested nonzero error rates"]),
        (105, CARD_GREY, None, "Limits",
         ["One training seed (23) and two confirmation signers",
          "Joined clips: no sentence syntax, no transitions between signs",
          "Gemma has not yet read Imitator’s own drafts"]),
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

    # Future work -------------------------------------------------------------
    s = new("WAIMLAp Content", NOTES["future"], section="Future work")
    title(s, "Next: real signed sentences")
    drop(s, 104)
    f = flow(s, [
        ("1 · Real sentences", "signed sentences with\nsyntax and context", LT_TEAL, TEAL_LINE),
        ("2 · Correction in context", "can the LLM repair a draft\nwhen it has a sentence?", LT_TEAL, TEAL_LINE),
        ("3 · Richer drafts", "pass top-k alternatives,\nstill scoring the transcript", LT_TEAL, TEAL_LINE),
    ], y0=2.0, h=1.9, gap=0.9)
    for c in (f[1], f[3]):  # three directions, not a pipeline
        c._element.getparent().remove(c._element)
    t = s.shapes.add_textbox(Inches(0.6), Inches(4.5), Inches(12.1), Inches(0.8))
    fill_text(t, [("Goal: a reader receives the signer’s message.",
                   {"bold": True, "size": 24, "color": PURPLE, "align": PP_ALIGN.CENTER})])
    animate(s, [([f[0]], 0), ([f[2]], 0), ([f[4]], 0), ([t], 0)])

    # 9 · Closing -------------------------------------------------------------
    s = new("WAIMLAp Closing", NOTES[9], number=False)
    fill_text(ph(s, 105), [("Thank you · Questions?", {"bold": True, "size": 54, "color": "FFFFFF"})])
    fill_text(ph(s, 106), [
        ("Christian Velasquez Borasino", {"bold": True, "size": 20, "color": "FFFFFF"}),
        ("Universidad Peruana de Ciencias Aplicadas (UPC)  ·  Paper ID 58", {"size": 18, "color": LAVENDER}),
    ])

    s.shapes.add_picture(str(FIG / "linkedin_qr.png"), Inches(10.3), Inches(3.6), width=Inches(2.3))
    t = s.shapes.add_textbox(Inches(10.3), Inches(5.95), Inches(2.3), Inches(0.4))
    fill_text(t, [("LinkedIn", {"bold": True, "size": 16, "color": "FFFFFF", "align": PP_ALIGN.CENTER})])

    # 10 · Backup -------------------------------------------------------------
    s = new("WAIMLAp Figure", NOTES[10], section="Backup")
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
       "and Rody Vilchez. Before the method, let me show you a short video.",
    "hook": "Many of us have tried this: you give an AI model a video, you ask what is happening, and it "
       "describes the scene. So here is a short one. [click the clip to play] What did this person just "
       "say? [pause, let the room look] [click] Could a model understand it? [click] It is the sign for "
       "perfume in Argentinian Sign Language. Most of us in this room could not read it. An LLM can write "
       "a whole paragraph about perfume. But its input is text, and this is movement.",
    "context": "So this is our starting point. [click] On one side a signer moves. [click] On the other side "
       "the LLM already knows language: it writes fluent Spanish. What it does not know is sign language. "
       "[click] So do we have to train a sign translator from scratch? [click] Our bet is no: maybe a "
       "small adapter is enough. A small visual model that learns to write in the LLM's own input format, "
       "while the LLM stays frozen. That is the idea we call Imitator. [pause] But if a small adapter "
       "talks to the LLM, what exactly should it hand over?",
    "history": "One slide of previous work. When you type the word perfume, the LLM first turns it into token "
       "IDs, whole numbers, one per word piece. Each ID then becomes an embedding, a vector. [click] Our "
       "first, published Imitator aimed at the embeddings: it produced vectors meant to sit close to the "
       "LLM's own, and the preliminary loss was very low, about 8 times 10 to the minus 4. But closeness "
       "does not guarantee the intended word, and we had no exact way to check what the LLM would read. "
       "We are not saying it failed. We could not verify it. "
       "[click] So in this work the adapter writes the exact Gemma token IDs. Each ID is right or wrong, "
       "and we can score the draft before the LLM reads anything. [pause] That is the method. Here is "
       "the model.",
    3: "From the video we take 2D keypoints, 111 per frame. An encoder reads the movement and keeps the "
       "order of what it saw. A small decoder then writes Gemma token IDs one at a time until an end "
       "marker. Those IDs are ordinary text for Gemma 3n, our frozen LLM. [click] We score Imitator on its "
       "own, before any LLM touches the draft, so fluent text cannot hide a visual error. [click] In this "
       "paper Gemma is tested apart, on labels where we inject errors. It has not yet read Imitator's own "
       "drafts. [pause] Every ID must match. So we start with the easiest case: one sign.",
    4: "We use LSA64: 64 Argentinian signs from ten signers. The target is the Gemma tokenization of "
       "each sign's Spanish label. First, one clip with one sign. Then, as a step toward our original goal "
       "of sentences, we join two to eight clips in random order and ask for the labels in order. These "
       "are joined clips, not real sentences: there is no sentence syntax. [click] Every fold tests on a "
       "signer the model never saw. [click] For signers 7 and 8 we wrote the pass marks down before "
       "testing. [pause] So: one sign, an unseen signer. Does Imitator write the right label?",
    "isolated": "These are retrained copies of the model, which we call replicas, tested on single clips of "
                "signers 7 and 8. Does the output start with every token of the correct "
                "label? It does on 229 of 320 clips on fold 7 and 220 of 320 on fold 8. The whiskers "
                "show 95 percent bootstrap intervals from resampling the 64 signs within each signer. "
                "They do not estimate variation across signers. A prefix match can still continue: "
                "on one clip the signer signed música, and Imitator wrote música música. For a strict "
                "match the decoder must stop after the label. That happens on 44 and 35 clips, about 14 "
                "and 11 percent. The model was trained on streams of two or more signs, so running "
                "long is consistent with not having learned to stop after one. Now imagine a video has several signs. Can we recover them in order?",
    5: "On the development signers, strict accuracy, meaning the whole sequence exactly right, goes from about "
       "2 percent with our initial autoregressive decoder to 17 to 20 percent. Roughly ten times. Adding "
       "positional encoding moved pairwise label order from chance to about 0.99. But to be clear: most "
       "sequences still have at least one error. It is viable, not solved. [pause] Does it hold on new signers?",
    6: "On the two signers we had never touched, Imitator reaches 12.5 and 14.1 percent, above the "
       "11.9 pass mark we fixed in advance, against 1.8 and 2.7 for CIF, our earlier model and the "
       "comparator named in the preregistration. Paired, it wins 99 times and loses 3 on "
       "signer 7, and 108 to 6 on signer 8. Edit similarity is 53 and 59 percent. And if we shuffle the "
       "sign segments while keeping the target, edit similarity drops by about 35 points: the order really comes "
       "from the video. All three criteria pass. Now back to the idea that started Imitator: could a "
       "correction prompt help Gemma recover a word when the visual draft gets some tokens wrong?",
    7: "For illustration, if per, fu, me became per, ma, me, could the surviving pieces still suggest "
       "perfume? Those pieces are schematic. In the audit, we replaced IDs at selected positions in "
       "reference labels of at least two IDs. Replacements came from real one-ID LSA64 labels. At each "
       "nominal rate, Gemma received 120 decoded inputs and an explicit correction prompt. It preserved "
       "95 percent of clean inputs. At the tested nonzero rates, exact match stayed at zero and chrF, "
       "a character-overlap score, decreased. This measures recovery for synthetic substitutions in isolated labels.",
    8: "To conclude. A small adapter can write a draft in Gemma's own vocabulary. On isolated signs the "
       "replicas start with the right label on about 70 percent of clips, and are exact on 11 to 14. On streams of two to eight signs the "
       "original checkpoints give 12 to 14 percent fully correct, visually ordered sequences on signers "
       "never seen in training. And in the exploratory audit, Gemma recovered no exact label at the tested "
       "nonzero error rates. [click] The limits: one training seed, two confirmation signers, joined clips "
       "without sentence syntax, and Gemma has not yet read Imitator's own drafts.",
    "future": "So what comes next. [click] First, real signed sentences, with syntax and context. [click] "
       "Second, correction in context: with a whole sentence around a damaged word, can the LLM repair "
       "it? [click] Third, richer drafts: pass the LLM top-k alternatives, while we keep scoring the "
       "transcript on its own. [click] Think back to the person at the start. The goal is that a reader "
       "receives that person's message.",
    9: "Thank you. I am happy to take questions.",
    10: "Backup for questions on interpretability. For one unseen-signer sequence, input-times-gradient and "
        "region ablation both point to the left hand for the central subtokens. It is a single example, so "
        "it is descriptive, not a dataset-wide claim.",
}

if __name__ == "__main__":
    make_figures()
    build()
