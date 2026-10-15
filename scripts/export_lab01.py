"""Export executed notebooks with labelled figures and an optional projector view."""
from pathlib import Path
import re
import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = {
    "FirstLab": [
        "Four 5 by 5 pixel images: centred vertical, horizontal, shifted vertical, and blank.",
        "Input pixels, vertical detector weights, and their elementwise contributions.",
        "A 3 by 3 matrix of accidental pairwise prediction-target differences caused by broadcasting.",
        "A unit square before and after an area-preserving shear.",
        "Animation of a grid and unit square during a shear transformation.",
    ],
    "LearningPreview": [
        "A line before and after training, alongside loss curves for three learning rates.",
        "Animation of an adjustable line approaching the data while training loss decreases.",
        "Two functions agreeing at three training points but disagreeing on a held-out point.",
    ],
}
ICONS = {"minus":"−", "plus":"+", "fast-backward":"⏮", "step-backward":"◀|", "backward":"◀", "play fa-flip-horizontal":"◀", "play":"▶", "pause":"Ⅱ", "step-forward":"|▶", "forward":"▶", "fast-forward":"⏭"}

for name, descriptions in DESCRIPTIONS.items():
    path = ROOT / "Lab01" / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    for projector in (False, True):
        exporter = HTMLExporter()
        exporter.exclude_input = projector
        exporter.exclude_input_prompt = projector
        exporter.exclude_output_prompt = projector
        body, _ = exporter.from_notebook_node(nb)
        # These patterns target the known HTML emitted by nbconvert/Matplotlib.
        image_index = [0]
        def describe(match):
            tag = match.group(0)
            index = image_index[0]
            image_index[0] += 1
            if index >= len(descriptions):
                return tag
            tag = re.sub(r'\s+alt="[^"]*"', '', tag)
            return tag[:-1] + f' alt="{descriptions[index]}">'
        body = re.sub(r'<img\b[^>]*>', describe, body)
        for icon, label in ICONS.items():
            body = re.sub(r'<i class="fa fa-' + icon + r'"></i>', label, body)
        body = re.sub(r'<link[^>]*font-awesome[^>]*>', '', body)
        if projector:
            body = body.replace('</head>', '<style>body{font-size:18px}main{max-width:1200px;margin:auto}.jp-RenderedHTMLCommon p,.jp-RenderedHTMLCommon li{font-size:18px;line-height:1.6}</style></head>')
        suffix = '-projector' if projector else ''
        output = path.with_name(name + suffix + '.html')
        output.write_text(body)
        print(output.name)
