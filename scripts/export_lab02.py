"""Export both lab parts, plus any internal answer notebooks passed as arguments."""
from pathlib import Path
import re,sys,html,json,os
import nbformat
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parents[1]
ICONS={'minus':'−','plus':'+','fast-backward':'⏮','step-backward':'◀|','backward':'◀',
       'play fa-flip-horizontal':'◀','play':'▶','pause':'Ⅱ','step-forward':'|▶','forward':'▶','fast-forward':'⏭'}
PATHS=json.loads((Path(__file__).parent/'lab02_paths.json').read_text())
LESSONS={ROOT/'Lab02'/PATHS[f'{name}.ipynb']:name for name in ['SecondLab','PerceptronLab']}
paths=list(LESSONS)+[Path(p) for p in sys.argv[1:]]
for path in paths:
    nb=nbformat.read(path,as_version=4)
    descriptions=[]
    for cell in nb.cells:
        for out in cell.get('outputs',[]):
            data=out.get('data',{})
            if 'image/png' in data:
                descriptions.append(cell.metadata.get('figure_description','Classroom demonstration figure.'))
            elif 'text/html' in data:
                descriptions.extend([cell.metadata.get('figure_description','Classroom animation.')] * len(re.findall(r'<img\b',data['text/html'])))
    for projector in ([False,True] if path in LESSONS else [False]):
        exp=HTMLExporter();exp.exclude_input=projector;exp.exclude_input_prompt=projector;exp.exclude_output_prompt=projector
        body,_=exp.from_notebook_node(nb);counter=[0]
        def label(match):
            i=counter[0];counter[0]+=1;tag=match.group(0)
            if i>=len(descriptions):return tag
            tag=re.sub(r'\s+alt="[^"]*"','',tag)
            return tag[:-1]+f' alt="{html.escape(descriptions[i],quote=True)}">'
        body=re.sub(r'<img\b[^>]*>',label,body)
        for icon,text in ICONS.items():body=body.replace(f'<i class="fa fa-{icon}"></i>',text)
        body=re.sub(r'<link[^>]*font-awesome[^>]*>','',body)
        body=re.sub(r'<script[^>]*src="https?://[^"]*"[^>]*>\s*</script>','',body)
        if projector:
            body=body.replace('</head>','''<style>body{font-size:19px}main{max-width:1180px;margin:auto}.jp-RenderedHTMLCommon p,.jp-RenderedHTMLCommon li{font-size:19px;line-height:1.6}.jp-RenderedHTMLCommon h3{margin-top:2em}.jp-OutputArea-output img{max-width:100%;height:auto}</style></head>''')
        output=(ROOT/'Lab02'/PATHS[LESSONS[path]+('-projector' if projector else '')+'.html']) if path in LESSONS else path.with_suffix('.html')
        def relocate(m):
            url=m.group(1)
            if re.match(r'^[a-z]+:',url) or url.startswith('#'):return m.group(0)
            return 'href="'+os.path.relpath((path.parent/url).resolve(),output.parent)+'"'
        body=re.sub(r'href="([^" ]+)"',relocate,body)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(body);print(output)
