"""Build Part A (NumPy/homework) and Part B (perceptron). Overwrites authored notebooks."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf
import json, os, re
PATHS=json.loads((Path(__file__).parent/"lab02_paths.json").read_text())
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(dedent(s).strip()))
def code(s,id=None,alt=None):
    c=nbf.v4.new_code_cell(dedent(s).strip())
    if id:c.id=id
    if alt:c.metadata['figure_description']=alt
    cells.append(c)
def save(name):
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    target=ROOT/'Lab02'/PATHS[f'{name}.ipynb']
    for cell in nb.cells:
        if cell.cell_type=='markdown':
            for old,new in PATHS.items():
                cell.source=cell.source.replace(f'({old})', '('+os.path.relpath(ROOT/'Lab02'/new,target.parent)+')')
    target.parent.mkdir(parents=True,exist_ok=True)
    nbf.validate(nb);nbf.write(nb,target)
setup='''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
plt.rcParams.update({'font.size':12,'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
BLUE, ORANGE = '#1768ac', '#c76319'
print('NumPy', np.__version__)
'''
md('''
# Lab 02 · Part A — NumPy and Homework 1
**First 60 minutes.** Then open [Part B: the perceptron](PerceptronLab.ipynb)
or its [projector view](PerceptronLab-projector.html).

## Goal
Read array shapes, select and reduce data, distinguish elementwise from matrix multiplication,
and translate matrix operations into plain Python for Homework 1.

**Scope:** NumPy exercises and linear systems follow the official Lab02. Part B is an additional
classroom explanation of the published perceptron course, requested by your teacher.
**Homework 1 must use Python lists and standard Python, without NumPy or a linear algebra framework.**
NumPy here is a way to explore the mathematics, not an allowed homework implementation.

## Setup
CPU only; data are included. Select **RN 2026 (local)** on the classroom machine.
For a fresh environment: `python -m pip install -r requirements-lab02.txt`.
On Colab upload the notebook; it needs NumPy and Matplotlib.
''')
code(setup)
md('''
## Steps
### 1 · Shapes, indexing and reductions (0–12 min)
A matrix has rows and columns. Before Run: what are the shapes of `A`, `A[0]`, `A[:, 0]`,
and `A.T`? What is the difference between summing over axis 0 and axis 1?
''')
code('''
A = np.array([[1., 2., 3.], [4., 5., 6.]])
print('A:', A, '\\nshape:', A.shape)
print('row 0:', A[0], 'column 0:', A[:, 0])
print('column sums (axis 0):', A.sum(axis=0))
print('row sums (axis 1):', A.sum(axis=1))
print('transpose:', A.T)
''')
md('''
`*` multiplies element by element; `@` takes row dot products. Broadcasting repeats the
vector along rows. Predict both outputs, then connect each row of `A @ x` to an equation.
''')
code('''
A = np.array([[2., 1.], [1., 2.]])
x = np.array([2., 3.])
print('A * x (shape 2×2):\\n', A * x)
print('A @ x (shape 2):', A @ x)
print('sum of each row of A * x:', (A * x).sum(axis=1))
assert np.allclose((A*x).sum(axis=1), A@x)
''')
md('''
### 2 · Masks: select first, then count (12–22 min)
Official NumPy exercise data: **1 = spam, 0 = ordinary mail**. No classifier is trained here.
`&` combines array conditions; use parentheses. The count of `True` values is a reduction.
''')
code('''
actual = np.array([1,1,1,1,0,0,1,0,0,0])
predicted = np.array([1,1,1,0,1,0,1,1,0,0])
missed_spam = (actual == 1) & (predicted == 0)
print('missed-spam mask:', missed_spam)
print('message IDs:', np.flatnonzero(missed_spam)+1)
print('missed spam count:', np.count_nonzero(missed_spam))
''')
md('''
**A1 (5 min):** return `(TP, FP, FN, TN)` using masks, then compute accuracy.
TP = actual 1 / predicted 1; FP = actual 0 / predicted 1; FN = actual 1 / predicted 0;
TN = actual 0 / predicted 0. Inputs are nonempty, same-length 1D binary arrays.
Continue the official [NumPy exercises](NumpyExcercises.ipynb) for F1 and benchmarking;
[MetricsPractice.ipynb](MetricsPractice.ipynb) provides extra explanations after class.
''')
code('''
def counts_numpy(gt, pred):
    # TODO: four Boolean masks and counts; return TP, FP, FN, TN.
    return None
result=counts_numpy(actual,predicted)
if result is None: print('A1: NOT ATTEMPTED')
else:
    assert tuple(result)==(4,2,1,3)
    tp,fp,fn,tn=result
    print('counts:',result,'accuracy:',(tp+tn)/len(actual))
''','exercise-counts')
md('''
### 3 · Copies matter when replacing matrix columns (22–30 min)
The homework's Cramer matrices replace a column. A shallow copy of a nested Python list
still shares the rows. NumPy slices may also share data. Predict which originals change.
''')
code('''
original = [[2,1],[1,2]]
shallow = original.copy()
shallow[0][0] = 99
print('shallow-copy mutation reaches original:', original)
original = [[2,1],[1,2]]
independent = [row.copy() for row in original]
independent[0][0] = 99
print('row copies preserve original:',original)
arr = np.array([[2,1],[1,2]])
view = arr[:,0]
copy = arr[:,0].copy()
view[0] = 99
print('NumPy original:',arr.tolist(),'independent column:',copy.tolist())
''')
md('''
### 4 · A·x = B: equations and transformations (30–42 min)
Two ticket bundles cost 7 and 8 units: `2x + y = 7`, `x + 2y = 8`.
Every equation is a line. The intersection satisfies both.
A zero determinant means no inverse: the system may have no solution or infinitely many.
''')
code('''
A=np.array([[2.,1.],[1.,2.]])
B=np.array([7.,8.])
solution=np.linalg.solve(A,B) # demonstration only: prohibited in Homework 1
print('solution:',solution,'residual:',A@solution-B)
u=np.linspace(-1,6,120)
fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
for ax,title,v1,v2 in zip(axes,['One solution','No solution','Infinitely many'],
                         [7-2*u]*3,[(8-u)/2,9-2*u,7-2*u]):
    ax.plot(u,v1,color=BLUE,label='Equation 1',lw=3)
    ax.plot(u,v2,color=ORANGE,ls='--',label='Equation 2',lw=2)
    ax.set(title=title,xlabel='x',xlim=(-1,6),ylim=(-2,10));ax.grid(alpha=.2)
axes[0].set_ylabel('y');axes[0].scatter(*solution,color='black',zorder=4)
axes[0].annotate('(2, 3)',solution,xytext=(7,8),textcoords='offset points')
axes[2].legend(fontsize=9);fig.tight_layout();plt.show()
''',alt='Two equations with a single intersection at (2,3), parallel lines, and coincident lines.')
md('''
**On paper:** determinant `ad−bc = 3`. Replace the first column with B: determinant 6 → x=2.
Replace the second column: determinant 9 → y=3. Verify both equations.

**A2 (6 min):** implement a matrix-vector product using **plain Python lists**.
For a 2×2 matrix and a length-2 vector, return the two row dot products.
''')
code('''
def matvec_2x2(matrix, vector):
    # TODO: standard Python only
    return None
answer=matvec_2x2([[2,1],[1,2]],[2,3])
if answer is None: print('A2: NOT ATTEMPTED')
else:
    assert answer==[7,8]
    assert matvec_2x2([[1,0],[0,1]],[-2,5])==[-2,5]
    print('A2: checks passed')
''','exercise-matvec')
md('''
### 5 · Translate the ideas into Homework 1 (42–55 min)
Open [Assignment1.ipynb](Assignment1.ipynb), with [system.txt](system.txt) beside it.
The graded system is 3×3, different from our ticket example.

| Task | Implementation idea | Test it with |
|---|---|---|
| Parse equations (1 point) | split tokens; signs; implicit coefficients ±1 | `x - y + 4z = 6` |
| Determinant, trace, norm, transpose, matvec (5) | list indexing, sums, loops | identity / diagonal matrices |
| Cramer (1) | copy rows, replace one column, determinant ratio | verify original A is unchanged |
| Inverse / adjugate (3) | minors → signed cofactors → transpose → divide by det | your own matvec gives A·solution ≈ B |

The input guarantees x,y,z order and space-separated terms. `- z` means coefficient −1;
it does not require a general symbolic algebra parser. Guard determinant zero before division.
For floating-point answers compare with a tolerance, not exact equality.
''')
code('''
# A tiny parsing building block, not the full homework parser.
tokens='x - y + 4z = 6'.split()
print('tokens:',tokens)
for token in ['x','y','4z']:
    coefficient=token[:-1]
    print(token,'coefficient magnitude:',float(coefficient) if coefficient else 1.0)
# Keep track of separate + / - tokens when assembling a full row.
''')
md('''
## Checks (55–60 min)
1. What changes if a vector has shape `(n, 1)` instead of `(n,)`?
2. Why is copying only the outer list insufficient for Cramer's rule?
3. How do you verify a candidate solution without comparing it to a stored answer?
4. Which tools are forbidden in Homework 1?

## Next Steps
The official deadline is **before Lab 3**. Use standard Python and lists for Homework 1.
Now move to [Part B: the perceptron](PerceptronLab.ipynb) / [projector](PerceptronLab-projector.html).

Sources: [official Lab02](https://github.com/Tensor-Reloaded/Neural-Networks-2026/tree/main/Lab02),
reviewed at `4dc46e1a`; [published courses](https://sites.google.com/view/rbenchea/neural-networks).
The optional tree challenge remains in [ProgrammingChallenge.ipynb](ProgrammingChallenge.ipynb).
''')
save('SecondLab')
cells=[]
md('''
# Lab 02 · Part B — A perceptron learns from one mistake
**Second 60 minutes · Companion to the published Course 2.**
This is additional classroom material, not a new requirement in the official Lab02 NumPy assignment.
[Part A](SecondLab.ipynb) · [Interactive playground](PerceptronPlayground.html)

## Goal
Explain the decision boundary, calculate one update by hand, and understand how repeated
sample updates move a classifier. We use labels **−1 and +1** throughout this part.

## Setup
NumPy + Matplotlib, CPU only, no dataset downloads. The plots have saved outputs;
animations have embedded frames in the HTML export. This notebook runs independently of Part A.
''')
code(setup)
md('''
## Steps
### 1 · A weighted vote, then a decision (60–70 min)
`z = w₁x₁ + w₂x₂ + b`; prediction = **+1 if z ≥ 0, otherwise −1**.
Weights control each feature's influence. Bias changes the cutoff.
The score is not a calibrated probability. `np.sign(0)` returns 0, so it would not implement our rule.

The boundary `w₁x₁ + w₂x₂ + b = 0` is a line in 2D. The vector w is perpendicular to it,
pointing towards increasing score. Moving along the boundary leaves the score zero.
''')
code('''
def predict_pm1(X,w,b):
    return np.where(np.asarray(X)@w+b >= 0,1,-1)

def draw_model(ax,w,b,X,y,title,highlight=None):
    ax.clear()
    grid=np.linspace(-3.4,3.4,100)
    gx,gy=np.meshgrid(grid,grid)
    zz=w[0]*gx+w[1]*gy+b
    ax.contourf(gx,gy,np.where(zz>=0,1,-1),levels=[-2,0,2],colors=['#faebdc','#e2edf8'])
    if np.linalg.norm(w)>1e-12 and zz.min()<0<zz.max():
        ax.contour(gx,gy,zz,levels=[0],colors=['#172337'],linewidths=2)
    for label,marker,color in [(1,'o',BLUE),(-1,'s',ORANGE)]:
        rows=X[y==label]
        ax.scatter(rows[:,0],rows[:,1],marker=marker,color=color,s=80,
                   edgecolors='white',label=f'True {label:+d}',zorder=3)
    mistakes=predict_pm1(X,w,b)!=y
    ax.scatter(X[mistakes,0],X[mistakes,1],s=200,facecolors='none',edgecolors='#172337',
               linewidths=1.5,label='Mistake',zorder=4)
    norm=np.linalg.norm(w)
    if norm>1e-12:
        unit=w/norm
        ax.annotate('',xy=unit,xytext=(0,0),arrowprops=dict(arrowstyle='->',color='#172337',lw=2))
        ax.text(unit[0]+.12,unit[1]+.12,'w',fontsize=11)
    if highlight is not None:
        ax.scatter(*highlight,marker='*',s=500,facecolors='none',edgecolors='#7a3c8c',
                   linewidths=1.8,label='Current sample',zorder=5)
    ax.set(xlim=(-3.4,3.4),ylim=(-3.4,3.4),xlabel='x₁',ylabel='x₂',title=title,aspect='equal')
    ax.grid(alpha=.15);ax.legend(loc='upper left',fontsize=8)

X_demo=np.array([[-2.,-1.],[-1.,-2.],[1.,2.],[2.,1.]])
y_demo=np.array([-1,-1,1,1])
fig,axes=plt.subplots(1,3,figsize=(13,4.6))
for ax,w,b,title in zip(axes,[np.array([1.,0.]),np.array([1.,1.]),np.array([1.,1.])],
                        [0.,0.,-2.],['w=(1,0), b=0','Rotate w: (1,1), b=0','Keep w; shift b to −2']):
    draw_model(ax,w,b,X_demo,y_demo,title)
fig.tight_layout();plt.show()
''',alt='Three perceptron boundaries: a vertical line, a rotated line, and a parallel shifted line; shaded predicted classes and normal arrows.')
md('''
### 2 · Move w and b yourself (70–78 min)
Open the [interactive playground](PerceptronPlayground.html).
First change only b: the boundary shifts in parallel. Then change one weight: its direction changes.
Try w₂=0: the line can be vertical, so solving for x₂ by dividing by w₂ is not a universal plotting method.
Try w=(0,0): prediction is constant; there is no unique separating line.

Blue background predicts +1; orange predicts −1. Circles/squares show **true** labels.
The background is a decision region, not a confidence estimate.
''')
md('''
### 3 · A mistake changes the next vote (78–88 min)
We follow the rule in the supplied exercise image exactly:

```
error = prediction - true
w_new = w - learning_rate * error * x
b_new = b - learning_rate * error
```

With labels ±1, error is **−2, 0, or +2**. On correct predictions, nothing changes.
If the true label is +1 but we predict −1, error = −2, so the update moves w towards x
and increases b. For the opposite mistake, it moves w away from x and decreases b.

Another common convention adds `η*y*x` only on mistakes. Here the step is **twice as large**
for the same numeric η, because `prediction − true = −2*y` on a mistake. Do not mix these conventions.
''')
code('''
def perceptron_step(w,b,x,y,eta):
    z=float(w@x+b)
    prediction=1 if z>=0 else -1
    error=prediction-y
    return dict(z=z,prediction=prediction,error=error,
                w_new=w-eta*error*x,b_new=b-eta*error)

# Worked example with easy numbers; different from your exercise below.
w0=np.array([-.5,.5]);b0=-.5
x0=np.array([1.,1.]);y0=1;eta0=.1
worked=perceptron_step(w0,b0,x0,y0,eta0)
print('Before:',worked['z'],'prediction:',worked['prediction'],'error:',worked['error'])
print('After: w =',worked['w_new'],'b =',worked['b_new'])
print('Score after:',worked['w_new']@x0+worked['b_new'])
''')
md('''
The worked example has `z=−0.5`, prediction −1, true +1, error −2.
So Δw=(+0.2,+0.2), Δb=+0.2; the new score is +0.1.

**Predict the motion before playing.** The animation interpolates between the old and new
parameters to make the motion visible; the algorithm itself makes **one discrete update**.
One update need not fix every mistake, or even the current sample for every learning rate.
''')
code('''
fig,axes=plt.subplots(1,2,figsize=(11,5))
scene_X=np.vstack([X_demo,x0]);scene_y=np.append(y_demo,y0)
def draw_update(alpha):
    w=(1-alpha)*w0+alpha*worked['w_new'];b=(1-alpha)*b0+alpha*worked['b_new']
    draw_model(axes[0],w0,b0,scene_X,scene_y,'Before the update',x0)
    draw_model(axes[1],w,b,scene_X,scene_y,
               f'Visual transition {alpha:.0%} · sample z={w@x0+b:+.2f}',x0)
fig.subplots_adjust(top=.87,bottom=.13,wspace=.28)
draw_update(1);plt.show()
''',alt='Before and after one update on x=(1,1), true label +1: the sample score changes from −0.5 to +0.1.')
code('''
fig,axes=plt.subplots(1,2,figsize=(11,5))
fig.subplots_adjust(top=.87,bottom=.13,wspace=.28)
anim=FuncAnimation(fig,draw_update,frames=np.linspace(0,1,21),interval=120,repeat=False)
plt.close(fig);display(HTML(anim.to_jshtml(default_mode='once')))
''',alt='Animation of a single perceptron update, with an explicitly labelled visual interpolation between old and new parameters.')
md('''
### 4 · Your turn: calculate one sample update (88–103 min)
Work individually for 5 minutes, compare with a neighbour for 3, then discuss.
Use the displayed numbers below as exact inputs; round only the final answer.
They are transcribed from the teacher's screenshot.

```
w = [+0.7092, -0.4812]
b = -0.6033
learning_rate = 0.1
x = [+1.993, -2.936]
true label = -1
prediction = +1 if z >= 0, else -1
```

**B1 — Complete on paper:** z → prediction → error → Δw₁, Δw₂, Δb → new w, new b.
Then compute the new score on the same sample. Did its predicted class change?

**Before arithmetic:** sketch the old boundary and predict which way the update should move it.
In the playground, Load exercise restores these exact values; don't reveal the calculation yet.
''')
code('''
exercise_w=np.array([.7092,-.4812]);exercise_b=-.6033
exercise_x=np.array([1.993,-2.936]);exercise_y=-1;exercise_eta=.1
fig,ax=plt.subplots(figsize=(7,5))
draw_model(ax,exercise_w,exercise_b,np.array([exercise_x]),np.array([exercise_y]),
           'Exercise B1 · initial model only',exercise_x)
fig.tight_layout();plt.show()
''',alt='Unsolved exercise B1: initial decision boundary and a highlighted true −1 sample lying in the predicted +1 region.')
code('''
def hand_calculation():
    # After doing the arithmetic, return a dictionary:
    # z, prediction, error, w_new (list of two numbers), b_new
    return None
answer=hand_calculation()
if answer is None: print('B1: NOT ATTEMPTED — calculate on paper first.')
else:
    expected=perceptron_step(exercise_w,exercise_b,exercise_x,exercise_y,exercise_eta)
    for key in ['z','prediction','error','w_new','b_new']:
        assert np.allclose(answer[key],expected[key],rtol=0,atol=5e-5),f'Recheck {key}'
    print('B1: correct. New score:',np.dot(answer['w_new'],exercise_x)+answer['b_new'])
''','exercise-hand')
md('''
### 5 · A correct example needs no correction (103–110 min)
**B2 — Implement one step yourself.** Return `(new_weights, new_bias)`.
Use the same ±1 labels, z≥0 convention and prediction−true error. Do not modify the input array.
Test a positive mistake, negative mistake, correct example, and a score exactly zero.
''')
code('''
def student_update(w,b,x,y,eta):
    # TODO: score -> prediction -> error -> new parameters
    return None
result=student_update(np.array([1.,0.]),0.,np.array([1.,2.]),1,.1)
if result is None: print('B2: NOT ATTEMPTED')
else:
    cases=[([1.,0.],0.,[1.,2.],1),([1.,0.],0.,[1.,2.],-1),
           ([-1.,0.],0.,[1.,2.],1),([0.,0.],0.,[1.,2.],-1),([0.,0.],0.,[1.,2.],1)]
    for w,b,x,y in cases:
        w=np.array(w);x=np.array(x);old=w.copy()
        expected=perceptron_step(w,b,x,y,.1)
        wn,bn=student_update(w,b,x,y,.1)
        assert np.allclose(wn,expected['w_new']) and np.isclose(bn,expected['b_new'])
        assert np.array_equal(w,old),'Input weights changed in place'
    print('B2: five cases passed (both mistakes, correct prediction, and zero-score ties).')
''','exercise-update')
md('''
### 6 · Repeat: one sample, one decision, possibly one update (110–117 min)
An epoch visits all samples once. In online training the next sample sees the updated weights,
so order matters. Here we use a fixed order and an explicit maximum of 20 epochs.
The animation shows actual parameter states after mistakes, not interpolated intermediate weights.
A full pass with no mistakes certifies separation of this training set, not generalization.
''')
code('''
X_train=np.array([[-2.,1.],[-1.,0.],[1.,-2.],[2.,-1.],[1.,0.],[-1.,2.]])
y_train=np.array([-1,-1,-1,1,1,1])
w=np.array([.2,-.8]);b=.1;eta=.1
history=[(w.copy(),b,'Initial model')]
for epoch in range(20):
    errors=0
    for i,(x,y) in enumerate(zip(X_train,y_train)):
        step=perceptron_step(w,b,x,int(y),eta)
        if step['error']!=0:
            w,b=step['w_new'],step['b_new'];errors+=1
            history.append((w.copy(),b,f'Epoch {epoch+1}, sample {i+1}: true {y:+d}'))
    if errors==0: break
print('Epochs:',epoch+1,'updates:',len(history)-1,
      'training accuracy:',np.mean(predict_pm1(X_train,w,b)==y_train))
assert np.all(predict_pm1(X_train,w,b)==y_train)
fig,axes=plt.subplots(1,2,figsize=(11,5))
draw_model(axes[0],history[0][0],history[0][1],X_train,y_train,'Before training')
draw_model(axes[1],w,b,X_train,y_train,'After training · this small dataset')
fig.tight_layout();plt.show()
''',alt='The same six training samples before and after online perceptron training; all training samples are correctly classified at the end.')
code('''
fig,ax=plt.subplots(figsize=(7,5))
def draw_training(i):
    w,b,event=history[i]
    draw_model(ax,w,b,X_train,y_train,f'State {i}/{len(history)-1} · {event}')
fig.subplots_adjust(top=.87,bottom=.13)
training_anim=FuncAnimation(fig,draw_training,frames=len(history),interval=1100,repeat=False)
plt.close(fig);display(HTML(training_anim.to_jshtml(default_mode='once')))
''',alt='Actual perceptron parameter states after successive online mistake updates on a fixed six-point dataset.')
md('''
## Checks / Exit ticket (117–120 min)
1. What happens when the prediction is correct? What if z is exactly zero?
2. Why does the screenshot convention give an error of 2 rather than 1?
3. Does one update always improve accuracy on every training sample?
4. Can a single line solve XOR? Sketch alternating labels at the four corners of a square.

## Next Steps
For nonseparable data such as XOR, the perceptron has no separating line to find; keep a maximum
number of epochs. Convergence on separable data does not guarantee a maximum-margin separator
or good test accuracy. More expressive models come later.

Homework 1 remains the **plain-Python linear system assignment** from Part A.
This perceptron practice is not an additional graded submission.
Source: [Course 2 on the official course page](https://sites.google.com/view/rbenchea/neural-networks).
The exact B1 numbers and update convention come from the teacher's supplied screenshot.
''')
save('PerceptronLab')
print('Built SecondLab.ipynb (Part A) and PerceptronLab.ipynb (Part B)')
