#!/usr/bin/env python3
"""Plot from verified BDF outputs with true references where available."""
import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.14/site-packages"))
OUT = "/home/dskanda/Projects/yabi-bdf/plots"
os.makedirs(OUT, exist_ok=True)

def parse_trajectory(csv_path, expected_cols):
    traj = []
    try:
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('t,') or line.startswith('PASS') or 'PASS' in line or 'FAIL' in line or 'ACCURACY' in line or line.startswith('y1('):
                    continue
                parts = line.split(',')
                if len(parts) >= expected_cols:
                    try:
                        row = [float(p) for p in parts[:expected_cols]]
                        traj.append(row)
                    except ValueError:
                        continue
    except:
        return None
    if not traj: return None
    return np.array(traj)

def plot_brusselator():
    traj = parse_trajectory("/tmp/plot_bruss_traj.csv", 3)
    if traj is None: return
    t,y1,y2 = traj[:,0],traj[:,1],traj[:,2]
    fig,ax = plt.subplots(figsize=(7,4))
    ax.plot(t,y1,'o-',label='BDF y1',markersize=4,color='#d62728')
    ax.plot(t,y2,'s-',label='BDF y2',markersize=4,color='#1f77b4')
    ax.scatter([5],[0.938299],color='red',s=150,marker='*',zorder=10,label='Testbench ref y1(5)=0.938299')
    ax.scatter([5],[4.52728],color='blue',s=150,marker='*',zorder=10,label='Testbench ref y2(5)=4.52728')
    ax.set_xlabel('t'); ax.set_ylabel('y')
    ax.set_title('Brusselator (A=1,B=3, y0=[1,2]) — BDF vs testbench reference')
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"brusselator_curve.png"),dpi=150); plt.close(fig)

def plot_robertson():
    traj = parse_trajectory("/tmp/plot_robert_traj.csv", 4)
    if traj is None: return
    t,y1,y2,y3 = traj[:,0],traj[:,1],traj[:,2],traj[:,3]
    fig,ax = plt.subplots(figsize=(8,5))
    ax.semilogx(t,y1,'o-',label='BDF y1',markersize=5,color='#d62728')
    ax.semilogx(t,y2,'s-',label='BDF y2',markersize=5,color='#1f77b4')
    ax.semilogx(t,y3,'^-',label='BDF y3',markersize=5,color='#2ca02c')
    ax.scatter([10000],[0.107300429907],color='red',s=150,marker='*',zorder=10,label='Testbench ref y1(10^4)=0.107300')
    ax.scatter([10000],[4.800167041e-07],color='blue',s=150,marker='*',zorder=10,label='Testbench ref y2(10^4)=4.800e-07')
    ax.scatter([10000],[0.892699090076],color='black',s=120,marker='*',zorder=10,label='Testbench ref y3(10^4)=0.892699')
    ax.set_xlabel('t (log scale)'); ax.set_ylabel('y')
    ax.set_title('Robertson — BDF vs Hairer-Wanner/testbench reference')
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3,which='both')
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"robertson_curve.png"),dpi=150); plt.close(fig)

def plot_heat():
    alpha = 0.01
    t_curve = np.linspace(0, 50, 500)
    y_ref = np.exp(-alpha*np.pi**2*t_curve)*np.sin(np.pi*0.5)
    traj = parse_trajectory("/tmp/plot_heat_traj.csv", 2)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(t_curve, y_ref, 'k-', linewidth=2, label='Analytical $e^{-\\alpha\\pi^2 t}\\sin(\\pi/2)$')
    if traj is not None:
        ax.plot(traj[:,0], traj[:,1], 'o-', color='red', markersize=3, label='BDF (MOL N=100)')
    ax.set_xlabel('t'); ax.set_ylabel('$u(0.5,t)$')
    ax.set_title('Heat Equation — BDF vs analytical')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"heat_curve.png"),dpi=150); plt.close(fig)

if __name__ == "__main__":
    plot_brusselator(); plot_robertson(); plot_heat()
    print("Done")

# Enhanced heat plot with error bars (overwrites previous)
def plot_heat_with_errors():
    alpha = 0.01
    t_curve = np.linspace(0, 50, 500)
    y_ref = np.exp(-alpha*np.pi**2*t_curve)*np.sin(np.pi*0.5)
    traj = parse_trajectory("/tmp/plot_heat_traj.csv", 2)
    if traj is None: return
    t,y = traj[:,0], traj[:,1]
    y_exact = np.exp(-alpha*np.pi**2*t)*np.sin(np.pi*0.5)
    err = np.abs(y - y_exact)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(t_curve, y_ref, 'k-', linewidth=2, label='Analytical')
    ax.errorbar(t,y,yerr=err,fmt='o-',color='red',capsize=3,markersize=3,label='BDF (MOL N=100) with error')
    ax.set_xlabel('t'); ax.set_ylabel('$u(0.5,t)$')
    ax.set_title('Heat Equation — BDF with numerical error bars')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"heat_curve.png"),dpi=150); plt.close(fig)

def plot_oscillator():
    traj = parse_trajectory("/tmp/plot_osc_traj.csv", 3)
    if traj is None: return
    t,y1,y2 = traj[:,0],traj[:,1],traj[:,2]
    t_ref = np.linspace(0,t[-1],500)
    fig,ax = plt.subplots(figsize=(8,4))
    ax.plot(t_ref,np.cos(t_ref),'k--',label='Analytical cos(t)',alpha=0.6)
    ax.plot(t,y1,'o-',label='BDF y1',markersize=3,color='red')
    ax.set_xlabel('t'); ax.set_ylabel('y1')
    ax.set_title('Harmonic oscillator — BDF vs analytical')
    ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,'oscillator_curve.png'),dpi=150); plt.close(fig)

# Add call
try:
    pass
except: pass
