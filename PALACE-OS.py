import json
import os
import sys
import time
import random
import shlex
import re
import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog, colorchooser
import subprocess
import platform

FS_FILENAME = "palaceos_fs.json"
BOOT_MSG = [
    "PALACE-OS SIMULATOR v0.1 ",
    "Single-user, single-process. Behold the simple glory. /commands on the right    =====║========>",
]
PROMPT = "PALACE> "

def load_fs():
    if os.path.exists(FS_FILENAME):
        try:
            with open(FS_FILENAME, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "files" in data:
                    return data
        except Exception:
            pass
    return {"files": {}, "meta": {"created": time.time()}}

def save_fs(fs):
    with open(FS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(fs, f, indent=2, ensure_ascii=False)


TOKEN_RE = re.compile(r'''
    "(?:[^"\\]|\\.)*"   |   # double quoted string
    [A-Za-z_]\w*        |   # identifiers
    \d+                 |   # integers
    [\+\-\*/%\(\),:=<>] |   # operators and punctuation
    <=|>=|<>             # double-char operators
''', re.VERBOSE)

def tokenize(line):
    parts = TOKEN_RE.findall(line)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


class BasicProgram:
    def __init__(self, source_text, console_print):
        self.console_print = console_print
        self.lines = {}
        for raw in source_text.splitlines():
            s = raw.strip()
            if not s:
                continue
            m = re.match(r'^(\d+)\s+(.*)$', s)
            if m:
                ln = int(m.group(1))
                code = m.group(2).strip()
                self.lines[ln] = code
            else:
                ln = max(self.lines.keys(), default=0) + 10
                self.lines[ln] = s
        self.ordered = sorted(self.lines.items())
        self.line_numbers = [ln for ln, _ in self.ordered]

    def run(self):
        env = {}
        ip_index = 0

        def eval_expr(tokens):
            out_q = []
            op_s = []
            prec = {'+':1, '-':1, '*':2, '/':2, '%':2}
            def apply_op():
                op = op_s.pop()
                if op == '(':
                    return
                b = out_q.pop()
                a = out_q.pop()
                if isinstance(a, str) or isinstance(b, str):
                    if op == '+':
                        out_q.append(str(a) + str(b))
                    else:
                        raise ValueError("Invalid string operation")
                else:
                    if op == '+': out_q.append(a + b)
                    elif op == '-': out_q.append(a - b)
                    elif op == '*': out_q.append(a * b)
                    elif op == '/': out_q.append(a // b if b != 0 else 0)
                    elif op == '%': out_q.append(a % b)
            i = 0
            while i < len(tokens):
                t = tokens[i]
                if t.startswith('"') and t.endswith('"'):
                    out_q.append(bytes(t[1:-1], "utf-8").decode("unicode_escape"))
                elif re.fullmatch(r'\d+', t):
                    out_q.append(int(t))
                elif re.fullmatch(r'[A-Za-z_]\w*', t):
                    out_q.append(env.get(t, 0))
                elif t == '(':
                    op_s.append('(')
                elif t == ')':
                    while op_s and op_s[-1] != '(':
                        apply_op()
                    if op_s and op_s[-1] == '(':
                        op_s.pop()
                elif t in prec:
                    while op_s and op_s[-1] != '(' and prec.get(op_s[-1],0) >= prec[t]:
                        apply_op()
                    op_s.append(t)
                i += 1
            while op_s:
                apply_op()
            if not out_q:
                return 0
            return out_q[-1]

        def find_line_index_for(lineno):
            if lineno in self.line_numbers:
                return self.line_numbers.index(lineno)
            return None

        while 0 <= ip_index < len(self.line_numbers):
            lineno = self.line_numbers[ip_index]
            stmt = self.lines[lineno].strip()
            if stmt.upper().startswith("REM"):
                ip_index += 1
                continue
            toks = tokenize(stmt)
            if not toks:
                ip_index += 1; continue
            head = toks[0].upper()
            try:
                if head == "PRINT":
                    rest = stmt[len("PRINT"):].strip()
                    parts = []
                    cur = ''
                    inq = False
                    for ch in rest:
                        if ch == '"' and not (cur and cur[-1] == '\\'):
                            inq = not inq
                            cur += ch
                        elif ch == ',' and not inq:
                            parts.append(cur.strip()); cur = ''
                        else:
                            cur += ch
                    if cur.strip():
                        parts.append(cur.strip())
                    out = []
                    for p in parts:
                        p_toks = tokenize(p)
                        val = eval_expr(p_toks)
                        out.append(str(val))
                    self.console_print(" ".join(out))
                    ip_index += 1
                elif head == "LET":
                    m = re.match(r'LET\s+([A-Za-z_]\w*)\s*=\s*(.*)$', stmt, re.I)
                    if not m:
                        raise ValueError("Bad LET syntax")
                    var = m.group(1)
                    expr = m.group(2)
                    val = eval_expr(tokenize(expr))
                    env[var] = val
                    ip_index += 1
                elif head == "INPUT":
                    if len(toks) < 2:
                        raise ValueError("Bad INPUT")
                    var = toks[1]
                    v = simpledialog.askstring("INPUT", f"? ", parent=None)
                    try:
                        vv = int(v)
                    except Exception:
                        vv = v
                    env[var] = vv
                    ip_index += 1
                elif head == "GOTO":
                    if len(toks) < 2 or not re.fullmatch(r'\d+', toks[1]):
                        raise ValueError("Bad GOTO")
                    target = int(toks[1])
                    idx = find_line_index_for(target)
                    if idx is None:
                        raise ValueError(f"Line {target} not found")
                    ip_index = idx
                elif head == "IF":
                    try:
                        then_pos = [i for i,t in enumerate(toks) if t.upper() == "THEN"][0]
                    except IndexError:
                        raise ValueError("Missing THEN")
                    cmp_ops = ['=', '<>', '<=', '>=', '<', '>']
                    cmp_pos = None
                    for i,t in enumerate(toks[1:], start=1):
                        if t in cmp_ops:
                            cmp_pos = i
                            break
                    if cmp_pos is None:
                        raise ValueError("Missing comparison")
                    left_t = toks[1:cmp_pos]
                    op = toks[cmp_pos]
                    right_t = toks[cmp_pos+1:then_pos]
                    target = int(toks[then_pos+1]) if then_pos+1 < len(toks) else None
                    if target is None:
                        raise ValueError("Missing target line")
                    lv = eval_expr(left_t)
                    rv = eval_expr(right_t)
                    cond = False
                    if op == '=': cond = (lv == rv)
                    elif op == '<>': cond = (lv != rv)
                    elif op == '<': cond = (lv < rv)
                    elif op == '<=': cond = (lv <= rv)
                    elif op == '>': cond = (lv > rv)
                    elif op == '>=': cond = (lv >= rv)
                    if cond:
                        idx = find_line_index_for(target)
                        if idx is None:
                            raise ValueError(f"Line {target} not found")
                        ip_index = idx
                    else:
                        ip_index += 1
                elif head == "END":
                    break
                else:
                    m = re.match(r'([A-Za-z_]\w*)\s*=\s*(.*)', stmt)
                    if m:
                        var = m.group(1)
                        val = eval_expr(tokenize(m.group(2)))
                        env[var] = val
                        ip_index += 1
                    else:
                        self.console_print(f"(?) Unknown statement at {lineno}: {stmt}")
                        ip_index += 1
            except Exception as e:
                self.console_print("Runtime error: " + str(e))
                break


ORACLE_LINES = [
    "THE PALACE OF THE KINGDOM IS CREATED.",
    "THE KNIGHTS ARE DANCING IN THE BATTLEFIELD.",
    "THE KINGS MESSANGER SPEAKS IN LOOPS.",
    "YOU HAVE FOUND THE SIMPLE TRUTH: KEEP IT SMALL.",
    "REMEMBER: SAVE EARLY, SAVE OFTEN.",
]


def launch_tiny_doom():
    win = tk.Toplevel()
    win.title("Tiny DOOM (Tkinter Edition)")
    WIDTH, HEIGHT = 640, 480
    win.geometry(f"{WIDTH}x{HEIGHT}")
    canvas = tk.Canvas(win, width=WIDTH, height=HEIGHT, bg="black")
    canvas.pack()
    game_map = [
        [1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,1],
        [1,0,1,0,1,0,0,1],
        [1,0,0,0,0,1,0,1],
        [1,0,1,0,0,0,0,1],
        [1,0,0,0,1,0,0,1],
        [1,1,1,1,1,1,1,1]
    ]
    MAP_W = len(game_map[0])
    MAP_H = len(game_map)
    TILE = 64
    px, py = 3.5 * TILE, 3.5 * TILE
    angle = 0
    fov = math.radians(60)
    depth = 8 * TILE

    def draw_frame():
        nonlocal angle
        canvas.delete("all")
        for x in range(0, WIDTH, 2):
            ray_angle = angle - fov/2 + (x / WIDTH) * fov
            dist = 0
            hit = False
            eye_x = math.cos(ray_angle)
            eye_y = math.sin(ray_angle)
            while not hit and dist < depth:
                dist += 2
                test_x = int((px + eye_x * dist) / TILE)
                test_y = int((py + eye_y * dist) / TILE)
                if test_x < 0 or test_x >= MAP_W or test_y < 0 or test_y >= MAP_H:
                    hit = True
                    dist = depth
                else:
                    if game_map[test_y][test_x] == 1:
                        hit = True
            if dist > 0:
                wall_h = int(HEIGHT / (dist / 100))
                shade = max(0, min(255, int(255 - dist / 5)))
                color = f"#{shade:02x}{0:02x}{0:02x}"
                canvas.create_line(
                    x, (HEIGHT - wall_h)//2,
                    x, (HEIGHT + wall_h)//2,
                    fill=color
                )
        win.after(30, draw_frame)

    def move_forward(event=None):
        nonlocal px, py
        nx = px + math.cos(angle) * 10
        ny = py + math.sin(angle) * 10
        if game_map[int(ny/TILE)][int(nx/TILE)] == 0:
            px, py = nx, ny

    def move_backward(event=None):
        nonlocal px, py
        nx = px - math.cos(angle) * 10
        ny = py - math.sin(angle) * 10
        if game_map[int(ny/TILE)][int(nx/TILE)] == 0:
            px, py = nx, ny

    def turn_left(event=None):
        nonlocal angle
        angle -= 0.1

    def turn_right(event=None):
        nonlocal angle
        angle += 0.1

    win.bind("<w>", move_forward)
    win.bind("<s>", move_backward)
    win.bind("<a>", turn_left)
    win.bind("<d>", turn_right)
    win.bind("<Up>", move_forward)
    win.bind("<Down>", move_backward)
    win.bind("<Left>", turn_left)
    win.bind("<Right>", turn_right)
    draw_frame()


class Sandbox:
    
    EMPTY = 0
    WALL = 1
    SAND = 2
    WATER = 3
    FIRE = 4
    PLANT = 5
    SMOKE = 6
    OIL = 7
    GAS = 8
    
    MAT_INFO = {
        EMPTY: ("#000000", "Empty"),
        WALL: ("#7f7f7f", "Wall"),
        SAND: ("#d3b56d", "Sand"),
        WATER: ("#4aa3ff", "Water"),
        FIRE: ("#ff4b2b", "Fire"),
        PLANT: ("#3fbf3f", "Plant"),
        SMOKE: ("#aaaaaa", "Smoke"),
        OIL: ("#2b2b2b", "Oil"),
        GAS: ("#ffd27f", "Gas"),
    }

    def __init__(self, parent):
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Sandboxels (Sandbox)")
        
        self.CW = 4  
        self.GW = 200  
        self.GH = 120  
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        maxw = min(int(sw * 0.9), self.GW * self.CW)
        maxh = min(int(sh * 0.8), self.GH * self.CW)
        self.canvas_w = maxw
        self.canvas_h = maxh
        
        self.cols = self.canvas_w // self.CW
        self.rows = self.canvas_h // self.CW
        
        self.cols = max(60, min(self.cols, 300))
        self.rows = max(40, min(self.rows, 200))
        self.CW = max(2, self.canvas_w // self.cols)
        self.canvas_w = self.CW * self.cols
        self.canvas_h = self.CW * self.rows

        
        self.grid = [[self.EMPTY for _ in range(self.cols)] for __ in range(self.rows)]
        self.next_grid = [[self.EMPTY for _ in range(self.cols)] for __ in range(self.rows)]

        
        self.brush = self.SAND
        self.brush_size = 4
        self.paused = False
        self.speed = 1  
        self.gravity = True
        self.wind = 0  
        self.show_grid = False

        
        self.unlocked_oil = True
        self.unlocked_gas = True
        self.unlocked_plant = True
        self.unlocked_fire = True
        self.unlocked_smoke = True

        
        self._build_ui()
        self._bind_events()

       
        self.running = True
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    def _build_ui(self):
        top = tk.Frame(self.win)
        top.pack(side="top", fill="x")
        
        mats_frame = tk.Frame(top)
        mats_frame.pack(side="left", padx=6, pady=6)
        for mid in [self.SAND, self.WATER, self.WALL, self.FIRE, self.PLANT, self.OIL, self.GAS, self.SMOKE]:
            color, name = self.MAT_INFO[mid]
            btn = tk.Button(mats_frame, text=name, width=8, bg=color, command=lambda m=mid: self.set_brush(m))
            btn.pack(side="left", padx=2)

        tools_frame = tk.Frame(top)
        tools_frame.pack(side="left", padx=6)
        tk.Label(tools_frame, text="Brush").pack(anchor='w')
        self.brush_size_scale = tk.Scale(tools_frame, from_=1, to=20, orient="horizontal", command=self._set_brush_size)
        self.brush_size_scale.set(self.brush_size)
        self.brush_size_scale.pack()

        tk.Button(tools_frame, text="Eraser", command=lambda: self.set_brush(self.EMPTY)).pack(fill='x', pady=3)
        tk.Button(tools_frame, text="Fill (bucket)", command=self._bucket_mode).pack(fill='x', pady=3)
        tk.Button(tools_frame, text="Clear", command=self._clear_grid).pack(fill='x', pady=3)

        sim_frame = tk.Frame(top)
        sim_frame.pack(side="left", padx=6)
        self.pause_btn = tk.Button(sim_frame, text="Pause", command=self._toggle_pause)
        self.pause_btn.pack(fill='x')
        tk.Button(sim_frame, text="Randomize Sand", command=self._randomize_sand).pack(fill='x', pady=2)
        tk.Button(sim_frame, text="Spawn Mountain", command=self._spawn_mountain).pack(fill='x', pady=2)
        
        tk.Label(sim_frame, text="Speed").pack()
        self.speed_scale = tk.Scale(sim_frame, from_=1, to=6, orient="horizontal", command=self._set_speed)
        self.speed_scale.set(self.speed)
        self.speed_scale.pack()

        
        ctrl_frame = tk.Frame(top)
        ctrl_frame.pack(side="left", padx=6)
        self.gravity_var = tk.IntVar(value=1)
        tk.Checkbutton(ctrl_frame, text="Gravity", variable=self.gravity_var, command=self._toggle_gravity).pack(anchor='w')
        tk.Label(ctrl_frame, text="Wind").pack()
        self.wind_var = tk.IntVar(value=0)
        wind_box = tk.Frame(ctrl_frame)
        wind_box.pack()
        tk.Button(wind_box, text="←", command=lambda: self._set_wind(-1)).pack(side='left')
        tk.Button(wind_box, text="0", command=lambda: self._set_wind(0)).pack(side='left')
        tk.Button(wind_box, text="→", command=lambda: self._set_wind(1)).pack(side='left')

        save_frame = tk.Frame(top)
        save_frame.pack(side="right", padx=6)
        tk.Button(save_frame, text="Save Snapshot", command=self._save_snapshot).pack(fill='x', pady=2)
        tk.Button(save_frame, text="Load Snapshot", command=self._load_snapshot).pack(fill='x', pady=2)
        tk.Button(save_frame, text="Close", command=self._on_close).pack(fill='x', pady=2)

        
        self.canvas = tk.Canvas(self.win, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.pack(padx=6, pady=6)
        
        self.rects = [[None for _ in range(self.cols)] for __ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * self.CW
                y1 = r * self.CW
                x2 = x1 + self.CW
                y2 = y1 + self.CW
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline="" if not self.show_grid else "#222222", fill=self.MAT_INFO[self.EMPTY][0])
                self.rects[r][c] = rect

    def _bind_events(self):
        
        self.canvas.bind("<B1-Motion>", self._on_paint)
        self.canvas.bind("<Button-1>", self._on_paint)
        
        self.canvas.bind("<Button-3>", self._on_right)
        
        self.win.bind("<space>", lambda e: self._toggle_pause())
        self.win.bind("c", lambda e: self._clear_grid())
        self.win.bind("s", lambda e: self._save_snapshot())

    
    def set_brush(self, m):
        self.brush = m

    def _set_brush_size(self, v):
        try:
            self.brush_size = int(v)
        except Exception:
            pass

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")

    def _set_speed(self, v):
        try:
            self.speed = int(v)
        except Exception:
            pass

    def _toggle_gravity(self):
        self.gravity = bool(self.gravity_var.get())

    def _set_wind(self, w):
        self.wind = int(w)
        self.wind_var.set(self.wind)

    def _randomize_sand(self):
        for r in range(self.rows//2):
            for c in range(self.cols):
                if random.random() < 0.08:
                    self.grid[r][c] = self.SAND
        self._render_all()

    def _spawn_mountain(self):
        
        mid = self.cols // 2
        height = min(self.rows // 2, 40)
        for i in range(height):
            for c in range(mid - i, mid + i):
                rr = self.rows - height + i
                if 0 <= rr < self.rows and 0 <= c < self.cols:
                    self.grid[rr][c] = self.WALL
        self._render_all()

    def _bucket_mode(self):
        
        messagebox.showinfo("Bucket", "Right-click on canvas to bucket-fill with current material.")
        

    def _clear_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = self.EMPTY
        self._render_all()

    def _on_paint(self, event):
        x = event.x
        y = event.y
        c = x // self.CW
        r = y // self.CW
        bs = self.brush_size
        for rr in range(r - bs//2, r + bs//2 + 1):
            if 0 <= rr < self.rows:
                for cc in range(c - bs//2, c + bs//2 + 1):
                    if 0 <= cc < self.cols:
                        self.grid[rr][cc] = self.brush
        self._render_region(max(0, r - bs), max(0, c - bs), min(self.rows-1, r + bs), min(self.cols-1, c + bs))

    def _on_right(self, event):
        
        x = event.x; y = event.y
        c = x // self.CW; r = y // self.CW
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
        target = self.grid[r][c]
        replacement = self.brush
        if target == replacement:
            return
        
        stack = [(r, c)]
        while stack:
            rr, cc = stack.pop()
            if 0 <= rr < self.rows and 0 <= cc < self.cols and self.grid[rr][cc] == target:
                self.grid[rr][cc] = replacement
                stack.append((rr+1, cc))
                stack.append((rr-1, cc))
                stack.append((rr, cc+1))
                stack.append((rr, cc-1))
        self._render_all()

    def _save_snapshot(self):
        fname = filedialog.asksaveasfilename(defaultextension=".sbs", filetypes=[("Sandbox snapshot", "*.sbs"), ("JSON", "*.json")])
        if not fname:
            return
        try:
            data = {
                "rows": self.rows,
                "cols": self.cols,
                "grid": self.grid
            }
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(data, f)
            messagebox.showinfo("Saved", "Snapshot saved.")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _load_snapshot(self):
        fname = filedialog.askopenfilename(filetypes=[("Sandbox snapshot", "*.sbs;*.json"), ("All", "*.*")])
        if not fname:
            return
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "grid" in data and isinstance(data["grid"], list):
                self.grid = data["grid"]
                
                self.rows = len(self.grid)
                self.cols = len(self.grid[0]) if self.rows else 0
                
                self._rebuild_canvas()
                self._render_all()
                messagebox.showinfo("Loaded", "Snapshot loaded.")
            else:
                messagebox.showerror("Load error", "Invalid snapshot.")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _rebuild_canvas(self):
        
        self.canvas.config(width=self.cols * self.CW, height=self.rows * self.CW)
        for r in range(self.rows):
            for c in range(self.cols):
                if r >= len(self.rects) or c >= len(self.rects[r]):
                    pass
        
        self.canvas.delete("all")
        self.rects = [[None for _ in range(self.cols)] for __ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * self.CW
                y1 = r * self.CW
                x2 = x1 + self.CW
                y2 = y1 + self.CW
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline="" if not self.show_grid else "#222222", fill=self.MAT_INFO[self.EMPTY][0])
                self.rects[r][c] = rect

    def _render_region(self, r1, c1, r2, c2):
        for r in range(r1, r2+1):
            for c in range(c1, c2+1):
                mat = self.grid[r][c]
                color = self.MAT_INFO.get(mat, ("#000000",))[0]
                self.canvas.itemconfig(self.rects[r][c], fill=color)

    def _render_all(self):
        for r in range(self.rows):
            for c in range(self.cols):
                mat = self.grid[r][c]
                color = self.MAT_INFO.get(mat, ("#000000",))[0]
                self.canvas.itemconfig(self.rects[r][c], fill=color)

    def _tick(self):
        if not self.running:
            return
        if not self.paused:
            for _ in range(self.speed):
                self._update_step()
            self._render_all()
        self.win.after(30, self._tick)

    def _update_step(self):
        
        rows = self.rows
        cols = self.cols
        g = 1 if self.gravity else 0
        
        for r in range(rows):
            self.next_grid[r][:] = self.grid[r][:]
        
        for r in range(rows-1, -1, -1):
            for c in range(cols):
                mat = self.grid[r][c]
                if mat == self.SAND:
                    
                    nr = r + 1
                    if nr < rows:
                        below = self.grid[nr][c]
                        if below in (self.EMPTY, self.WATER, self.GAS):
                            
                            self.next_grid[nr][c] = self.SAND
                            self.next_grid[r][c] = below if below != self.GAS else self.EMPTY
                        else:
                            
                            dirs = [-1,1]
                            if self.wind != 0:
                                dirs = [self.wind, -self.wind]
                            random.shuffle(dirs)
                            moved = False
                            for d in dirs:
                                nc = c + d
                                if 0 <= nc < cols and self.grid[r+1][nc] in (self.EMPTY, self.WATER, self.GAS):
                                    self.next_grid[r+1][nc] = self.SAND
                                    self.next_grid[r][c] = self.grid[r+1][nc] if self.grid[r+1][nc] != self.GAS else self.EMPTY
                                    moved = True
                                    break
                            if not moved:
                                
                                pass
                elif mat == self.WATER:
                    
                    nr = r + 1
                    if nr < rows and self.grid[nr][c] == self.EMPTY:
                        self.next_grid[nr][c] = self.WATER
                        self.next_grid[r][c] = self.EMPTY
                    else:
                        
                        dirs = [1, -1]
                        if self.wind != 0:
                            dirs = [self.wind, -self.wind]
                        random.shuffle(dirs)
                        moved = False
                        for d in dirs:
                            nc = c + d
                            if 0 <= nc < cols and self.grid[r][nc] == self.EMPTY:
                                self.next_grid[r][nc] = self.WATER
                                self.next_grid[r][c] = self.EMPTY
                                moved = True
                                break
                        
                elif mat == self.FIRE:
                    
                    burned = False
                    for dr in (-1,0,1):
                        for dc in (-1,0,1):
                            rr = r + dr; cc = c + dc
                            if 0 <= rr < rows and 0 <= cc < cols:
                                target = self.grid[rr][cc]
                                if target == self.PLANT or target == self.OIL:
                                    self.next_grid[rr][cc] = self.FIRE
                                    burned = True
                    
                    if random.random() < 0.02:
                        self.next_grid[r][c] = self.SMOKE if self.unlocked_smoke else self.EMPTY
                    else:
                        self.next_grid[r][c] = self.FIRE
                elif mat == self.PLANT:
                    
                    if random.random() < 0.01:
                        nr = r - 1
                        if nr >= 0 and self.grid[nr][c] == self.EMPTY:
                            self.next_grid[nr][c] = self.PLANT
                elif mat == self.SMOKE:
                    
                    nr = r - 1
                    if nr >= 0 and self.grid[nr][c] == self.EMPTY:
                        self.next_grid[nr][c] = self.SMOKE
                        self.next_grid[r][c] = self.EMPTY
                    else:
                        
                        if random.random() < 0.01:
                            self.next_grid[r][c] = self.EMPTY
                elif mat == self.OIL:
                    
                    nr = r + 1
                    if nr < rows and self.grid[nr][c] == self.EMPTY:
                        self.next_grid[nr][c] = self.OIL
                        self.next_grid[r][c] = self.EMPTY
                    else:
                        dirs = [1, -1]
                        random.shuffle(dirs)
                        for d in dirs:
                            nc = c + d
                            if 0 <= nc < cols and self.grid[r][nc] == self.EMPTY:
                                self.next_grid[r][nc] = self.OIL
                                self.next_grid[r][c] = self.EMPTY
                                break
                elif mat == self.GAS:
                    
                    nr = r - 1
                    nc = c + self.wind
                    if nr >= 0 and self.grid[nr][c] == self.EMPTY:
                        self.next_grid[nr][c] = self.GAS
                        self.next_grid[r][c] = self.EMPTY
                    elif 0 <= nc < cols and self.grid[r][nc] == self.EMPTY:
                        self.next_grid[r][nc] = self.GAS
                        self.next_grid[r][c] = self.EMPTY
                    else:
                        
                        pass
        
        self.grid, self.next_grid = self.next_grid, self.grid
        
        for r in range(rows):
            for c in range(cols):
                self.next_grid[r][c] = self.grid[r][c]

    def _on_close(self):
        self.running = False
        try:
            self.win.destroy()
        except Exception:
            pass



class SimplePaint:
    def __init__(self, root):
        self.root = root
        root.title("Simple Paint (Tkinter)")
        root.geometry("900x650")
        root.minsize(600, 400)
        self.bg_color = "white"
        self.brush_color = "#000000"
        self.brush_size = 5
        self.eraser_on = False
        self.strokes = []
        self.current_stroke = []
        self._make_toolbar()
        self._make_canvas()
        self._bind_events()

    def _make_toolbar(self):
        toolbar = tk.Frame(self.root, padx=5, pady=5)
        toolbar.pack(side="left", fill="y")
        self.color_btn = tk.Button(toolbar, text="Color", command=self.choose_color, width=12)
        self.color_btn.pack(pady=(0, 6))
        self.color_display = tk.Canvas(toolbar, width=36, height=24, bg=self.brush_color, bd=1, relief="sunken")
        self.color_display.pack(pady=(0, 10))
        tk.Label(toolbar, text="Brush size:").pack(anchor="w")
        self.size_slider = tk.Scale(toolbar, from_=1, to=50, orient="horizontal", command=self._update_size)
        self.size_slider.set(self.brush_size)
        self.size_slider.pack(fill="x", pady=(0,10))
        self.eraser_btn = tk.Button(toolbar, text="Eraser", command=self.toggle_eraser, width=12)
        self.eraser_btn.pack(pady=(0,6))
        self.undo_btn = tk.Button(toolbar, text="Undo", command=self.undo, width=12)
        self.undo_btn.pack(pady=(0,6))
        self.clear_btn = tk.Button(toolbar, text="Clear", command=self.clear_canvas, width=12)
        self.clear_btn.pack(pady=(0,10))
        self.save_btn = tk.Button(toolbar, text="Save (.ps)", command=self.save_canvas, width=12)
        self.save_btn.pack(pady=(0,6))
        tk.Label(toolbar, text="\nTip:\n- Draw with mouse\n- Eraser uses background color\n- Undo removes last stroke").pack(pady=(10,0))

    def _make_canvas(self):
        canvas_frame = tk.Frame(self.root, bg="gray")
        canvas_frame.pack(side="right", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg=self.bg_color, cursor="cross")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-s>", lambda e: self.save_canvas())

    def choose_color(self):
        color = colorchooser.askcolor(initialcolor=self.brush_color, title="Choose brush color")
        if color and color[1]:
            self.brush_color = color[1]
            self.color_display.configure(bg=self.brush_color)
            self.eraser_on = False
            self.eraser_btn.config(relief="raised")

    def _update_size(self, val):
        self.brush_size = int(val)

    def toggle_eraser(self):
        self.eraser_on = not self.eraser_on
        self.eraser_btn.config(relief='sunken' if self.eraser_on else 'raised')

    def on_button_press(self, event):
        self.current_stroke = []
        self.lastx, self.lasty = event.x, event.y

    def on_paint(self, event):
        x, y = event.x, event.y
        color = self.bg_color if self.eraser_on else self.brush_color
        line_id = self.canvas.create_line(
            self.lastx, self.lasty, x, y,
            width=self.brush_size, capstyle="round", smooth=True,
            fill=color
        )
        self.current_stroke.append(line_id)
        self.lastx, self.lasty = x, y

    def on_button_release(self, event):
        if self.current_stroke:
            self.strokes.append(self.current_stroke)

    def undo(self):
        if not self.strokes:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return
        for item in self.strokes.pop():
            self.canvas.delete(item)

    def clear_canvas(self):
        if messagebox.askyesno("Clear", "Clear the entire canvas?"):
            self.canvas.delete("all")
            self.strokes.clear()

    def save_canvas(self):
        filename = filedialog.asksaveasfilename(defaultextension=".ps", filetypes=[("PostScript file", "*.ps")])
        if not filename:
            return
        try:
            self.root.update()
            ps = self.canvas.postscript(colormode='color')
            with open(filename, "w", encoding="utf-8") as f:
                f.write(ps)
            messagebox.showinfo("Saved", "Canvas saved successfully.")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


class PalaceGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Safe golden palette (works cross-platform)
        try:
            self.tk.call(
                'tk_setPalette',
                background='goldenrod2',
                foreground='black',
                activeBackground='goldenrod3',
                activeForeground='black',
                highlightColor='black'
            )
        except Exception:
            pass

        # Additionally set some explicit widget defaults so theme is obviously golden
        # without changing any functionality.
        self._gold_bg = 'goldenrod2'
        self._gold_btn_bg = 'goldenrod3'
        self._gold_active = 'goldenrod4'
        self._gold_fg = 'black'

        self.title("PALACE-OS (GUI)")
        self.geometry("1000x600")
        self.fs = load_fs()
        self.current_user = None
        self._ensure_user_storage()
        try:
            self.login_required = self.fs.get('meta', {}).get('settings', {}).get('login_on_startup', True)
        except Exception:
            self.login_required = True

        # build usual widgets but we will hide them and boot to desktop
        self._make_widgets()
        self.boot_sequence()

        # launch desktop immediately on start
        self.launch_desktop()

    def _make_widgets(self):
        # left frame (give golden background)
        left = tk.Frame(self, bg=self._gold_bg)
        left.pack(side="left", fill="both", expand=True)
        # console should remain dark for readability
        self.console = tk.Text(left, bg='black', fg='white', insertbackground='white')
        self.console.pack(fill='both', expand=True)
        self.console.configure(state='disabled')
        entry_frame = tk.Frame(left, bg=self._gold_bg)
        entry_frame.pack(fill='x')
        # entry with golden background
        self.cmd_entry = tk.Entry(entry_frame, bg='goldenrod1', fg=self._gold_fg, insertbackground=self._gold_fg)
        self.cmd_entry.pack(side='left', fill='x', expand=True)
        self.cmd_entry.bind('<Return>', lambda e: self._on_command())
        tk.Button(entry_frame, text='Run', command=self._on_command,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(side='left')
        # right frame
        right = tk.Frame(self, width=320, bg=self._gold_bg)
        right.pack(side='right', fill='y')
        tk.Label(right, text='Disk Files', bg=self._gold_bg, fg=self._gold_fg).pack()
        # file list with golden background
        self.file_list = tk.Listbox(right, bg='goldenrod1', fg=self._gold_fg)
        self.file_list.pack(fill='both', expand=True)
        self.file_list.bind('<Double-1>', lambda e: self.open_editor_selected())
        btn_frame = tk.Frame(right, bg=self._gold_bg)
        btn_frame.pack(fill='x')
        tk.Button(btn_frame, text='New', command=self.mk_file_gui,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(side='left', padx=2, pady=4)
        tk.Button(btn_frame, text='Edit', command=self.open_editor_selected,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(side='left', padx=2)
        tk.Button(btn_frame, text='Cat', command=self.cat_selected,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(side='left', padx=2)
        tk.Button(btn_frame, text='Run', command=self.run_selected,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(side='left', padx=2)
        tk.Button(btn_frame, text='Delete', command=self.rm_selected,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(side='left', padx=2)
        misc = tk.Frame(right, bg=self._gold_bg)
        misc.pack(fill='x', pady=6)
        tk.Button(misc, text='Save FS', command=self.save_fs_gui,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x')
        tk.Button(misc, text='Format FS', command=self.format_fs_gui,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x')
        tk.Button(misc, text='Oracle', command=self.oracle_gui,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x')
        tk.Button(misc, text='Doom', command=launch_tiny_doom,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x')
        tk.Button(misc, text='Art', command=self.launch_art,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x')
        tk.Button(misc, text='Tetris', command=self.launch_tetris,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x')
        # sandbox app button in sidebar
        tk.Button(misc, text='Sandboxels', command=self.launch_sandbox_app,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x')
        # Terminal button (opens an external system terminal)
        tk.Button(misc, text='Terminal', command=self.open_terminal,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x')
        tk.Button(misc, text='Exit', command=self.destroy,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x')
        tk.Button(misc, text='Version', command=lambda: self.console_print("This is version 1.2.1 of Palace OS"),
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x')
        self.refresh_file_list()

    def console_print(self, *args, sep=' ', end='\n'):
        s = sep.join(str(a) for a in args) + end
        self.console.configure(state='normal')
        self.console.insert('end', s)
        self.console.see('end')
        self.console.configure(state='disabled')

    def boot_sequence(self):
        self.console.configure(state='normal')
        self.console.delete('1.0', 'end')
        self.console.configure(state='disabled')
        for line in BOOT_MSG:
            self.console_print(line)
            self.update()
            time.sleep(0.12)
        self.console_print('\nBooting...')
        time.sleep(0.2)
        self.console_print('\nSuccesfully booted')
    def _desktop_update_clock(self):
        try:
            current = time.strftime("%a %b %d — %H:%M:%S", time.localtime())
            if not hasattr(self, 'clock_var'):
                self.clock_var = tk.StringVar()
            self.clock_var.set(current)
            # schedule
            self._clock_job = self.after(1000, self._desktop_update_clock)
        except Exception:
            pass



    def refresh_file_list(self):
        self.file_list.delete(0, 'end')
        for name in sorted(self.fs['files'].keys()):
            self.file_list.insert('end', name)

    def mk_file_gui(self):
        name = simpledialog.askstring('New file', 'Filename:')
        if not name:
            return
        if name in self.fs['files']:
            messagebox.showinfo('Info', 'File exists.')
            return
        self.fs['files'][name] = ''
        save_fs(self.fs)
        self.refresh_file_list()

    def open_editor_selected(self):
        sel = self.file_list.curselection()
        if not sel:
            messagebox.showinfo('Info', 'Select a file to edit.')
            return
        name = self.file_list.get(sel[0])
        self.open_editor(name)

    def open_editor(self, filename):
        content = self.fs['files'].get(filename, '')
        ed = tk.Toplevel(self)
        ed.title(f"Editor - {filename}")
        txt = tk.Text(ed)
        txt.pack(fill='both', expand=True)
        txt.insert('1.0', content)
        btns = tk.Frame(ed)
        btns.pack(fill='x')
        def do_save():
            self.fs['files'][filename] = txt.get('1.0', 'end').rstrip('\n')
            save_fs(self.fs)
            self.refresh_file_list()
            self.console_print(f"Saved '{filename}'.")
        def do_close():
            do_save()
            ed.destroy()
        tk.Button(btns, text='Save', command=do_save,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(side='left')
        tk.Button(btns, text='Close', command=do_close,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(side='left')

    def cat_selected(self):
        sel = self.file_list.curselection()
        if not sel:
            messagebox.showinfo('Info', 'Select a file to view.')
            return
        name = self.file_list.get(sel[0])
        txt = self.fs['files'].get(name, '')
        viewer = tk.Toplevel(self)
        viewer.title(f"{name}")
        t = tk.Text(viewer)
        t.pack(fill='both', expand=True)
        t.insert('1.0', txt)
        t.configure(state='disabled')

    def rm_selected(self):
        sel = self.file_list.curselection()
        if not sel:
            messagebox.showinfo('Info', 'Select a file to delete.')
            return
        name = self.file_list.get(sel[0])
        if messagebox.askyesno('Confirm', f'Delete {name}?'):
            del self.fs['files'][name]
            save_fs(self.fs)
            self.refresh_file_list()
            self.console_print(f"Removed '{name}'.")

    def save_fs_gui(self):
        save_fs(self.fs)
        self.console_print('Saved.')

    def format_fs_gui(self):
        if messagebox.askyesno('Format', "Format disk? This will erase all files."):
            if os.path.exists(FS_FILENAME):
                os.remove(FS_FILENAME)
            self.fs = load_fs()
            save_fs(self.fs)
            self.refresh_file_list()
            self.console_print('Formatted disk (FS reset).')

    def oracle_gui(self):
        self.console_print(random.choice(ORACLE_LINES))

    def run_selected(self):
        sel = self.file_list.curselection()
        if not sel:
            messagebox.showinfo('Info', 'Select a file to run.')
            return
        name = self.file_list.get(sel[0])
        content = self.fs['files'].get(name)
        if content is None:
            messagebox.showinfo('Info', 'No such file.')
            return
        prog = BasicProgram(content, self.console_print)
        prog.run()

    def launch_art(self):
        win = tk.Toplevel(self)
        SimplePaint(win)

    def open_terminal(self):
        system = platform.system()
        try:
            if system == "Windows":
                try:
                    os.startfile("cmd.exe")
                except Exception:
                    subprocess.Popen(["cmd.exe"])
            elif system == "Darwin":  
                subprocess.Popen(["open", "-a", "Terminal"])
            else:  
                for term in ["x-terminal-emulator", "gnome-terminal", "konsole", "xfce4-terminal", "xterm"]:
                    try:
                        subprocess.Popen([term])
                        return
                    except FileNotFoundError:
                        pass
                messagebox.showerror("Terminal Error", "No terminal emulator found.")
        except Exception as e:
            messagebox.showerror("Terminal Error", str(e))


    
    def open_calculator(self):
        win = tk.Toplevel(self)
        win.title("Calculator")
        entry = tk.Entry(win, width=20, font=("TkDefaultFont", 16))
        entry.grid(row=0, column=0, columnspan=4, padx=5, pady=5)

        def click(v):
            entry.insert("end", v)

        def clear():
            entry.delete(0, "end")

        def equal():
            try:
                res = eval(entry.get())
                entry.delete(0, "end")
                entry.insert("end", str(res))
            except Exception:
                entry.delete(0, "end")
                entry.insert("end", "Error")

        buttons = [
            '7','8','9','/',
            '4','5','6','*',
            '1','2','3','-',
            '0','.','=','+'
        ]
        r = 1; c = 0
        for b in buttons:
            if b == '=':
                cmd = equal
            else:
                cmd = (lambda v=b: click(v))
            tk.Button(win, text=b, width=5, height=2, command=cmd).grid(row=r, column=c, padx=2, pady=2)
            c += 1
            if c > 3:
                c = 0; r += 1
        tk.Button(win, text="Clear", width=22, command=clear).grid(row=r, column=0, columnspan=4, pady=5)


    
    # --- User accounts / Login system ---
    def _ensure_user_storage(self):
        # Ensure fs meta structure for users/settings
        meta = self.fs.get("meta", {})
        if "users" not in meta:
            meta["users"] = ["guest"]
        if "settings" not in meta:
            meta["settings"] = {"theme":"gold", "wallpaper_color": self._gold_bg, "login_on_startup": True}
        self.fs["meta"] = meta
        save_fs(self.fs)

    def show_login(self):
        """Modal login dialog. Returns True on success, False on cancel."""
        self._ensure_user_storage()
        users = self.fs["meta"]["users"]

        win = tk.Toplevel(self)
        win.title("Login - PALACE-OS")
        win.geometry("320x180")
        win.transient(self)
        win.grab_set()

        tk.Label(win, text="Select user to login", font=("TkDefaultFont", 12, "bold")).pack(pady=6)
        lb = tk.Listbox(win, height=5)
        for u in users:
            lb.insert("end", u)
        lb.pack(padx=10, fill='x')

        frame = tk.Frame(win)
        frame.pack(pady=8)
        username_var = tk.StringVar()

        def add_user():
            name = simpledialog.askstring("Add user", "Enter new username:", parent=win)
            if name:
                if name in users:
                    messagebox.showinfo("Users", "User exists.")
                else:
                    users.append(name)
                    self.fs["meta"]["users"] = users
                    save_fs(self.fs)
                    lb.insert("end", name)

        def remove_user():
            sel = lb.curselection()
            if not sel:
                return
            name = lb.get(sel[0])
            if messagebox.askyesno("Confirm", f"Delete user '{name}'?"):
                try:
                    users.remove(name)
                except Exception:
                    pass
                self.fs["meta"]["users"] = users
                save_fs(self.fs)
                lb.delete(sel[0])

        def do_login():
            sel = lb.curselection()
            if sel:
                name = lb.get(sel[0])
            else:
                name = simpledialog.askstring("Login", "Enter username:", parent=win)
                if not name:
                    return
            self.current_user = name
            win.destroy()

        btnf = tk.Frame(win)
        btnf.pack(pady=4)
        tk.Button(btnf, text="Add", command=add_user).pack(side='left', padx=4)
        tk.Button(btnf, text="Remove", command=remove_user).pack(side='left', padx=4)
        tk.Button(btnf, text="Login", command=do_login).pack(side='left', padx=4)
        tk.Button(btnf, text="Cancel", command=win.destroy).pack(side='left', padx=4)

        self.wait_window(win)
        return getattr(self, "current_user", None) is not None

    def open_users(self):
        """Open Users management window"""
        self._ensure_user_storage()
        win = tk.Toplevel(self)
        win.title("Users - PALACE-OS")
        win.geometry("360x260")
        win.configure(bg=self._gold_bg)

        users = self.fs["meta"]["users"]

        tk.Label(win, text="Users", bg=self._gold_bg, fg=self._gold_fg, font=("TkDefaultFont", 12, "bold")).pack(pady=6)
        lb = tk.Listbox(win)
        for u in users:
            lb.insert("end", u)
        lb.pack(fill='both', expand=True, padx=8, pady=6)

        def add_user():
            name = simpledialog.askstring("Add user", "Enter new username:", parent=win)
            if name:
                if name in users:
                    messagebox.showinfo("Users", "User exists.")
                else:
                    users.append(name)
                    self.fs["meta"]["users"] = users
                    save_fs(self.fs)
                    lb.insert("end", name)

        def delete_user():
            sel = lb.curselection()
            if not sel:
                return
            name = lb.get(sel[0])
            if messagebox.askyesno("Delete", f"Delete user '{name}'?"):
                users.remove(name)
                self.fs["meta"]["users"] = users
                save_fs(self.fs)
                lb.delete(sel[0])

        def switch_user():
            sel = lb.curselection()
            if not sel:
                messagebox.showinfo("Users", "Select a user to switch to.")
                return
            name = lb.get(sel[0])
            self.current_user = name
            messagebox.showinfo("Users", f"Switched to {name}")

        btnf = tk.Frame(win, bg=self._gold_bg)
        btnf.pack(pady=6)
        tk.Button(btnf, text="Add", command=add_user, bg=self._gold_btn_bg).pack(side='left', padx=4)
        tk.Button(btnf, text="Delete", command=delete_user, bg=self._gold_btn_bg).pack(side='left', padx=4)
        tk.Button(btnf, text="Switch", command=switch_user, bg=self._gold_btn_bg).pack(side='left', padx=4)

    # --- Settings app ---
    def open_settings(self):
        self._ensure_user_storage()
        win = tk.Toplevel(self)
        win.title("Settings - PALACE-OS")
        win.geometry("420x260")
        win.configure(bg=self._gold_bg)

        settings = self.fs["meta"].get("settings", {"theme":"gold", "wallpaper_color": self._gold_bg, "login_on_startup": True})

        tk.Label(win, text="Settings", bg=self._gold_bg, fg=self._gold_fg, font=("TkDefaultFont", 14, "bold")).pack(pady=6)

        # Theme toggle
        theme_var = tk.StringVar(value=settings.get("theme","gold"))
        def apply_theme():
            val = theme_var.get()
            settings["theme"] = val
            if val == "dark":
                self._gold_bg = "gray12"; self._gold_btn_bg = "gray22"; self._gold_active="gray30"; self._gold_fg="white"
            else:
                self._gold_bg = 'goldenrod2'; self._gold_btn_bg='goldenrod3'; self._gold_active='goldenrod4'; self._gold_fg='black'
            settings["wallpaper_color"] = self._gold_bg
            self.fs["meta"]["settings"] = settings
            save_fs(self.fs)
            messagebox.showinfo("Settings","Theme applied. Some changes may require restart.")

        fr = tk.Frame(win, bg=self._gold_bg)
        fr.pack(pady=6)
        tk.Radiobutton(fr, text="Gold Theme", variable=theme_var, value="gold", bg=self._gold_bg, fg=self._gold_fg).pack(anchor='w')
        tk.Radiobutton(fr, text="Dark Theme", variable=theme_var, value="dark", bg=self._gold_bg, fg=self._gold_fg).pack(anchor='w')
        tk.Button(win, text="Apply Theme", command=apply_theme, bg=self._gold_btn_bg).pack(pady=6)

        # Wallpaper color
        def pick_color():
            c = colorchooser.askcolor(title="Choose wallpaper color", initialcolor=settings.get("wallpaper_color", self._gold_bg))
            if c and c[1]:
                settings["wallpaper_color"] = c[1]
                self.fs["meta"]["settings"] = settings
                save_fs(self.fs)
                messagebox.showinfo("Settings","Wallpaper color saved. Press 'Show Console' then re-open desktop to see effect.")

        tk.Button(win, text="Pick Wallpaper Color", command=pick_color, bg=self._gold_btn_bg).pack(pady=4)

        # Login on startup toggle
        login_var = tk.BooleanVar(value=settings.get("login_on_startup", True))
        def toggle_login():
            settings["login_on_startup"] = login_var.get()
            self.fs["meta"]["settings"] = settings
            save_fs(self.fs)
            messagebox.showinfo("Settings","Saved.")
        tk.Checkbutton(win, text="Require login on startup", variable=login_var, bg=self._gold_bg, fg=self._gold_fg, command=toggle_login).pack(pady=6)

    # --- Simple Snake game ---

    def launch_tetris(self):
        """Tetris: reliable falling blocks and controls."""
        win = tk.Toplevel(self)
        win.title("Tetris")
        CELL = 24
        COLS = 10
        ROWS = 20
        WIDTH = COLS * CELL
        HEIGHT = ROWS * CELL
        win.geometry(f"{WIDTH}x{HEIGHT+40}")
        canvas = tk.Canvas(win, width=WIDTH, height=HEIGHT, bg="black")
        canvas.pack(fill='both', expand=True)

        SHAPES = [
            [[1,1,1,1]],
            [[1,1],[1,1]],
            [[0,1,0],[1,1,1]],
            [[1,0,0],[1,1,1]],
            [[0,0,1],[1,1,1]],
            [[1,1,0],[0,1,1]],
            [[0,1,1],[1,1,0]],
        ]

        import random
        grid = [[0]*COLS for _ in range(ROWS)]
        score = {'v':0}
        current = {'shape':None, 'r':0, 'c':COLS//2-1}
        running = {'v':True, 'drop_ms':600}

        def collides(r,c,shape):
            for i,row in enumerate(shape):
                for j,val in enumerate(row):
                    if val:
                        rr = r + i; cc = c + j
                        if rr<0 or cc<0 or cc>=COLS or rr>=ROWS: return True
                        if grid[rr][cc]: return True
            return False

        def rotate(shape):
            return [list(row) for row in zip(*shape[::-1])]

        def new_piece():
            s = random.choice(SHAPES)
            current['shape'] = [row[:] for row in s]
            current['r'] = 0
            current['c'] = COLS//2 - len(s[0])//2
            # if immediate collision => game over
            if collides(current['r'], current['c'], current['shape']):
                game_over()

        def lock_piece():
            for i,row in enumerate(current['shape']):
                for j,val in enumerate(row):
                    if val:
                        grid[current['r']+i][current['c']+j] = 1
            clear_lines()
            new_piece()

        def clear_lines():
            newg = []
            cleared = 0
            for row in grid:
                if all(row):
                    cleared += 1
                else:
                    newg.append(row)
            for _ in range(cleared):
                newg.insert(0, [0]*COLS)
            if cleared:
                score['v'] += cleared * 100
            for r in range(ROWS):
                grid[r] = newg[r]

        def draw():
            canvas.delete("all")
            for r in range(ROWS):
                for c in range(COLS):
                    if grid[r][c]:
                        x1 = c*CELL; y1 = r*CELL; x2 = x1+CELL; y2 = y1+CELL
                        canvas.create_rectangle(x1,y1,x2,y2, fill="cyan", outline="grey")
            if current['shape']:
                for i,row in enumerate(current['shape']):
                    for j,val in enumerate(row):
                        if val:
                            rr = current['r']+i; cc = current['c']+j
                            x1 = cc*CELL; y1 = rr*CELL; x2 = x1+CELL; y2 = y1+CELL
                            canvas.create_rectangle(x1,y1,x2,y2, fill="orange", outline="grey")
            canvas.create_text(6, HEIGHT+20, text=f"Score: {score['v']}", anchor="w", fill="white")

        def step():
            if not running['v']:
                return
            # ensure there is a piece
            if current['shape'] is None:
                new_piece()
            # try move down
            if not collides(current['r']+1, current['c'], current['shape']):
                current['r'] += 1
            else:
                lock_piece()
            draw()
            win.after(running['drop_ms'], step)

        def move(dx):
            if current['shape'] and not collides(current['r'], current['c']+dx, current['shape']):
                current['c'] += dx
                draw()

        def soft_drop(evt=None):
            if current['shape'] and not collides(current['r']+1, current['c'], current['shape']):
                current['r'] += 1
            else:
                lock_piece()
            draw()

        def hard_drop(evt=None):
            if current['shape']:
                while not collides(current['r']+1, current['c'], current['shape']):
                    current['r'] += 1
                lock_piece()
                draw()

        def rotate_piece(evt=None):
            if current['shape']:
                new = rotate(current['shape'])
                if not collides(current['r'], current['c'], new):
                    current['shape'] = new
                draw()

        def pause_toggle(evt=None):
            running['v'] = not running['v']
            if running['v']:
                step()

        def game_over():
            running['v'] = False
            try:
                messagebox.showinfo("Game Over", f"Game Over! Score: {score['v']}", parent=win)
            except Exception:
                messagebox.showinfo("Game Over", f"Game Over! Score: {score['v']}")
            try:
                win.destroy()
            except Exception:
                pass

        # key handling
        def _on_key(event):
            k = event.keysym
            if k in ("Left","a","A"):
                move(-1)
            elif k in ("Right","d","D"):
                move(1)
            elif k in ("Down","s","S"):
                soft_drop()
            elif k in ("Up","w","W"):
                rotate_piece()
            elif k == "space":
                hard_drop()
            elif k in ("p","P"):
                pause_toggle()

        def bind_keys():
            win.bind_all("<Key>", _on_key)

        def unbind_keys():
            try:
                win.unbind_all("<Key>")
            except Exception:
                pass

        win.protocol("WM_DELETE_WINDOW", lambda: (unbind_keys(), win.destroy()))
        # ensure focus and start
        def focus_and_start():
            try:
                win.focus_force()
                canvas.focus_set()
            except Exception:
                pass
            bind_keys()
            draw()
            step()

        win.after(80, focus_and_start)

    
    def launch_personalize(self):
        win = tk.Toplevel(self)
        win.title("Personalize")
        win.geometry("300x400")
        themes = {
            "Gold": ("#d4af37","#b8860b","white"),
            "Dark": ("#2b2b2b","#3c3c3c","white"),
            "Light A": ("gainsboro","lightgray","black"),
            "Light B": ("white","whitesmoke","black"),
            "Light C": ("#dbe9f6","#c5d8ee","black"),
            "Terminal A": ("black","#003300","#00ff00"),
            "Terminal B": ("black","#004400","#33ff33"),
            "Neon A": ("#00111a","#003355","#33ccff"),
            "Neon B": ("#002244","#004488","#00aaff")
        }
        import json, os
        lb = tk.Listbox(win)
        lb.pack(fill="both", expand=True)
        for t in themes: lb.insert("end", t)

        def apply_theme(evt=None):
            sel = lb.get("anchor")
            if sel in themes:
                bg, btn, fg = themes[sel]
                self._gold_bg = bg
                self._gold_btn_bg = btn
                self._gold_fg = fg
                self.configure(bg=bg)
                try:
                    self.desktop_frame.configure(bg=bg)
                except Exception:
                    pass
        lb.bind("<<ListboxSelect>>", apply_theme)

    def launch_desktop(self):
        """Open a desktop-like UI and hide the default panels."""
        
        for w in self.winfo_children():
            try:
                w.pack_forget()
            except Exception:
                try:
                    w.place_forget()
                except Exception:
                    pass

        
        self.desktop_frame = tk.Frame(self, bg=self._gold_bg)
        self.desktop_frame.pack(fill='both', expand=True)

        
        tk.Label(self.desktop_frame, text="PALACE DESKTOP", font=("TkDefaultFont", 24, "bold"),
                 bg=self._gold_bg, fg=self._gold_fg).pack(pady=8)

        # --- Digital Clock (auto timezone, padded top-right) ---
        self.clock_var = tk.StringVar()

        clock_label = tk.Label(
            self.desktop_frame,
            textvariable=self.clock_var,
            font=("TkDefaultFont", 14, "bold"),
            bg=self._gold_bg,
            fg=self._gold_fg
        )
        clock_label.place(relx=1.0, y=10, anchor="ne")  # slight padding

        # clock updater
        self._desktop_update_clock()


        icons_frame = tk.Frame(self.desktop_frame, bg=self._gold_bg)
        icons_frame.pack(expand=True, padx=20, pady=10)

        def make_icon(emoji, label, command):
            f = tk.Frame(icons_frame, bg=self._gold_bg, bd=0)
            btn = tk.Button(f, text=emoji + "\n" + label, font=("Segoe UI Emoji", 12), compound='top',
                            bg='goldenrod1', activebackground=self._gold_active, fg=self._gold_fg,
                            command=command, width=10, height=4)
            btn.pack()
            return f

        
        icons = [
            ("🎨", "Personalize", self.launch_personalize),
            ("🖥️", "Console", self._restore_ui),
            ("💾", "Files", lambda: self.open_file_manager_desktop()),
            ("🎨", "Art", self.launch_art),
            ("🕹️", "Tiny DOOM", launch_tiny_doom),
            ("🧪", "Sandboxels", self.launch_sandbox_app),
            ("🧱", "Tetris", self.launch_tetris),
            ("⌨️", "Terminal", self.open_terminal),
                        ("🧮", "Calc", self.open_calculator),
            ("👤", "Users", self.open_users),
            ("⚙️", "Settings", self.open_settings),
            ("❌", "Quit", self.destroy),
        ]

        
        cols = 4
        r = 0; c = 0
        for emo, lab, cmd in icons:
            f = make_icon(emo, lab, cmd)
            f.grid(row=r, column=c, padx=18, pady=18)
            c += 1
            if c >= cols:
                c = 0; r += 1

        
        dock = tk.Frame(self.desktop_frame, bg=self._gold_bg)
        dock.pack(side='bottom', pady=10)
        tk.Button(dock, text="Show Console", command=self._restore_ui,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(side='left', padx=6)
        tk.Label(dock, text="Tip: Click icons to launch apps", bg=self._gold_bg, fg=self._gold_fg).pack(side='left', padx=12)

    def _restore_ui(self):
        
        try:
            self.desktop_frame.destroy()
        except Exception:
            pass
        
        for w in self.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        
        self._make_widgets()
        
        try:
            self.deiconify()
            self.lift()
        except Exception:
            pass

    def open_file_manager_desktop(self, parent=None):
        """Small file manager window usable from the desktop icons."""
        win = tk.Toplevel(self)
        win.title("Files - Desktop Manager")
        win.geometry("600x400")
        win.configure(bg=self._gold_bg)
        left = tk.Frame(win, bg=self._gold_bg)
        left.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        lbl = tk.Label(left, text='Disk Files', bg=self._gold_bg, fg=self._gold_fg, font=("TkDefaultFont", 12, "bold"))
        lbl.pack(anchor='nw')
        file_list = tk.Listbox(left, bg='goldenrod1', fg=self._gold_fg)
        file_list.pack(fill='both', expand=True, pady=6)
        for name in sorted(self.fs['files'].keys()):
            file_list.insert('end', name)
        btns = tk.Frame(win, bg=self._gold_bg)
        btns.pack(side='right', fill='y', padx=8, pady=8)
        def refresh():
            file_list.delete(0, 'end')
            for n in sorted(self.fs['files'].keys()):
                file_list.insert('end', n)
        def open_edit():
            sel = file_list.curselection()
            if not sel:
                messagebox.showinfo("Info", "Select a file.")
                return
            self.open_editor(file_list.get(sel[0]))
        def open_cat():
            sel = file_list.curselection()
            if not sel:
                messagebox.showinfo("Info", "Select a file.")
                return
            name = file_list.get(sel[0])
            txt = self.fs['files'].get(name, '')
            viewer = tk.Toplevel(win)
            viewer.title(name)
            t = tk.Text(viewer)
            t.pack(fill='both', expand=True)
            t.insert('1.0', txt)
            t.configure(state='disabled')
        def run_file():
            sel = file_list.curselection()
            if not sel:
                messagebox.showinfo("Info", "Select a file.")
                return
            name = file_list.get(sel[0])
            content = self.fs['files'].get(name)
            if content is None:
                messagebox.showinfo("Info", "No such file.")
                return
            prog = BasicProgram(content, self.console_print)
            prog.run()
        def delete_file():
            sel = file_list.curselection()
            if not sel:
                messagebox.showinfo("Info", "Select a file.")
                return
            name = file_list.get(sel[0])
            if messagebox.askyesno("Confirm", f"Delete {name}?"):
                del self.fs['files'][name]
                save_fs(self.fs)
                refresh()
                self.refresh_file_list()
        tk.Button(btns, text='Refresh', command=refresh,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x', pady=4)
        tk.Button(btns, text='Edit', command=open_edit,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x', pady=4)
        tk.Button(btns, text='Cat', command=open_cat,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x', pady=4)
        tk.Button(btns, text='Run', command=run_file,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x', pady=4)
        tk.Button(btns, text='Delete', command=delete_file,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x', pady=4)
        tk.Button(btns, text='Close', command=win.destroy,
                  bg=self._gold_btn_bg, activebackground=self._gold_active, fg=self._gold_fg).pack(fill='x', pady=10)

    
    def launch_sandbox_app(self):
        try:
            Sandbox(self)
        except Exception as e:
            messagebox.showerror("Sandbox error", str(e))


    def _on_command(self):
        line = self.cmd_entry.get().strip()
        self.cmd_entry.delete(0, 'end')
        if not line:
            return
        self.console_print(PROMPT + line)
        try:
            parts = shlex.split(line)
        except Exception as e:
            self.console_print('Parse error:', e)
            return
        if not parts:
            return
        cmd = parts[0].lower()
        args = parts[1:]
        if cmd in ('exit', 'quit'):
            self.console_print('Goodbye.')
            self.destroy()
        elif cmd == 'ls':
            for name in sorted(self.fs['files'].keys()):
                size = len(self.fs['files'][name].splitlines())
                self.console_print(f"{name:20} {size:6} lines")
        elif cmd == 'cat':
            if not args: self.console_print('Usage: cat <file>')
            else:
                txt = self.fs['files'].get(args[0])
                if txt is None: self.console_print('No such file.')
                else: self.console_print(txt)
        elif cmd == 'edit':
            if not args: self.console_print('Usage: edit <file>')
            else: self.open_editor(args[0])
        elif cmd == 'mk':
            if not args: self.console_print('Usage: mk <file>')
            else:
                name = args[0]
                if name in self.fs['files']:
                    self.console_print('File exists.')
                else:
                    self.fs['files'][name] = ''
                    save_fs(self.fs)
                    self.refresh_file_list()
                    self.console_print('Created.')
        elif cmd == 'rm':
            if not args: self.console_print('Usage: rm <file>')
            else:
                name = args[0]
                if name in self.fs['files']:
                    del self.fs['files'][name]
                    save_fs(self.fs)
                    self.refresh_file_list()
                    self.console_print('Removed.')
                else:
                    self.console_print('No such file.')
        elif cmd == 'save':
            save_fs(self.fs)
            self.console_print('Saved.')
        elif cmd == 'format':
            confirm = simpledialog.askstring('Confirm', "Type 'YES' to confirm format:")
            if confirm == 'YES':
                if os.path.exists(FS_FILENAME):
                    os.remove(FS_FILENAME)
                self.fs = load_fs()
                save_fs(self.fs)
                self.refresh_file_list()
                self.console_print('Formatted disk (FS reset).')
            else:
                self.console_print('Canceled.')
        elif cmd == 'oracle':
            self.oracle_gui()
        elif cmd == 'run':
            if not args: self.console_print('Usage: run <file>')
            else:
                name = args[0]
                content = self.fs['files'].get(name)
                if content is None:
                    self.console_print('No such file.')
                else:
                    prog = BasicProgram(content, self.console_print)
                    prog.run()
        elif cmd == 'reboot':
            self.console_print('Rebooting...')
            time.sleep(0.2)
            self.boot_sequence()
        elif cmd == 'doom':
            launch_tiny_doom()
        elif cmd == 'help':
            self.console_print("Available commands: ls, cat <file>, edit <file>, mk <file>, rm <file>, run <file>, save, format, oracle, reboot, doom, art, users, settings, snake, tetris, help, exit")
        elif cmd == 'art':
            self.launch_art()
        elif cmd == 'users':
            self.open_users()
        elif cmd == 'settings':
            self.open_settings()
        
        elif cmd == 'tetris':
            self.launch_tetris()
        else:
            self.console_print('Unknown command. Type help.')


def main():
    app = PalaceGUI()
    app.mainloop()

if __name__ == '__main__':
    main()

    
#PALACE OS COPYRIGHT© 2025 - OWNERSHIP RAUL BUTA