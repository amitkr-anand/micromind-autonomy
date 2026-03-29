#!/usr/bin/env python3
"""demo_overlay.py — MicroMind Pre-HIL Phase 4 Demo Overlay (DR-4)"""
import sys, curses, time, json, os, threading

OVERLAY_FILE = '/tmp/micromind_overlay.json'

_FLIGHT_MODES = {
    0: 'MANUAL', 65536: 'ALTCTL', 131072: 'POSCTL',
    393216: 'OFFBOARD', 327680: 'AUTO.LOITER',
    262144: 'AUTO.RTL',
}

def _mode_str(m): return _FLIGHT_MODES.get(m, f'MODE({m})')

class OverlayState:
    def __init__(self):
        self.vio_mode='WAITING...'; self.drift_m=0.0
        self.sp_x=self.sp_y=self.sp_z=0.0
        self.ac_x=self.ac_y=self.ac_z=0.0
        self.sp_hz=0.0; self.px4_mode='DISCONNECTED'
        self.px4_armed=False; self.events=[]
        self.connected=False; self.demo_running=False
        self.elapsed_s=0.0; self.lock=threading.Lock()

    def update_from_file(self):
        try:
            if os.path.exists(OVERLAY_FILE):
                d = json.load(open(OVERLAY_FILE))
                with self.lock:
                    self.vio_mode=d.get('vio_mode',self.vio_mode)
                    self.drift_m=d.get('drift_m',self.drift_m)
                    self.sp_x=d.get('sp_x',self.sp_x)
                    self.sp_y=d.get('sp_y',self.sp_y)
                    self.sp_z=d.get('sp_z',self.sp_z)
                    self.sp_hz=d.get('sp_hz',self.sp_hz)
                    self.demo_running=d.get('running',False)
                    self.elapsed_s=d.get('elapsed_s',0.0)
                    evs=d.get('events',[])
                    if evs: self.events=evs[-8:]
        except Exception: pass

    def update_from_mavlink(self, msg):
        t=msg.get_type()
        with self.lock:
            if t=='HEARTBEAT' and msg.get_srcSystem()==1:
                self.px4_mode=_mode_str(msg.custom_mode)
                self.px4_armed=bool(msg.base_mode&128)
                self.connected=True
            elif t=='LOCAL_POSITION_NED':
                self.ac_x=msg.x; self.ac_y=msg.y; self.ac_z=msg.z

def _mavlink_thread(state):
    try:
        import pymavlink.mavutil as mavutil
        m=mavutil.mavlink_connection('udp:127.0.0.1:14550')
        m.wait_heartbeat(timeout=10)
        while True:
            msg=m.recv_match(type=['HEARTBEAT','LOCAL_POSITION_NED'],
                             blocking=True,timeout=0.5)
            if msg: state.update_from_mavlink(msg)
    except Exception: pass

def _vc(vio_mode):
    if 'OUTAGE' in vio_mode: return 3
    if 'RESUMPTION' in vio_mode: return 2
    return 1

def run_overlay(stdscr, test_mode=False):
    curses.curs_set(0); stdscr.nodelay(True)
    curses.start_color(); curses.use_default_colors()
    for i,fg in [(1,curses.COLOR_GREEN),(2,curses.COLOR_YELLOW),
                 (3,curses.COLOR_RED),(4,curses.COLOR_CYAN),(5,curses.COLOR_WHITE)]:
        curses.init_pair(i,fg,-1)
    state=OverlayState()
    if not test_mode:
        threading.Thread(target=_mavlink_thread,args=(state,),daemon=True).start()
    _tt=[0.0]
    while True:
        if stdscr.getch()==ord('q'): break
        if test_mode:
            _tt[0]+=0.1
            tt=_tt[0]
            with state.lock:
                state.connected=True; state.demo_running=True; state.elapsed_s=tt
                state.sp_hz=20.0; state.sp_x=min(tt/0.6,50.0)
                state.ac_x=state.sp_x-0.1; state.sp_z=-5.0; state.ac_z=-4.95
                if 20<tt<30: state.vio_mode='OUTAGE'; state.drift_m=(tt-20)*0.8
                elif 30<tt<32: state.vio_mode='RESUMPTION'; state.drift_m=0.0
                else: state.vio_mode='NOMINAL'; state.drift_m=0.0
                state.px4_mode='OFFBOARD'; state.px4_armed=True
                state.events=[{'t':round(tt,1),'event':'TELEMETRY',
                               'vio_mode':state.vio_mode,
                               'north_m':round(state.ac_x,1)}]
        else:
            state.update_from_file()
        stdscr.erase(); h,w=stdscr.getmaxyx()
        with state.lock:
            vm=state.vio_mode; dm=state.drift_m
            sx=state.sp_x; sy=state.sp_y; sz=state.sp_z
            ax=state.ac_x; ay=state.ac_y; az=state.ac_z
            hz=state.sp_hz; pm=state.px4_mode; pa=state.px4_armed
            evs=list(state.events); conn=state.connected; el=state.elapsed_s
        C=curses.color_pair; B=curses.A_BOLD
        # Header
        hdr=' MicroMind Pre-HIL — Live Demo Overlay (v1.2 §9.11) '
        stdscr.addstr(0,max(0,(w-len(hdr))//2),hdr[:w-1],C(4)|B)
        stdscr.addstr(1,0,'─'*(w-1),C(4))
        # Overlay 1+2: VIO mode + drift
        stdscr.addstr(3,2,'① VIO MODE:',B)
        stdscr.addstr(3,14,f'{vm:<14}',C(_vc(vm))|B)
        stdscr.addstr(3,30,'② DRIFT:',B)
        dc=C(3) if dm>5 else C(1)
        stdscr.addstr(3,39,f'{dm:6.2f} m',dc|B)
        stdscr.addstr(3,52,f'T+{el:6.1f}s',C(5))
        # Overlay 5+6: Setpoint rate + PX4 mode
        stdscr.addstr(5,2,'⑤ SP RATE:',B)
        hc=C(1) if 18<=hz<=22 else C(3)
        stdscr.addstr(5,13,f'{hz:5.1f} Hz',hc|B)
        stdscr.addstr(5,26,'⑥ PX4 MODE:',B)
        mc=C(1) if pm=='OFFBOARD' else C(2)
        stdscr.addstr(5,38,f'{pm:<12} {"ARMED" if pa else "DISARMED"}',mc|B)
        stdscr.addstr(7,0,'─'*(w-1),C(4))
        # Overlay 3: Trajectory
        stdscr.addstr(8,2,'③ TRAJECTORY (NED metres)',B)
        stdscr.addstr(9,4, f'Setpoint  N={sx:7.2f}  E={sy:7.2f}  D={sz:7.2f}',C(4))
        stdscr.addstr(10,4,f'Actual    N={ax:7.2f}  E={ay:7.2f}  D={az:7.2f}',C(5))
        en=abs(sx-ax); ee=abs(sy-ay)
        ec=C(1) if en<2 and ee<2 else C(2)
        stdscr.addstr(11,4,f'Error     N={en:7.2f}  E={ee:7.2f}',ec)
        stdscr.addstr(13,0,'─'*(w-1),C(4))
        # Overlay 4: Event log
        stdscr.addstr(14,2,'④ MODE TRANSITION LOG',B)
        r=15
        for ev in evs[-6:]:
            ts=ev.get('t',0); k=ev.get('event','')
            ex=' '.join(f"{a}={b}" for a,b in ev.items() if a not in('t','event'))
            line=f"  [{ts:7.3f}s] {k:<22} {ex}"
            ec2=C(3) if 'OUTAGE' in k else C(2) if 'RESUMPTION' in k else C(1) if 'COMPLETE' in k else C(5)
            try: stdscr.addstr(r,0,line[:w-1],ec2)
            except: pass
            r+=1
            if r>=h-3: break
        # Footer
        stdscr.addstr(h-2,0,'─'*(w-1),C(4))
        fs=' [q] quit  |  MicroMind Pre-HIL  |  DR-4 '
        cs='● LIVE' if conn else '○ WAIT'
        cc=C(1) if conn else C(2)
        try:
            stdscr.addstr(h-1,2,fs[:w-12])
            stdscr.addstr(h-1,w-8,cs,cc|B)
        except: pass
        stdscr.refresh()
        time.sleep(0.1)

def main():
    test_mode='--test' in sys.argv
    if test_mode: print("Test mode — press q to quit")
    curses.wrapper(run_overlay,test_mode)

if __name__=='__main__':
    main()
