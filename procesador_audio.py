import os
import sys
import queue
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    from pynput import keyboard
except ImportError:
    print("falta pynput instala con: pip install pynput")
    sys.exit(1)

import msvcrt

# ================= CONFIG =================
teclas_pads = ['a','b','c','d','f','g','h','j','k','l','z','x']
teclas_bancos = {'1':'A','2':'B','3':'C','4':'D'}

SAMPLERATE = 44100
BLOCKSIZE = 256

lock = threading.Lock()
event_queue = queue.Queue()
running = True

teclas_presionadas = set()
listener = None

# ================= INPUT LIMPIO =================
def limpiar_buffer_teclado():
    while msvcrt.kbhit():
        msvcrt.getch()

def input_no_echo(prompt):
    print(prompt, end='', flush=True)
    limpiar_buffer_teclado()

    result = ''
    while True:
        ch = msvcrt.getch()

        if ch == b'\r':
            print()
            break
        elif ch == b'\x08':
            if result:
                result = result[:-1]
                print('\b \b', end='', flush=True)
        else:
            try:
                char = ch.decode('utf-8')
                result += char
                print(char, end='', flush=True)
            except:
                pass
    return result

# ================= LISTENER =================
def iniciar_listener():
    global listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

def detener_listener():
    global listener
    if listener:
        listener.stop()
        listener = None

# ================= PAD =================
def crear_pad():
    return {'sample':None,'level':1.0,'mode':'trigger'}

def crear_banco_vacio():
    return {t:crear_pad() for t in teclas_pads}

bancos = {'A':crear_banco_vacio(),'B':crear_banco_vacio(),'C':crear_banco_vacio(),'D':crear_banco_vacio()}
banco_actual = 'A'
samples_activos = []

# ================= AUDIO =================
def cargar_sample(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(ruta)

    data, sr = sf.read(str(ruta), dtype='float32')

    if len(data.shape)>1:
        data = np.mean(data,axis=1)

    if sr != SAMPLERATE:
        dur = len(data)/sr
        new_len = int(dur*SAMPLERATE)
        x_old = np.linspace(0,1,len(data))
        x_new = np.linspace(0,1,new_len)
        data = np.interp(x_new,x_old,data)

    return data.astype(np.float32)

def callback(outdata, frames, time, status):
    global samples_activos
    outdata.fill(0)

    with lock:
        nuevos = []

        for st in samples_activos:
            s = st['sample']
            pos = st['pos']
            lvl = st['level']
            mode = st['mode']
            active = st['active']

            if mode=='gate' and not active:
                continue

            n = min(frames, len(s)-pos)
            outdata[:n,0] += s[pos:pos+n]*lvl
            pos += n

            if pos >= len(s):
                if mode=='loop' and active:
                    pos = 0
                else:
                    continue

            st['pos']=pos
            nuevos.append(st)

        samples_activos = nuevos

    np.clip(outdata,-1,1,out=outdata)

# ================= LOGICA =================
def cambiar_banco(n):
    global banco_actual
    banco_actual = n
    print("banco:",n)

def cargar_pad(tecla,ruta):
    try:
        bancos[banco_actual][tecla]['sample']=cargar_sample(ruta)
        print("Cargado:",banco_actual,tecla)
    except Exception as e:
        print("Error:",e)

def configurar_pad():
    op = input_no_echo("opcion [LEVEL/MODE/MARK]: ").upper()
    pad = input_no_echo("Pad: ").lower()

    entry = bancos[banco_actual][pad]

    if entry['sample'] is None:
        print("Pad vacío")
        return

    if op=="LEVEL":
        lvl=float(input_no_echo("Nivel: "))
        entry['level']=lvl

    elif op=="MODE":
        modo=input_no_echo("modo [trigger/gate/loop]: ")
        entry['mode']=modo

    elif op=="MARK":
        sample_len = len(entry['sample'])
        print(f"Largo total: {sample_len}")
        try:
            start_input = input_no_echo("Start (0): ").strip()
            start = int(start_input) if start_input else 0
            end_input = input_no_echo(f"End ({sample_len}): ").strip()
            end = int(end_input) if end_input else sample_len
            
            start = max(0, min(start, sample_len))
            end = max(start, min(end, sample_len))
            
            entry['start'] = start
            entry['end'] = end
            print(f"Marcado: {start} a {end} ({end-start} muestras)")
        except ValueError:
            print("Valores invalidos")

def iniciar_pad(tecla):
    with lock:
        pad=bancos[banco_actual][tecla]
        if pad['sample'] is None:
            print("Vacío")
            return

        if pad['mode']=='loop':
            for s in samples_activos:
                if s['tecla']==tecla:
                    s['active']=False
                    return

        samples_activos.append({
            'tecla':tecla,
            'sample':pad['sample'],
            'pos':0,
            'level':pad['level'],
            'mode':pad['mode'],
            'active':True
        })

def soltar_pad(tecla):
    with lock:
        for s in samples_activos:
            if s['tecla']==tecla and s['mode']=='gate':
                s['active']=False

# ================= INPUT =================
def procesar_tecla(k):
    global running

    k=k.lower()

    if k=='e':
        running=False
        return

    if k in teclas_bancos:
        cambiar_banco(teclas_bancos[k])
        return

    if k=='o':   # cargar
        event_queue.put({'type':'load'})
        return

    if k=='p':   # config
        event_queue.put({'type':'config'})
        return

    if k in teclas_pads:
        if k in teclas_presionadas:
            return
        teclas_presionadas.add(k)
        iniciar_pad(k)

def on_press(key):
    try:
        if key.char:
            procesar_tecla(key.char)
    except:
        pass

def on_release(key):
    try:
        if key.char:
            k=key.char.lower()
            teclas_presionadas.discard(k)
            if k in teclas_pads:
                soltar_pad(k)
    except:
        pass

# ================= EVENTOS =================
def procesar_eventos_console():
    global running

    while running:
        try:
            ev = event_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        detener_listener()
        limpiar_buffer_teclado()
        teclas_presionadas.clear()

        if ev['type']=='load':
            pad=input_no_echo("Pad: ").lower()
            ruta=input_no_echo("Ruta: ")
            cargar_pad(pad,ruta)

        elif ev['type']=='config':
            configurar_pad()

        limpiar_buffer_teclado()
        iniciar_listener()

# ================= MAIN =================
if __name__=='__main__':
    print("Sampler V2")
    print('1-4 cambiar banco')
    print("Pads:",teclas_pads)
    print("o=cargar sample | p=[level/mode/mark] | e=salir")

    stream=sd.OutputStream(
        samplerate=SAMPLERATE,
        channels=1,
        blocksize=BLOCKSIZE,
        callback=callback
    )
    stream.start()

    iniciar_listener()

    try:
        procesar_eventos_console()
    except KeyboardInterrupt:
        running=False

    detener_listener()
    stream.stop()
    stream.close()
