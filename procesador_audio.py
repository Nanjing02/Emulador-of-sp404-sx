import sys
import queue
import threading
from pathlib import Path

import msvcrt
import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    from pynput import keyboard
except ImportError:
    print('Falta pynput. Instala con: pip install pynput')
    sys.exit(1)

# ================= Seccion de los pads =================
TECLAS_PADS = ['a', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x']
TECLAS_BANCOS = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
SAMPLERATE = 44100
BLOCKSIZE = 256

lock = threading.Lock()
event_queue = queue.Queue()
teclas_presionadas = set()
listener = None
running = True

# ================= variables globales para resampling =================
resampling = False
resample_buffer = []
resample_target_bank = None
resample_target_pad = None

# ================= data pad =================
def crear_pad():
    return {
        'sample': None,
        'level': 1.0,
        'mode': 'trigger',
        'start': 0,
        'end': 0
    }


def crear_banco():
    return {k: crear_pad() for k in TECLAS_PADS}


bancos = {
    'A': crear_banco(),
    'B': crear_banco(),
    'C': crear_banco(),
    'D': crear_banco()
}

banco_actual = 'A'
samples_activos = []

# ================= input =================
def limpiar_buffer():
    while msvcrt.kbhit():
        msvcrt.getch()


def input_no_echo(prompt):
    print(prompt, end='', flush=True)
    limpiar_buffer()

    result = ''

    while True:
        ch = msvcrt.getch()

        if ch == b'\r':
            print()
            break

        if ch == b'\x08':
            if result:
                result = result[:-1]
                print('\b \b', end='', flush=True)
            continue

        try:
            char = ch.decode('utf-8')
            result += char
            print(char, end='', flush=True)
        except:
            pass

    return result.strip()

# ================= audio =================
def cargar_sample(ruta):
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f'No existe: {ruta}')

    data, sr = sf.read(str(ruta), dtype='float32')

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    if sr != SAMPLERATE:
        dur = len(data) / sr
        new_len = int(dur * SAMPLERATE)

        x_old = np.linspace(0, 1, len(data), endpoint=False)
        x_new = np.linspace(0, 1, new_len, endpoint=False)

        data = np.interp(x_new, x_old, data)

    return np.ascontiguousarray(data.astype(np.float32))


# ================= callback =================
def callback(outdata, frames, time, status):
    global samples_activos, resample_buffer

    if status:
        print(status)

    outdata.fill(0)
    nuevos = []

    with lock:
        for st in samples_activos:
            sample = st['sample']
            pos = st['pos']
            start = st['start']
            end = st['end']
            level = st['level']
            mode = st['mode']
            active = st['active']

            if sample is None or end <= start:
                continue

            if mode == 'gate' and not active:
                continue

            frames_restantes = frames
            out_pos = 0

            while frames_restantes > 0:
                restante = end - pos

                if restante <= 0:
                    if mode == 'loop' and active:
                        pos = start
                        restante = end - pos
                    else:
                        break

                n = min(frames_restantes, restante)

                outdata[out_pos:out_pos+n, 0] += sample[pos:pos+n] * level

                pos += n
                out_pos += n
                frames_restantes -= n

                if mode in ('trigger', 'gate'):
                    break

            if mode == 'loop':
                if active:
                    st['pos'] = pos
                    nuevos.append(st)

            elif mode == 'gate':
                if active and pos < end:
                    st['pos'] = pos
                    nuevos.append(st)

            else:
                if pos < end:
                    st['pos'] = pos
                    nuevos.append(st)

        samples_activos = nuevos

    np.clip(outdata, -1.0, 1.0, out=outdata)

    if resampling:
        resample_buffer.append(outdata.copy())

# ================= pad control =================
def cambiar_banco(banco):
    global banco_actual

    if banco not in bancos:
        print('Banco inválido')
        return

    banco_actual = banco
    print(f'Banco actual: {banco_actual}')


def cargar_pad(tecla, ruta):
    if tecla not in TECLAS_PADS:
        print('Pad inválido')
        return

    try:
        sample = cargar_sample(ruta)

        pad = bancos[banco_actual][tecla]
        pad['sample'] = sample
        pad['start'] = 0
        pad['end'] = len(sample)

        print(f'Cargado en {banco_actual}-{tecla}')

    except Exception as e:
        print('Error:', e)


def configurar_pad():
    opcion = input_no_echo('Opción [LEVEL/MODE/MARK]: ').upper()

    if opcion not in ('LEVEL', 'MODE', 'MARK'):
        print('Opción inválida')
        return

    pad_key = input_no_echo('Pad: ').lower()

    if pad_key not in TECLAS_PADS:
        print('Pad inválido')
        return

    pad = bancos[banco_actual][pad_key]

    if pad['sample'] is None:
        print('Pad vacío')
        return

    if opcion == 'LEVEL':
        try:
            level = float(input_no_echo('Nivel (0.0 - 2.0): '))
            level = max(0.0, min(level, 2.0))
            pad['level'] = level
            print(f'Nivel: {level}')
        except:
            print('Nivel inválido')

    elif opcion == 'MODE':
        modo = input_no_echo('Modo [trigger/gate/loop]: ').lower()

        if modo not in ('trigger', 'gate', 'loop'):
            print('Modo inválido')
            return

        pad['mode'] = modo
        print(f'Modo: {modo}')

    elif opcion == 'MARK':
        sample_len = len(pad['sample'])

        print(f'Largo: {sample_len}')

        try:
            start = input_no_echo('Start: ')
            end = input_no_echo('End: ')

            start = int(start) if start else 0
            end = int(end) if end else sample_len

            start = max(0, min(start, sample_len - 1))
            end = max(start + 1, min(end, sample_len))

            pad['start'] = start
            pad['end'] = end

            print(f'MARK: {start} -> {end}')

        except:
            print('Valores inválidos')


# ================= playback =================
def iniciar_pad(tecla):
    global samples_activos

    with lock:
        pad = bancos[banco_actual][tecla]

        if pad['sample'] is None:
            print('Pad vacío')
            return

        if pad['mode'] == 'loop':
            for st in samples_activos:
                if st['tecla'] == tecla and st['mode'] == 'loop' and st['active']:
                    st['active'] = False
                    return

        samples_activos.append({
            'tecla': tecla,
            'sample': pad['sample'],
            'pos': pad['start'],
            'start': pad['start'],
            'end': pad['end'],
            'level': pad['level'],
            'mode': pad['mode'],
            'active': True
        })


def soltar_pad(tecla):
    with lock:
        for st in samples_activos:
            if st['tecla'] == tecla and st['mode'] == 'gate':
                st['active'] = False

# ================= resample =================
def iniciar_resample():
    global resampling, resample_buffer
    global resample_target_bank, resample_target_pad

    if resampling:
        print('Ya estás resampleando')
        return

    for tecla in TECLAS_PADS:
        if bancos[banco_actual][tecla]['sample'] is None:
            resample_target_bank = banco_actual
            resample_target_pad = tecla
            resample_buffer = []
            resampling = True

            print(f'RESAMPLE REC -> {banco_actual}-{tecla}')
            return

    print('No hay pads vacíos')


def detener_resample():
    global resampling, resample_buffer
    global resample_target_bank, resample_target_pad

    if not resampling:
        return

    resampling = False

    if not resample_buffer:
        print('Nada grabado')
        return

    audio = np.concatenate(resample_buffer, axis=0)
    audio = audio[:, 0].astype(np.float32)

    pad = bancos[resample_target_bank][resample_target_pad]

    pad['sample'] = audio
    pad['start'] = 0
    pad['end'] = len(audio)

    print(f'RESAMPLE OK -> {resample_target_bank}-{resample_target_pad}')

    resample_buffer = []
    resample_target_bank = None
    resample_target_pad = None

# ================= teclado =================
def procesar_tecla(key):
    global running

    key = key.lower()

    if key == 'e':
        running = False
        return

    if key in TECLAS_BANCOS:
        cambiar_banco(TECLAS_BANCOS[key])
        return

    if key == 'o':
        event_queue.put({'type': 'load'})
        return

    if key == 'p':
        event_queue.put({'type': 'config'})
        return

    if key == '6':
        if resampling:
            detener_resample()
        else:
            iniciar_resample()
        return

    if key in TECLAS_PADS:
        if key in teclas_presionadas:
            return

        teclas_presionadas.add(key)
        iniciar_pad(key)


def on_press(key):
    try:
        if hasattr(key, 'char') and key.char:
            procesar_tecla(key.char)
    except Exception as e:
        print('ERROR:', e)


def on_release(key):
    try:
        if hasattr(key, 'char') and key.char:
            k = key.char.lower()
            teclas_presionadas.discard(k)

            if k in TECLAS_PADS:
                soltar_pad(k)

    except Exception as e:
        print('ERROR:', e)

# ================= Listener =================
def iniciar_listener():
    global listener

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )

    listener.start()


def detener_listener():
    global listener

    if listener:
        listener.stop()
        listener = None

# ================= eventos de consola =================
def procesar_eventos_console():
    global running

    while running:
        try:
            evento = event_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        detener_listener()
        limpiar_buffer()
        teclas_presionadas.clear()

        if evento['type'] == 'load':
            pad = input_no_echo('Pad: ').lower()
            ruta = input_no_echo('Ruta: ').strip('"').strip("'")

            cargar_pad(pad, ruta)

        elif evento['type'] == 'config':
            configurar_pad()

        limpiar_buffer()
        iniciar_listener()

# ================= MAIN =================
if __name__ == '__main__':
    print('Sampler SP404SX')
    print('1-4 = bancos')
    print('o = cargar sample')
    print('p = level/mode/mark')
    print('6 = resample')
    print('e = salir')

    stream = sd.OutputStream(
        samplerate=SAMPLERATE,
        channels=1,
        blocksize=BLOCKSIZE,
        dtype='float32',
        callback=callback
    )

    stream.start()
    iniciar_listener()

    try:
        procesar_eventos_console()

    except KeyboardInterrupt:
        running = False

    detener_listener()
    stream.stop()
    stream.close()