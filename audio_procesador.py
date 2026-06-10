import sys
import threading
from pathlib import Path
from functools import partial

import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt5.QtCore import pyqtSignal, QObject

from PyQt5 import QtCore, QtWidgets, uic

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


SAMPLERATE = 44100
BLOCKSIZE = 256

PAD_BUTTONS = [
    "Pad1", "Pad2", "Pad3", "Pad4",
    "Pad5", "Pad6", "Pad7", "Pad8",
    "Pad9", "Pad10", "Pad11", "Pad12",
]

TECLAS_PADS = ['a', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x']
TECLAS_BANCOS = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}

lock = threading.Lock()
teclas_presionadas = set()
running = True
listener = None

# variables globales de resample 
resampling = False
resample_buffer = []
resample_target_bank = None
resample_target_pad = None
resample_state = 'idle' 

# parámetros de lofi
LOFI_TARGET_SR = 11025
LOFI_BIT_DEPTH = 10
LOFI_CUTOFF_HZ = 10000.0
LOFI_DRIVE = 3.0


def crear_pad():
    return {
        'sample': None,
        'level': 1.0,
        'mode': 'trigger',
        'start': 0,
        'end': 0,
        'lofi': False,
        'lofi_filter_state': 0.0
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


def cargar_sample(ruta):
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f'No existe: {ruta}')

    data, sr = sf.read(str(ruta), dtype='float32')

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != SAMPLERATE:
        dur = len(data) / sr
        new_len = int(dur * SAMPLERATE)

        x_old = np.linspace(0, 1, len(data), endpoint=False)
        x_new = np.linspace(0, 1, new_len, endpoint=False)
        data = np.interp(x_new, x_old, data)

    return np.ascontiguousarray(data.astype(np.float32))

def aplicar_lofi(data, filter_state):
    if data.size == 0:
        return data, filter_state

    decimation = max(1, SAMPLERATE // LOFI_TARGET_SR)

    if decimation > 1:
        decimated = data[::decimation]

        if decimated.size == 0:
            decimated = data

        data = np.repeat(decimated, decimation)[: len(data)]

    # saturación
    data = np.tanh(data * LOFI_DRIVE) * 0.7 + data * 0.3

    # reducción de bits
    levels = (2 ** (LOFI_BIT_DEPTH - 1)) - 1
    data = np.round(data * levels) / levels

    # low pass
    rc = 1.0 / (2.0 * np.pi * LOFI_CUTOFF_HZ)
    alpha = 1.0 / (1.0 + rc * SAMPLERATE)

    for i in range(len(data)):
        filter_state += alpha * (data[i] - filter_state)
        data[i] = filter_state

    return data, filter_state

def toggle_lofi(tecla):
    pad = bancos[banco_actual][tecla]
    pad['lofi'] = not pad['lofi']

    if not pad['lofi']:
        pad['lofi_filter_state'] = 0.0

    for st in samples_activos:
        if st['tecla'] == tecla:
            st['lofi'] = pad['lofi']
            if not st['lofi']:
                st['lofi_filter_state'] = 0.0

    return pad['lofi']

#definimos el callback
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
                chunk = sample[pos:pos + n] * level

                if st.get('lofi'):
                    chunk, st['lofi_filter_state'] = aplicar_lofi(chunk, st['lofi_filter_state'])

                outdata[out_pos:out_pos + n, 0] += chunk

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


def cambiar_banco(banco):
    global banco_actual

    if banco not in bancos:
        return

    banco_actual = banco
    return banco_actual


def cargar_pad(tecla, ruta):
    if tecla not in TECLAS_PADS:
        raise ValueError('Pad inválido')

    sample = cargar_sample(ruta)
    pad = bancos[banco_actual][tecla]
    pad['sample'] = sample
    pad['start'] = 0
    pad['end'] = len(sample)


def configurar_pad(tecla, opcion, valor):
    if tecla not in TECLAS_PADS:
        raise ValueError('Pad inválido')

    pad = bancos[banco_actual][tecla]

    if pad['sample'] is None:
        raise ValueError('Pad vacío')

    opcion = opcion.upper()

    if opcion == 'LEVEL':
        level = float(valor)
        level = max(0.0, min(level, 2.0))
        pad['level'] = level
        return level

    if opcion == 'MODE':
        modo = str(valor).lower()
        if modo not in ('trigger', 'gate', 'loop'):
            raise ValueError('Modo inválido')
        pad['mode'] = modo
        return modo

    if opcion == 'MARK':
        sample_len = len(pad['sample'])
        start, end = valor
        start = int(start)
        end = int(end)

        start = max(0, min(start, sample_len - 1))
        end = max(start + 1, min(end, sample_len))

        pad['start'] = start
        pad['end'] = end
        return (start, end)

    raise ValueError('Opción inválida')


def iniciar_pad(tecla):
    global samples_activos

    with lock:
        pad = bancos[banco_actual][tecla]

        if pad['sample'] is None:
            return False

        if pad['mode'] == 'loop':
            for st in samples_activos:
                if st['tecla'] == tecla and st['mode'] == 'loop' and st['active']:
                    st['active'] = False
                    return True

        samples_activos.append({
            'tecla': tecla,
            'sample': pad['sample'],
            'pos': pad['start'],
            'start': pad['start'],
            'end': pad['end'],
            'level': pad['level'],
            'mode': pad['mode'],
            'active': True,
            'lofi': pad['lofi'],
            'lofi_filter_state': pad.get('lofi_filter_state', 0.0)
        })

    return True


def soltar_pad(tecla):
    with lock:
        for st in samples_activos:
            if st['tecla'] == tecla and st['mode'] == 'gate':
                st['active'] = False


def iniciar_resample():
    global resample_state, resample_buffer
    global resample_target_bank, resample_target_pad, resampling

    if resample_state != 'idle':
        return False

    if not any(bancos[banco_actual][tecla]['sample'] is None for tecla in TECLAS_PADS):
        return False

    resample_state = 'select_target'
    resample_target_bank = None
    resample_target_pad = None
    resample_buffer = []
    resampling = False
    return True


def cancelar_resample():
    global resample_state, resample_buffer, resampling
    global resample_target_bank, resample_target_pad

    resample_state = 'idle'
    resample_buffer = []
    resample_target_bank = None
    resample_target_pad = None
    resampling = False
    return True


def armar_resample():
    global resample_state

    if resample_state != 'ready':
        return False

    resample_state = 'recording'
    return True


def detener_resample():
    global resampling, resample_buffer, resample_state
    global resample_target_bank, resample_target_pad

    if resample_state == 'idle':
        return False

    resample_state = 'idle'
    resampling = False

    if not resample_buffer or resample_target_bank is None or resample_target_pad is None:
        resample_buffer = []
        resample_target_bank = None
        resample_target_pad = None
        return False

    audio = np.concatenate(resample_buffer, axis=0)
    audio = audio[:, 0].astype(np.float32)

    pad = bancos[resample_target_bank][resample_target_pad]
    pad['sample'] = audio
    pad['start'] = 0
    pad['end'] = len(audio)

    resample_buffer = []
    resample_target_bank = None
    resample_target_pad = None
    return True


def procesar_tecla(key):
    global running

    key = key.lower()

    if key == 'e':
        running = False
        return

    if key in TECLAS_BANCOS:
        cambiar_banco(TECLAS_BANCOS[key])
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


def iniciar_listener():
    global listener

    if keyboard is None:
        return

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

class SP404Dialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        ui_path = Path(__file__).resolve().with_name('interfaz_404.ui')
        if not ui_path.exists():
            ui_path = Path('/mnt/data/interfaz_404.ui')
        uic.loadUi(str(ui_path), self)

        self.setWindowTitle('SP-404SX Simulator')
        self._selected_pad = None
        self._connect_ui()

        self.stream = sd.OutputStream(
            samplerate=SAMPLERATE,
            channels=1,
            blocksize=BLOCKSIZE,
            dtype='float32',
            callback=callback
        )
        self.stream.start()

        iniciar_listener()
        self._update_status('SP-404SX.')
        #el banco A esta inicado por default

    def _connect_ui(self):
        self._pad_map = {}
        for ui_name, tecla in zip(PAD_BUTTONS, TECLAS_PADS):
            button = getattr(self, ui_name)
            self._pad_map[button] = tecla
            button.pressed.connect(partial(self._pad_pressed, tecla))
            button.released.connect(partial(self._pad_released, tecla))
            button.installEventFilter(self)

        self.Banco_A_Button.clicked.connect(lambda: self._set_bank('A'))
        self.Banco_B_Button.clicked.connect(lambda: self._set_bank('B'))
        self.Banco_C_Button.clicked.connect(lambda: self._set_bank('C'))
        self.Banco_D_Button.clicked.connect(lambda: self._set_bank('D'))

        self.resample_button.clicked.connect(self._toggle_resample)
        self.Rec_button.clicked.connect(self._toggle_rec)
        self.Lofi_button.clicked.connect(self._toggle_lofi)
        self.delay_button.clicked.connect(self._toggle_delay)
        
        #botones futuros para enlazar 
        self.gate_button.clicked.connect(lambda: self._placeholder('Gate'))
        self.loop_button.clicked.connect(lambda: self._placeholder('Loop'))
        self.mark_sample_button.clicked.connect(lambda: self._placeholder('Mark'))
        self.stereo_button.clicked.connect(lambda: self._placeholder('Stereo'))
        self.start_end_level.clicked.connect(lambda: self._placeholder('Start/End/Level'))
        self.Del_button.clicked.connect(lambda: self._placeholder('Delete'))
        self.Hold.clicked.connect(lambda: self._placeholder('Hold'))
        self.exitaudio.clicked.connect(self.close)
        self.subpad.clicked.connect(lambda: self._placeholder('Sub Pad'))

    def _update_status(self, text):
        if len(text) > 22:
            text = text[:19] + '...'
        if hasattr(self, 'Info_Salida'):
            self.Info_Salida.setText(text)
        print(text)

    def _placeholder(self, name):
        self._update_status('Pendiente')

    def _set_bank(self, bank):
        cambiar_banco(bank)
        self._update_status(f'Banco {banco_actual}')

    def _set_resample_target(self, bank, tecla):
        global resample_state, resample_target_bank, resample_target_pad
        resample_target_bank = bank
        resample_target_pad = tecla
        resample_state = 'ready'

    def _start_resample_capture(self):
        global resampling, resample_buffer
        if resample_state != 'recording':
            return
        resample_buffer = []
        resampling = True

    def _pad_pressed(self, tecla):
        self._selected_pad = tecla

        if resample_state == 'select_target':
            if bancos[banco_actual][tecla]['sample'] is not None:
                self._update_status(f'Pad no vacío {banco_actual}-{tecla.upper()}')
                return

            self._update_status(f'RSMP PAD {banco_actual}-{tecla.upper()}')
            self._selected_pad = tecla
            self._set_resample_target(banco_actual, tecla)
            return

        if resample_state == 'recording' and not resampling:
            self._update_status('RSMP INICIADO')
            self._start_resample_capture()

        if iniciar_pad(tecla):
            self._update_status(f'Pad {tecla.upper()} -> {banco_actual}')
        else:
            self._update_status(f'Pad vacío {banco_actual}-{tecla.upper()}')

    def _pad_released(self, tecla):
        soltar_pad(tecla)

    def _toggle_resample(self):
        if resample_state != 'idle':
            cancelar_resample()
            self._update_status('RSMP cancelado')
            return

        ok = iniciar_resample()
        if ok:
            self._update_status('RSMP select pad vacío')
        else:
            self._update_status('No pads libres RSMP')

    def _toggle_rec(self):
        if resample_state == 'ready':
            ok = armar_resample()
            if ok:
                self._update_status('RSMP listo, toca pads')
            else:
                self._update_status('RSMP error')
            return

        if resample_state == 'recording' or resampling:
            ok = detener_resample()
            if ok:
                self._update_status('RSMP guardado')
            else:
                self._update_status('RSMP detenido')
            return

        self._update_status('Presiona RESAMPLE + pad vacío primero')
    
    def _toggle_lofi(self):
        if self._selected_pad is None:
            self._update_status('Select pad')
            return

        estado = toggle_lofi(self._selected_pad)
        if estado:
            self._update_status(f'LO-FI ON {banco_actual}-{self._selected_pad.upper()}')
        else:
            self._update_status(f'LO-FI OFF {banco_actual}-{self._selected_pad.upper()}')
            
    def _toggle_delay(self):
        if hasattr(self, 'delay.py'): 
            self.delay.bypass = not self.delay.bypass
            status = "OFF" if self.delay.bypass else "ON"
            self._update_status(f"Delay {status}")
        else:
            self._update_status("Delay no disponible")

    def eventFilter(self, obj, event):
        if obj in self._pad_map and event.type() == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.RightButton:
                tecla = self._pad_map[obj]
                self._load_sample_dialog(tecla)
                return True
        return super().eventFilter(obj, event)

    def _load_sample_dialog(self, tecla):
        ruta, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f'Cargar sample en {banco_actual}-{tecla.upper()}',
            '',
            'Audio (*.wav *.aiff *.aif *.flac *.ogg *.mp3);;Todos (*.*)'
        )

        if not ruta:
            return

        self._selected_pad = tecla
        try:
            cargar_pad(tecla, ruta)
            self._update_status(f'Cargado {banco_actual}-{tecla.upper()}')
        except Exception as e:
            self._update_status('Error carga')

    def closeEvent(self, event):
        global running
        running = False

        try:
            detener_listener()
        except Exception:
            pass

        try:
            if hasattr(self, 'stream') and self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass

        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SP404Dialog()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
