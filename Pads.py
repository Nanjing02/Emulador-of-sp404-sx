import os
import sys
from pathlib import Path

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None

try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
except ImportError:
    AudioSegment = None
    pydub_play = None

pads = {
    'a': 'a.wav',
    'b': 'b.wav',
    'c': 'c.wav',
    'd': 'd.wav',
    'f': 'f.wav',
    'g': 'g.wav',
    'h': 'h.wav',
    'j': 'j.wav',
    'k': 'k.wav',
    'l': 'l.wav',
    'z': 'z.wav',
    'x': 'x.wav',
}

pads_samples = {}


def cargar_sample(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"no se encontro el archivo: {ruta}")

    if sf is not None:
        data, samplerate = sf.read(str(ruta), dtype='float32')
        return {
            'ruta': str(ruta),
            'data': data,
            'samplerate': samplerate,
            'backend': 'sounddevice',
        }

    if AudioSegment is not None:
        audio = AudioSegment.from_wav(str(ruta))
        return {
            'ruta': str(ruta),
            'audiosegment': audio,
            'backend': 'pydub',
        }

def cargar_samples_pad(carpeta):
    carpeta = Path(carpeta)
    if not carpeta.exists():
        raise FileNotFoundError(f"on existe la carpeta de samples: {carpeta}")

    samples = {}
    for tecla, nombre_archivo in pads.items():
        ruta_sample = carpeta / nombre_archivo
        try:
            sample = cargar_sample(ruta_sample)
            samples[tecla] = sample
        except Exception as exc:
            print(f'no se pudo cargar sample "{tecla}" {ruta_sample}: {exc}')
    return samples


def reproducir_sample(sample):
    backend = sample.get('backend')

    if backend == 'sounddevice':
        if sd is None:
            raise RuntimeError('sounddevice no está disponible.')

        sd.play(sample['data'], sample['samplerate'])
        sd.wait()
        return

    if backend == 'pydub':
        if pydub_play is None:
            raise RuntimeError('pydub no esta disponible.')

        pydub_play(sample['audiosegment'])
        return

    raise RuntimeError(f"backend desconocido para reproducir el sample: {backend}")


def manejar_input(tecla):
    tecla = tecla.lower()

    if tecla == 'q':
        print('saliendo...')
        return False

    if tecla in pads_samples:
        print(f'Reproduciendo pad "{tecla}"...')
        try:
            reproducir_sample(pads_samples[tecla])
        except Exception as exc:
            print(f'error al reproducir: {exc}')
        return True

    print(f'tecla "{tecla}" no asignada. usa: {", ".join(sorted(pads_samples.keys()))} o Q para salir.')
    return True


if __name__ == '__main__':
    print('Motor de audio basico para WAV')
    print('================================')

    if len(sys.argv) > 1:
        ruta_carpeta = sys.argv[1]
    else:
        ruta_carpeta = input('ruta de la carpeta de samples: ').strip()

    try:
        pads_samples = cargar_samples_pad(ruta_carpeta)
        if not pads_samples:
            raise RuntimeError('no se cargo ningún sample de pads.')
        print(f'samples cargados: {len(pads_samples)} pads.')
    except Exception as exc:
        print(f'Error al cargar los samples de pads: {exc}')
        sys.exit(1)

    print('presiona una tecla de pads (a,b,c,d,f,g,h,j,k,l,z,x) o Q para salir.')

    if os.name == 'nt':
        import msvcrt

        while True:
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if not key:
                continue
            if not manejar_input(key):
                break
    else:
        while True:
            key = input('> ').strip().lower()
            if not key:
                continue
            if not manejar_input(key):
                break
