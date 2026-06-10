import sys
import numpy as np
import sounddevice as sd
from PyQt5 import QtWidgets
from audio_procesador import SP404Dialog

class DelayEffect:
    def __init__(self, samplerate=44100):
        self.samplerate = samplerate
        self.buffer = None
        self.buffer_size = 0
        self.write_pos = 0
        
        self.mix = 0.5     
        self.feedback = 0.5
        self.delay_ms = 500 
        self.delay_samples = int(self.delay_ms * self.samplerate / 1000)
        
        self.update_delay_buffer(self.delay_samples)
        
    def update_delay_buffer(self, delay_samples):
        if delay_samples < 1:
            delay_samples = 1
        if delay_samples > self.samplerate * 2:
            delay_samples = self.samplerate * 2
            
        new_buffer = np.zeros(delay_samples, dtype=np.float32)
        
        if self.buffer is not None:
            copy_size = min(delay_samples, self.buffer_size)
            new_buffer[:copy_size] = self.buffer[:copy_size]
            
        self.buffer = new_buffer
        self.buffer_size = delay_samples
        self.write_pos = 0
        
    def set_delay_time(self, value):
        self.delay_ms = int((value / 127.0) * 2000.0)
        new_samples = int(self.delay_ms * self.samplerate / 1000)
        self.update_delay_buffer(new_samples)
        return self.delay_ms
        
    def set_feedback(self, value):
        self.feedback = (value / 127.0) * 0.95
        return self.feedback
        
    def set_mix(self, value):
        self.mix = value / 127.0
        return self.mix
        
    def process(self, audio_chunk):
        if self.buffer is None or self.buffer_size == 0 or self.delay_ms == 0:
            return audio_chunk
            
        output = audio_chunk.copy()
        samples = len(audio_chunk)
        
        for i in range(samples):
            read_pos = (self.write_pos - self.delay_samples) % self.buffer_size
            wet_sample = self.buffer[read_pos]
            
            dry_sample = audio_chunk[i, 0]
            mixed = (1 - self.mix) * dry_sample + self.mix * wet_sample
            output[i, 0] = np.clip(mixed, -1.0, 1.0)
            
            feedback_sample = dry_sample + wet_sample * self.feedback
            self.buffer[self.write_pos] = np.clip(feedback_sample, -1.0, 1.0)
            self.write_pos = (self.write_pos + 1) % self.buffer_size
            
        return output

class DelayProcessor:
    def __init__(self, sp404_instance):
        self.sp404 = sp404_instance
        self.delay = DelayEffect(samplerate=44100)
        self.bypass = False
        self.stream = None
        self.sp404.delay = self  
    
        self.sp404.knob2.valueChanged.connect(lambda v: self.on_mix_changed(v))
        self.sp404.knob3.valueChanged.connect(lambda v: self.on_feedback_changed(v))
        self.sp404.knob4.valueChanged.connect(lambda v: self.on_time_changed(v))
        
        self._install_delay_processor()
        
    def on_mix_changed(self, value):
        mix = self.delay.set_mix(value)
        wet_percent = int(mix * 100)
        self.sp404.Info_Salida.setText(f"Delay Mix: {wet_percent}% wet")
        print(f"Delay Mix: {value} -> {wet_percent}% wet")
        
    def on_feedback_changed(self, value):
        feedback = self.delay.set_feedback(value)
        fb_percent = int(feedback * 100)
        self.sp404.Info_Salida.setText(f"Delay Feedback: {fb_percent}%")
        print(f"Delay Feedback: {value} -> {fb_percent}%")
        
    def on_time_changed(self, value):
        delay_ms = self.delay.set_delay_time(value)
        self.sp404.Info_Salida.setText(f"Delay Time: {delay_ms}ms")
        print(f"Delay Time: {value} -> {delay_ms}ms")
        
    def _install_delay_processor(self):
        original_stream = self.sp404.stream
        
        if original_stream is None:
            print("No hay stream de audio")
            return
            
        import audio_procesador as ap
        
        def delayed_callback(outdata, frames, time, status):
            ap.callback(outdata, frames, time, status)
            if not self.bypass:
                delayed = self.delay.process(outdata)
                outdata[:] = delayed
                
        original_stream.stop()
        original_stream.close()
        
        self.stream = sd.OutputStream(
            samplerate=ap.SAMPLERATE,
            channels=1,
            blocksize=ap.BLOCKSIZE,
            dtype='float32',
            callback=delayed_callback
        )
        self.stream.start()
        self.sp404.stream = self.stream
        print("Delay processor instalado correctamente")
        
    def cleanup(self):
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()

def main():
    app = QtWidgets.QApplication(sys.argv)
    sp404 = SP404Dialog()
    sp404.show()
    
    delay_proc = DelayProcessor(sp404)
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()