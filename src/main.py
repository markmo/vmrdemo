from argparse import ArgumentParser
from collections import deque
from datetime import datetime
import numpy as np
import platform
import pyaudio
import soundfile
import struct
import sys
from threading import Thread

sys.path.append('/Users/d777710/src/DeepLearning/Porcupine/binding/python')

from porcupine import Porcupine


class VmrDemo(Thread):

    def __init__(self, library_path, model_file_path, keyword_file_paths,
                 sensitivity=0.5, silence_threshold=100, input_device_index=None, output_path=None):
        super(VmrDemo, self).__init__()
        self._library_path = library_path
        self._model_file_path = model_file_path
        self._keyword_file_paths = keyword_file_paths
        self._sensitivity = sensitivity
        self._silence_threshold = silence_threshold
        self._input_device_index = input_device_index
        self._output_path = output_path
        if self._output_path is not None:
            self._recorded_frames = []

    def run(self):
        n_keywords = len(self._keyword_file_paths)
        porcupine = None
        pa = None
        audio_stream = None
        try:
            porcupine = Porcupine(
                library_path=self._library_path,
                model_file_path=self._model_file_path,
                keyword_file_paths=self._keyword_file_paths,
                sensitivities=[self._sensitivity] * n_keywords
            )
            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length,
                input_device_index=self._input_device_index
            )
            buffer = deque(maxlen=1)
            trace = deque(maxlen=3)
            recording = False
            while True:
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = struct.unpack_from('h' * porcupine.frame_length, pcm)

                if np.abs(pcm).mean() < self._silence_threshold:
                    sys.stdout.write('-')
                    trace.append(0)
                else:
                    sys.stdout.write('!')
                    trace.append(1)

                sys.stdout.flush()

                if sum(trace) == 0:
                    recording = False

                if self._output_path is not None:
                    if recording:
                        if len(buffer) > 0:
                            self._recorded_frames.extend(buffer)
                            buffer.clear()

                        self._recorded_frames.append(pcm)
                    else:
                        buffer.append(pcm)

                result = porcupine.process(pcm)
                if n_keywords == 1 and result:
                    print('[%s] detected keyword' % str(datetime.now()))
                    recording = True
                elif n_keywords > 1 and result >= 0:
                    print('[%s] detected keyword #%d' % (str(datetime.now()), result))
                    recording = True

        except KeyboardInterrupt:
            print('stopping...')
        finally:
            if porcupine is not None:
                porcupine.delete()

            if audio_stream is not None:
                audio_stream.close()

            if pa is not None:
                pa.terminate()

            if self._output_path is not None and len(self._recorded_frames) > 0:
                recorded_audio = np.concatenate(self._recorded_frames, axis=0).astype(np.int16)
                soundfile.write(self._output_path, recorded_audio, samplerate=porcupine.sample_rate, subtype='PCM_16')

    _AUDIO_DEVICE_INFO_KEYS = ['index', 'name', 'defaultSampleRate', 'maxInputChannels']

    @classmethod
    def show_audio_devices_info(cls):
        pa = pyaudio.PyAudio()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            print(', '.join("'%s': '%s'" % (k, str(info[k])) for k in cls._AUDIO_DEVICE_INFO_KEYS))

        pa.terminate()


def _default_library_path():
    machine = platform.machine()
    return '/Users/d777710/src/DeepLearning/Porcupine/lib/mac/%s/libpv_porcupine.dylib' % machine


if __name__ == '__main__':
    # read args
    parser = ArgumentParser(description='Run VMR Demo')
    parser.add_argument('--keyword_file_paths', type=str,
                        default='/Users/d777710/src/DeepLearning/Porcupine/hey harold_mac.ppn',
                        help='comma-separated absolute paths to keyword files')
    parser.add_argument('--library_path', type=str,
                        help="absolute path to Porcupine's dynamic library")
    parser.add_argument('--model_file_path', type=str,
                        default='/Users/d777710/src/DeepLearning/Porcupine/lib/common/porcupine_params.pv',
                        help='absolute path to model parameter file')
    parser.add_argument('--sensitivity', type=str, default=0.5, help='detection sensitivity [0, 1]')
    parser.add_argument('--silence_threshold', type=int, default=100, help='silence threshold')
    parser.add_argument('--input_audio_device_index', type=int, default=None, help='index of input audio device')
    parser.add_argument('--output_path', type=str, default=None,
                        help='absolute path to where recorded audio will be stored. If not set, it will be bypassed.')
    parser.add_argument('--show_audio_devices_info', action='store_true')
    args = parser.parse_args()
    if args.show_audio_devices_info:
        VmrDemo.show_audio_devices_info()
    else:
        VmrDemo(
            library_path=args.library_path if args.library_path is not None else _default_library_path(),
            model_file_path=args.model_file_path,
            keyword_file_paths=[x.strip() for x in args.keyword_file_paths.split(',')],
            silence_threshold=args.silence_threshold,
            sensitivity=args.sensitivity,
            output_path=args.output_path,
            input_device_index=args.input_audio_device_index
        ).run()
