Wake-word Detection
===================

Lightweight and accurate wake word detection engine using Deep Learning trained in
real-world environments (noise, reverberation).

Uses `Porcupine Library <https://github.com/Picovoice/Porcupine>`_.

`Performance benchmarks <https://github.com/Picovoice/wakeword-benchmark>`_ against alternatives.

::

    python src/main.py
        --keyword_file_paths        # comma-separated absolute paths to keyword files
        --library_path              # absolute path to Porcupine's dynamic library
        --model_file_path           # absolute path to model parameter file
        --sensitivity               # detection sensitivity [0, 1]
        --silence_threshold         # silence threshold
        --input_audio_device_index  # index of input audio device
        --output_path               # absolute path to where recorded audio will be stored.
                                    # If not set, it will be bypassed
        --show_audio_devices_info
