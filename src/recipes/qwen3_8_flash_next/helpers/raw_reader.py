"""Validated bounded raw checkpoint reader extracted without runtime hooks."""
from __future__ import annotations
import json, os, re, struct, sys
from collections import OrderedDict
from pathlib import Path
from .csf_codec import decode_csf_torch

class CSFCheckpoint:
    """Read immutable small tensors by file offsets without mapping a shard.

    On Windows, safetensors' private mapping reserves the whole shard in system
    commit. Raw read-only files and at most three parsed headers avoid retaining
    that reservation. Every returned tensor owns its bounded bytearray payload;
    its lifetime is independent of the file/header LRU and checkpoint context.
    """
    _ITEM_BYTES = {
        'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
        'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
        'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8,
    }

    def __init__(self, model, max_scale_bytes=1024 * 1024, max_open_files=3):
        self.model = Path(model).resolve()
        self.index = json.loads((self.model / 'model.safetensors.index.json').read_text())['weight_map']
        self.config = json.loads((self.model / 'config.json').read_text())
        self._paths = {}
        for filename in set(self.index.values()):
            path = (self.model / filename).resolve()
            if not path.is_relative_to(self.model):
                raise ValueError('Checkpoint tensor path escapes the model directory')
            self._paths[filename] = path
        self.max_scale_bytes = max_scale_bytes
        if not 1 <= max_open_files <= 3:
            raise ValueError('Raw checkpoint file/header budget must be 1..3')
        self.max_open_files = max_open_files
        self._files = OrderedDict()
        self._closed = False
        self.stats = dict(decoded_projections=0, reconstructed_scale_bytes=0, maximum_projection_scale_bytes=0, skipped_ple_tensors=0,
                          header_reads=0, header_bytes=0, maximum_cached_files=0,
                          payload_reads=0, payload_bytes=0, maximum_payload_bytes=0,
                          current_open_files=0, mapped_shard_bytes=0)

    def path(self, name):
        return self._paths[self.index[name]]

    def _file(self, name):
        if self._closed:
            raise RuntimeError('CSF checkpoint is closed')
        path = self.path(name)
        if path in self._files:
            self._files.move_to_end(path)
            return self._files[path]
        if len(self._files) == self.max_open_files:
            _, evicted = self._files.popitem(last=False)
            evicted['stream'].close()
        stream = path.open('rb', buffering=0)
        try:
            file_size = os.fstat(stream.fileno()).st_size
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('Truncated safetensors length prefix')
            header_size, = struct.unpack('<Q', prefix)
            if not 2 <= header_size <= min(128 * 1024 * 1024, file_size - 8):
                raise ValueError('Invalid or excessive safetensors header length')
            header_bytes = stream.read(header_size)
            if len(header_bytes) != header_size:
                raise ValueError('Truncated safetensors header')
            header = json.loads(header_bytes)
            if not isinstance(header, dict):
                raise ValueError('Safetensors header must be an object')
            entry = dict(stream=stream, header=header, base=8 + header_size,
                         file_size=file_size, path=path)
        except BaseException:
            stream.close()
            raise
        self._files[path] = entry
        self.stats['header_reads'] += 1
        self.stats['header_bytes'] += header_size
        self.stats['current_open_files'] = len(self._files)
        self.stats['maximum_cached_files'] = max(
            self.stats['maximum_cached_files'], len(self._files))
        return entry

    def metadata(self, name):
        """Return validated original dtype, geometry and absolute byte extent."""
        entry = self._file(name)
        source = entry['header'][name]
        dtype, shape, offsets = (
            source['dtype'], source['shape'], source['data_offsets'])
        if dtype not in self._ITEM_BYTES:
            raise ValueError(f'Unsupported original tensor dtype: {dtype}')
        if not isinstance(shape, list) or any(
                type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError('Invalid original tensor shape')
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
                type(offset) is not int for offset in offsets):
            raise ValueError('Invalid original tensor offsets')
        elements = 1
        for dimension in shape:
            elements *= dimension
        start, end = offsets
        byte_length = elements * self._ITEM_BYTES[dtype]
        if not 0 <= start <= end <= entry['file_size'] - entry['base']:
            raise ValueError('Original tensor extent is outside its shard')
        if end - start != byte_length:
            raise ValueError('Original tensor shape/dtype extent mismatch')
        return dict(dtype=dtype, shape=tuple(shape), byte_length=byte_length,
                    byte_offset=entry['base'] + start, path=str(entry['path']))

    def read_tensor(self, name, max_bytes=1024 * 1024):
        """Read one bounded owned CPU tensor; no mmap or payload cache."""
        import torch
        if sys.byteorder != 'little':
            raise RuntimeError('Original checkpoint requires little-endian CPU')
        metadata = self.metadata(name)
        byte_length = metadata['byte_length']
        if not 0 < max_bytes <= 1024 * 1024 or byte_length > max_bytes:
            raise MemoryError('Original tensor exceeds the one-MiB raw-read budget')
        dtype_names = {
            'BOOL': 'bool', 'U8': 'uint8', 'I8': 'int8',
            'F8_E4M3': 'float8_e4m3fn', 'F8_E5M2': 'float8_e5m2',
            'I16': 'int16', 'U16': 'uint16', 'F16': 'float16',
            'BF16': 'bfloat16', 'I32': 'int32', 'U32': 'uint32',
            'F32': 'float32', 'I64': 'int64', 'U64': 'uint64', 'F64': 'float64',
        }
        dtype = getattr(torch, dtype_names[metadata['dtype']])
        if byte_length:
            entry = self._file(name)
            entry['stream'].seek(metadata['byte_offset'])
            payload = bytearray(byte_length)
            view = memoryview(payload)
            consumed = 0
            while consumed < byte_length:
                count = entry['stream'].readinto(view[consumed:])
                if not count:
                    raise EOFError('Original tensor payload was truncated')
                consumed += count
            tensor = torch.frombuffer(payload, dtype=dtype)
        else:
            tensor = torch.empty(0, dtype=dtype)
        self.stats['payload_reads'] += 1
        self.stats['payload_bytes'] += byte_length
        self.stats['maximum_payload_bytes'] = max(
            self.stats['maximum_payload_bytes'], byte_length)
        return tensor.reshape(metadata['shape'])

    def close(self):
        if not self._closed:
            for entry in self._files.values():
                entry['stream'].close()
            self._files.clear()
            self.stats['current_open_files'] = 0
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if not getattr(self, '_closed', True):
            self.close()

    def scale(self, prefix):
        import torch
        fixed_name = prefix + '.weight_scale.nvfp4_csf_fixed'
        exceptions_name = prefix + '.weight_scale.nvfp4_csf_exceptions'
        weight_name = prefix + '.weight'
        quant_name = re.sub(r'\.experts\.\d+\.[^.]+$', '.experts', prefix)
        quant = self.config['quantization_config']['quantized_layers'][quant_name]
        if quant.get('quant_algo') != 'NVFP4' or quant.get('weight_scale_encoding') != 'csf' or quant.get('group_size') != 16:
            raise ValueError('Projection does not declare the inspected original NVFP4-CSF encoding')
        # Inspect the packed weight header without mapping/reading its payload.
        weight_metadata = self.metadata(weight_name)
        shape = weight_metadata['shape']
        if weight_metadata['dtype'] != 'U8' or len(shape) != 2 or shape[1] % 8:
            raise ValueError('Expected original packed expert U8 matrix geometry')
        rows, columns = shape[0], shape[1] * 2 // 16
        if rows * columns > self.max_scale_bytes:
            raise MemoryError('Expert scale projection exceeds the bounded decoder budget')
        fixed = self.read_tensor(fixed_name)
        if fixed.dtype != torch.uint8 or tuple(fixed.shape) != (rows // 16, 16 * (1 + columns // 2)):
            raise ValueError('Unexpected original fixed scale tensor')
        fixed_bytes = fixed.numpy().tobytes()
        exceptions = self.read_tensor(exceptions_name)
        if exceptions.dtype != torch.uint32 or exceptions.ndim != 1:
            raise ValueError('Unexpected original exception tensor')
        exception_bytes = exceptions.view(torch.uint8).numpy().tobytes()
        decoded = decode_csf_torch(fixed_bytes, exception_bytes, rows, columns).view(torch.float8_e4m3fn)
        self.stats['decoded_projections'] += 1
        self.stats['reconstructed_scale_bytes'] += rows * columns
        self.stats['maximum_projection_scale_bytes'] = max(self.stats['maximum_projection_scale_bytes'], rows * columns)
        return decoded
