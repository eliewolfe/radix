from __future__ import absolute_import
import numpy as np
from itertools import chain, accumulate
from functools import lru_cache
from operator import mul

def from_digits(digits_array, base, baseorder='big'):
    digits_array_as_array = np.asarray(digits_array)
    if min(digits_array_as_array.shape) > 0:
        if binary_base_test(base):
            return from_bits(digits_array_as_array, bitorder=baseorder)
        else:
            return np.matmul(digits_array, radix_converter(base, baseorder=baseorder))
    else:
        return digits_array_as_array.reshape(digits_array_as_array.shape[:-1])

def to_digits(integer: np.ndarray, base: tuple, baseorder='big'):
    if binary_base_test(base):
        return to_bits(integer, mantissa=len(base), baseorder=baseorder)
    elif len(base) <= 32:
        big_endian_answer =  np.stack(np.unravel_index(
            np.asarray(integer, dtype=np.intp),
            base), axis=-1)
        if baseorder == 'big':
            return big_endian_answer
        elif baseorder == 'little':
            return flip_array_last_axis(big_endian_answer)
        else:
            raise Exception(f'baseorder must be "little" or "big", but was given as {baseorder}')
    else:
        arrays = []
        x = np.array(integer, copy=True)
        for b in base[::-1]:
            x, remainder = np.divmod(x, b)
            arrays.append(remainder)
        little_endian_answer = np.stack(arrays, axis=-1)
        if baseorder == 'little':
            return little_endian_answer
        elif baseorder == 'big':
            return flip_array_last_axis(little_endian_answer)
        else:
            raise Exception(f'baseorder must be "little" or "big", but was given as {baseorder}')

def radix_converter(base, baseorder='big'):
    little_endian_base_converter = _radix_converter_little_endian(tuple(base))
    if baseorder == 'little':
        return little_endian_base_converter
    elif baseorder == 'big':
        return little_endian_base_converter[::-1]
    else:
        raise Exception(f'baseorder must be "little" or "big", but was given as {baseorder}')

@lru_cache(maxsize=16)
def _radix_converter_little_endian(base: tuple):
    as_tuple = tuple(accumulate(base[:0:-1], func=mul, initial=1))
    if np.can_cast(np.min_scalar_type(as_tuple), np.uintp):
        return np.array(as_tuple, dtype=np.uintp)
    else:
        return np.array(as_tuple, dtype=object)

@lru_cache(maxsize=16)
def _binary_base_test(base):
    return np.array_equiv(2, base)
def binary_base_test(base):
    return _binary_base_test(tuple(base))

def flip_array_last_axis(m: np.ndarray):
    return m[..., ::-1]

def to_bytes(integers: np.ndarray, mantissa=0, byteorder='little', cleanup=False):
    if mantissa == 0:
        bit_mantissa = int.bit_length(np.amax(integers))
        mantissa = np.floor_divide(bit_mantissa, 8) + 1
    if mantissa > 8:
        # Cannot use native numpy operation. :-(
        fake_ufunc = np.vectorize(lambda x: np.frombuffer(int.to_bytes(x, length=mantissa, byteorder=byteorder), dtype=np.uint8), signature='()->(n)')
        return fake_ufunc(integers)
    if byteorder == 'little':
        endian = '<'
    elif byteorder == 'big':
        endian = '>'
    else:
        raise Exception(f'byteorder must be "little" or "big", but was given as {byteorder}')
    if mantissa == 1:
        compact_dtype = 'u1'
    elif mantissa == 2:
        compact_dtype = 'u2'
    elif mantissa <= 4:
        compact_dtype = 'u4'
    else:
        compact_dtype = 'u8'
    prior_to_cleanup = np.expand_dims(np.asarray(integers, endian+compact_dtype), axis=-1).view(endian+'u1')
    if cleanup:
        if not mantissa in {1,2,4,8}:
            if byteorder == 'little':
                prior_to_cleanup = prior_to_cleanup[...,:mantissa]
            if byteorder == 'big':
                prior_to_cleanup = prior_to_cleanup[..., -mantissa:]
    return prior_to_cleanup

def to_bits(integers: np.ndarray, mantissa=0, bitorder='little', cleanup=False):
    if mantissa == 0:
        mantissa = int.bit_length(int(np.amax(integers)))
    mantissa_for_bytes = np.floor_divide(mantissa, 8) + 1
    as_bytes = to_bytes(integers, byteorder=bitorder, mantissa=mantissa_for_bytes, cleanup=False)
    prior_to_cleanup = np.unpackbits(as_bytes, axis=-1, bitorder=bitorder)
    if cleanup:
        if not mantissa in {8,16,32,64}:
            if bitorder == 'little':
                prior_to_cleanup = prior_to_cleanup[...,:mantissa]
            if bitorder == 'big':
                prior_to_cleanup = prior_to_cleanup[..., -mantissa:]
    return prior_to_cleanup

def _from_bytes_via_native(as_bytes: np.ndarray, byteorder='little'):
    #Assumes numpy array
    if as_bytes.ndim == 1:
        return int.from_bytes(as_bytes.tolist(), byteorder=byteorder, signed=False)
    else:
        return np.array([_from_bytes_via_native(subarray,  byteorder=byteorder) for subarray in as_bytes])

def from_bits_to_bytes(as_bits: np.ndarray, bitorder='little'):
    if bitorder == 'little':
        return np.packbits(as_bits, axis=-1, bitorder='little')
    elif bitorder == 'big':
        return flip_array_last_axis(np.packbits(flip_array_last_axis(as_bits),
                                       axis=-1, bitorder='little'))
    else:
        raise Exception(f'bitorder must be "little" or "big", but was given as {bitorder}')

def from_bytes(as_bytes: np.ndarray, byteorder='little'):
    if byteorder == 'little':
        endian = '<'
        pad_spot = 0
    elif byteorder == 'big':
        endian = '>'
        pad_spot = -1
    else:
        raise Exception(f'byteorder must be "little" or "big", but was given as {byteorder}')
    last_axis_length = as_bytes.shape[-1]
    if last_axis_length > 8:
        return _from_bytes_via_native(as_bytes, byteorder=byteorder)
    if last_axis_length == 0:
        return np.zeros(as_bytes.shape[:-1], dtype=int) # Potential conflict??
    nof_dim = as_bytes.ndim
    padding = np.zeros((2, nof_dim), dtype=int)
    padding[-1, pad_spot] = 1
    as_bytes = np.ascontiguousarray(as_bytes, dtype='u1')
    if last_axis_length == 1:
        pre_squeeze = as_bytes
    elif last_axis_length == 2:
        pre_squeeze = as_bytes.view(endian + 'u2')
    elif last_axis_length <= 4:
        pad_size = 4 - last_axis_length
        if pad_size > 0:
            padding *= pad_size
            pre_squeeze = np.ascontiguousarray(
                np.pad(as_bytes, pad_width=(padding*pad_size), mode='constant', constant_values=0),
                dtype = 'u1').view(endian + 'u4')
    else:   # Last axis length between 5 and 8
        pad_size = 8 - last_axis_length
        if pad_size > 0:
            padding *= pad_size
            pre_squeeze = np.ascontiguousarray(
                np.pad(as_bytes, pad_width=(padding*pad_size), mode='constant', constant_values=0),
                dtype = 'u1').view(endian + 'u4')
    return np.squeeze(pre_squeeze, axis=-1)

def from_bits(as_bits: np.ndarray, bitorder='little'):
    return from_bytes(from_bits_to_bytes(as_bits, bitorder=bitorder), byteorder=bitorder)

def from_bits_mat(as_bits: np.ndarray, bitorder='little'):
    (numrows, numcolumns) = as_bits.shape[-2:]
    if numrows*numcolumns <= 64:
        new_shape = as_bits.shape[:-2] + (-1,)
        reshaped_bit_array = as_bits.reshape(new_shape)
        return from_bits(reshaped_bit_array, bitorder=bitorder)
    else:
        rows_basis = np.broadcast_to(2**numcolumns, numrows)
        first_conversion = from_bits(as_bits, bitorder=bitorder)
        return from_digits(first_conversion, rows_basis, baseorder=bitorder)

def bitarray_to_int(as_bits: np.ndarray):
    # Function for legacy usage.
    return from_bits_mat(as_bits, bitorder='big')



def array_from_string(string_array):
    as_string_array = np.asarray(string_array, dtype=str)
    return np.fromiter(chain.from_iterable(as_string_array.ravel()), np.uint).reshape(
        as_string_array.shape + (-1,))

def from_string_digits(string_digits_array, base, baseorder='big'):
    return from_digits(array_from_string(string_digits_array), base, baseorder=baseorder)

def array_to_string(digits_array):
    before_string_joining = np.asarray(digits_array, dtype=str)
    raw_shape = before_string_joining.shape
    return np.array(list(
        map(''.join, before_string_joining.reshape((-1, raw_shape[-1])).astype(str))
    )).reshape(raw_shape[:-1]).tolist()

def to_string_digits(integer, base, baseorder='big'):
    return array_to_string(to_digits(integer, base, baseorder=baseorder))


if __name__ == '__main__':
    integers = [[234, 1237, 543, 23], [53, 234, 732, 123]]
    base = (2, 3, 2, 3, 4, 2, 2, 3, 2)
    print("Testing base arithmetic for n-dimensional arrays...")
    print(np.array_equiv(integers, from_digits(to_digits(integers, base, baseorder='big'), base, baseorder='big')))
    print(np.array_equiv(integers, from_digits(to_digits(integers, base, baseorder='little'), base, baseorder='little')))

    print("Testing string manipulation...")
    print(np.array_equiv(integers, from_string_digits(to_string_digits(integers, base), base)))
    integers = np.ravel(integers)
    print(np.array_equiv(integers, from_string_digits(to_string_digits(integers, base), base)))

    print("Testing bit arithmetic for n-dimensional arrays...")
    print(np.array_equiv(integers, from_bits(to_bits(integers, mantissa=11))))
    print(np.array_equiv(integers, from_bits(to_bits(integers, mantissa=0))))

    print("Testing for arithmetic overflow...")
    integers = [5,40,12312312,2**63,2**64,2**65]
    print(np.array_equiv(integers, from_bits(to_bits(integers, mantissa=66))))
    print(np.array_equiv(integers, from_bits(to_bits(integers, mantissa=0))))

    print("Integers to encode: ", integers)
    encoded_integers = to_bits(integers, 66)
    print("Encoded integers: ", encoded_integers)
    recoded_integers = from_bits(encoded_integers)
    print("Decoded integers: ", recoded_integers)