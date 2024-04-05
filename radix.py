from __future__ import absolute_import
import numpy as np
from itertools import chain, accumulate
from functools import lru_cache, partial
from operator import mul
from warnings import warn

@lru_cache(maxsize=16)
def _radix_converter(base: np.ndarray):
    as_tuple = tuple(accumulate(base[:0:-1], func=mul, initial=1))[::-1]
    if np.can_cast(np.min_scalar_type(as_tuple), np.uintp):
        return np.array(as_tuple, dtype=np.uintp)
    else:
        return np.array(as_tuple, dtype=object)

def radix_converter(base):
    return _radix_converter(tuple(base))

@lru_cache(maxsize=16)
def _uniform_base_test(base):
    return np.array_equiv(base[0], base)
def uniform_base_test(base):
    return _uniform_base_test(tuple(base))

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

# def to_bits(integers: np.ndarray, mantissa=0, sanity_check=False, verbose=False):
#     if sanity_check:
#         min_mantissa = int.bit_length(np.amax(integers))
#         if min_mantissa>mantissa:
#             if verbose:
#                 warn("Warning: Mantissa is too small to accommodate an integer, overriding.")
#             mantissa = min_mantissa
#     if mantissa==0:
#         mantissa = int.bit_length(np.amax(integers))
#     if mantissa<=8:
#         bytes_array = np.expand_dims(np.asarray(integers, np.uint8), axis=-1)
#     elif mantissa<=16:
#         bytes_array = flip_array_last_axis(np.expand_dims(np.asarray(integers,np.uint16), axis=-1).view(np.uint8))
#     elif mantissa <= 32:
#         bytes_array = flip_array_last_axis(np.expand_dims(np.asarray(integers, np.uint32), axis=-1).view(np.uint8))
#     elif mantissa <= 64:
#         bytes_array = flip_array_last_axis(np.expand_dims(np.asarray(integers, np.uint64), axis=-1).view(np.uint8))
#     else:
#         mantissa_for_bytes = np.floor_divide(mantissa, 8) + 1
#         to_bytes_ufunc = np.vectorize(lambda x: np.frombuffer(int.to_bytes(x, length=mantissa_for_bytes, byteorder='big'), dtype=np.uint8), signature='()->(n)')
#         bytes_array = to_bytes_ufunc(integers)
#     return np.unpackbits(bytes_array, axis=-1, bitorder='big')[..., -mantissa:]

# import numba
# @numba.vectorize([
#     numba.int64(numba.uint8, numba.uint8),
#     numba.int64(numba.uint8, numba.int64),
#     numba.int64(numba.uint, numba.uint),
#     numba.int64(numba.uint, numba.int64),
#     numba.int64(numba.int64, numba.int64)
# ], nopython = True)
# def pack_a_bit(byte, bit):
#     #return byte << 1 | bit
#     #one = 1
#     return np.bitwise_or(np.left_shift(byte, 1), bit)
#
# def from_bits(smooshed_bit_array):
#     return pack_a_bit.reduce(smooshed_bit_array, axis=-1)



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
    print("From bits to bytes: ", from_bits_to_bytes(as_bits, bitorder=bitorder))
    return from_bytes(from_bits_to_bytes(as_bits, bitorder=bitorder), byteorder=bitorder)











#
# def from_bits(smooshed_bit_array: np.ndarray):
#     mantissa = smooshed_bit_array.shape[-1]
#     if mantissa > 0:
#         ready_for_viewing = np.packbits(flip_array_last_axis(smooshed_bit_array), axis=-1, bitorder='little')
#         final_dimension = ready_for_viewing.shape[-1]
#         if mantissa<=8:
#             return np.squeeze(ready_for_viewing, axis=-1)
#         elif mantissa<=16:
#             return np.squeeze(ready_for_viewing.view(np.uint16), axis=-1)
#         elif mantissa <= 32:
#             pad_size = 4-final_dimension
#             if pad_size == 0:
#                 return np.squeeze(ready_for_viewing.view(np.uint32), axis=-1)
#             else:
#                 npad = [(0, 0)] * ready_for_viewing.ndim
#                 npad[-1] = (0, pad_size)
#                 return np.squeeze(np.ascontiguousarray(
#                     np.pad(ready_for_viewing, pad_width=npad, mode='constant', constant_values=0)
#                 ).view(np.uint32), axis=-1)
#         elif mantissa <= 64:
#             pad_size = 8-final_dimension
#             if pad_size == 0:
#                 return np.squeeze(ready_for_viewing.view(np.uint64), axis=-1)
#             else:
#                 npad = [(0, 0)] * ready_for_viewing.ndim
#                 npad[-1] = (0, pad_size)
#                 return np.squeeze(np.ascontiguousarray(
#                     np.pad(ready_for_viewing, pad_width=npad, mode='constant', constant_values=0)
#                 ).view(np.uint64), axis=-1)
#         elif mantissa > 64:
#             print("Warning: Integers exceeding 2^64, possible overflow errors.")
#             return np.squeeze(from_littleordered_bytes(ready_for_viewing), axis=-1)
#     else:
#         return np.zeros(smooshed_bit_array.shape[:-2], dtype=int)

def _from_digits(digits_array, base):
    return np.matmul(digits_array, radix_converter(base))

def from_digits(digits_array, base):
    digits_array_as_array = np.asarray(digits_array)
    if min(digits_array_as_array.shape) > 0:
        if binary_base_test(base):
            return from_bits(digits_array_as_array)
        else:
            return _from_digits(digits_array_as_array, base)
    else:
        return digits_array_as_array.reshape(digits_array_as_array.shape[:-1])




def array_from_string(string_array):
    as_string_array = np.asarray(string_array, dtype=str)
    return np.fromiter(chain.from_iterable(as_string_array.ravel()), np.uint).reshape(
        as_string_array.shape + (-1,))


def from_string_digits(string_digits_array, base):
    return from_digits(array_from_string(string_digits_array), base)


def _to_digits(integer, base):
    if len(base)<=32:
        return np.stack(np.unravel_index(np.asarray(integer, dtype=np.intp), base), axis=-1)
    else:
        return _to_digits_numba(integer, base)

def _to_digits_numba(integer, base):
    arrays = []
    x = np.array(integer, copy=True)
    #print(reversed_base(base))
    for b in np.flipud(base).flat:
        x, remainder = np.divmod(x, b)
        arrays.append(remainder)
    return flip_array_last_axis(np.stack(arrays, axis=-1))

def to_digits(integer, base, sanity_check=False):
    if sanity_check:
        does_it_fit = np.amax(integer) < np.multiply.reduce(np.flipud(base).astype(dtype=np.ulonglong))
        assert does_it_fit, "Base is too small to accommodate such large integers."
    if binary_base_test(base):
        return to_bits(integer, len(base))
    else:
        return _to_digits(integer, base)


def array_to_string(digits_array):
    before_string_joining = np.asarray(digits_array, dtype=str)
    raw_shape = before_string_joining.shape
    return np.array(list(
        map(''.join, before_string_joining.reshape((-1, raw_shape[-1])).astype(str))
    )).reshape(raw_shape[:-1]).tolist()


def to_string_digits(integer, base):
    return array_to_string(to_digits(integer, base))

# def bitarrays_to_ints(bit_array):
#     bit_array_as_array = np.asarray(bit_array)
#     shape = bit_array_as_array.shape
#     (numrows, numcolumns) = shape[-2:]
#     # return from_digits(
#     #     from_bits(bit_array_as_array),
#     #     np.broadcast_to(2**numcolumns, numrows))
#     return from_bits(bit_array_as_array.reshape(shape[:-2]+(numrows * numcolumns,)))

@lru_cache(maxsize=None)
def _bitarray_to_int(bit_array):
    if len(bit_array):
        bit_array_as_array = np.asarray(bit_array, dtype=bool)
        (numrows, numcolumns) = bit_array_as_array.shape
        rows_basis = np.repeat(2**numcolumns, numrows)
        first_conversion = from_bits(bit_array_as_array)
        return _from_digits(first_conversion, rows_basis)
    else:
        return 0
    # return from_bits(bit_array.ravel()).tolist()

def bitarray_to_int(bit_array):
    return _bitarray_to_int(tuple(map(tuple, bit_array)))

# def ints_to_bitarrays(integer, numcolumns):
#     # numrows = -np.floor_divide(-np.log1p(integer), np.log(2**numcolumns)).astype(int).max() #Danger border case
#     # return np.reshape(to_bits(integer, numrows * numcolumns), np.asarray(integer).shape + (numrows, numcolumns))
#     return to_bits(integer, numcolumns)

if __name__ == '__main__':

    print(to_digits([3,5,12,100], base=np.hstack((np.repeat(1,32),(3,4,5)))))
    print(to_digits([3, 5, 12, 100], base=np.hstack((np.repeat(1, 32), (3, 4, 5)))).shape)

    integers = [[234, 1237, 543, 23], [53, 234, 732, 123]]
    base = (2, 3, 2, 3, 4, 2, 2, 3, 2)
    # digits_array = to_digits(integers, base)
    # print(to_digits(integers, base))
    # print(to_string_digits(integers, base))
    # print(np.array_equiv(integers, from_digits(to_digits(integers, base), base)))
    print(np.array_equiv(integers, from_string_digits(to_string_digits(integers, base), base)))
    integers = np.ravel(integers)
    print(np.array_equiv(integers, from_string_digits(to_string_digits(integers, base), base)))
    # integers = 1237
    # print(to_digits(integers,base))
    # print(from_digits(to_digits(integers,base),base))
    #
    # integers = [[234, 1237, 543, 23], [53, 234, 732, 123]]
    # base =np.broadcast_to(2,11)
    # digits_array = to_digits(integers, base)
    # print(to_digits(integers, base))
    # print(to_string_digits(integers, base))
    # print(np.array_equiv(integers, from_digits(to_digits(integers, base), base)))
    # print(np.array_equiv(integers, from_string_digits(to_string_digits(integers, base), base)))
    # integers = 1237
    # print(to_digits(integers,base))
    # print(from_digits(to_digits(integers,base),base))
    #
    # print(from_digits([],base))
    # print(to_digits([], base))
    #print(int_to_bitarray(integers, 11))
    # as_bits = to_bits(integers, 11)
    # print("Effect of encoding to bits: ", as_bits, as_bits.shape)
    # print("Effect of bit encoding and decoding: ", from_bits(as_bits))
    print("Testing bit arithmetic for n-dimensional arrays...")
    print(np.array_equiv(integers, from_bits(to_bits(integers, 11))))

    print("Testing for arithmetic overflow.")
    integers = [5,40,12312312,2**63,2**64,2**65]
    print("Integers to encode: ", integers)
    # print("Testing sanity_check option:")
    # encoded_integers = to_bits(integers, 65, sanity_check=True)
    # print("Encoded integers: ", encoded_integers)
    # print(np.array(integers).view(np.uint8))
    encoded_integers = to_bits(integers, 66)
    print("Encoded integers: ", encoded_integers)
    packed_bits = np.packbits(encoded_integers[:,::-1], axis=-1, bitorder='little')
    print("Packed Bits: ", packed_bits)
    recoded_integers = from_bits(encoded_integers)
    print("Decoded integers: ", recoded_integers)
    print(np.array_equiv(integers, recoded_integers))