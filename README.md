# Multidimension Arrays

A pure-Rust library providing high-level manipulation of multi-dimensional
arrays.

## Design goals

The focus of this library is to provide an easy and bug-free way of programming
with multi-dimensional arrays. In particular:

 - The internal representation of an array is dense (a boxed slice) but not
SIMD optimized (unless you explicitly make an array of a SIMD type).
 - No sacrifices have been made to integrate with [BLAS] in the back-end, or
with [NumPy] in the front-end. It's simply the cleanest design I could make.
 - The API will be familiar to users of [NumPy] and of Rust's [`std::iter`].
 - The array indices can be of any type that implements `ArrayIndex`, and you
are encouraged to make type distinctions among array indices.
 - The library provides a high-level way of expressing many common operations,
and a safe, modular way of writing new operations if necessary.

## Examples



## Contributions

are welcome!

© 2023 Alistair Turnbull. Please use multidimension at minworks dot co dot uk.
