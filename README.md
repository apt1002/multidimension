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

[BLAS]: https://www.netlib.org/blas/
[NumPy]: https://numpy.org/
[std::iter]: https://doc.rust-lang.org/std/iter/index.html

## Examples

```
let a: Array<_, _> = usize::all(3).map(|x| x + 10).diagonal().collect();
assert_eq!(a.as_ref(), [
    10, 0, 0,
    0, 11, 0,
    0, 0, 12,
]);
```

```
let a: Array<bool, usize> = Array::new((), [2, 1]);
let b: Array<usize, &str> = Array::new(3, ["apple", "body", "crane"]);
let ab: Array<bool, &str> = a.compose(b).collect();
assert_eq!(ab.as_ref(), ["crane", "body"])
```

```
let a: Array<usize, usize> = usize::all(3).collect();
let b: Array<usize, &str> = Array::new(3, ["apple", "body", "crane"]);
let ab: Array<usize, (usize, &str)> = a.zip(b).collect();
assert_eq!(ab.as_ref(), [
    (0, "apple"),
    (1, "body"),
    (2, "crane"),
]);
```

```
let a: Array<_, _> = <(usize, usize)>::all((3, 2)).collect();
assert_eq!(a.as_ref(), [
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1),
]);
let b: Array<_, _> = a.transpose::<(), usize, usize, ()>().collect();
assert_eq!(b.as_ref(), [
    (0, 0), (1, 0), (2, 0),
    (0, 1), (1, 1), (2, 1),
]);
```

## Contributions

are welcome!

© 2023 Alistair Turnbull. Please use multidimension at minworks dot co dot uk.
