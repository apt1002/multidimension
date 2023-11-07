//! A pure-Rust library providing high-level manipulation of multi-dimensional
//! arrays.
//!
//! [`Array<I, T>`] represents an array of `T` indexed by `I`. The `T` values
//! are internally stored in a [`Box<[T]>`], which is a dense 1-dimensional
//! representation. The purpose of the `Array` wrapper is to look like a
//! multi-dimensional collection. To achieve this, the index type `I` can be
//! any type that implements [`Index`]. This includes scalar types such as
//! `usize` and `bool`, but also tuples of other types that implement
//! `Index`. You are encouraged to write your own index types.
//!
//! Trait [`View`] is the main way to access and manipulate [`Array`]s. Unlike
//! `Array`, a `View` doesn't store anything, but instead computes values on
//! demand. In this respect it is a bit like [`std::iter::Iterator`], and
//! indeed `View` offers some of the same methods as `Iterator`, including
//! [`View::map()`], [`View::zip()`] and [`View::collect()`]. However, unlike
//! `Iterator`, `View`s are immutable; getting values using [`View::at()`] does
//! not mutate the `View`. You are encouraged to use `View`s compositionally,
//! like `Iterator`s, and to `collect()` the results into an `Array` only at
//! the end of a chain of operations.
//!
//! All types in this crate which implement `View` (including `Array`) define
//! all of Rust's arithmetic operators (`+`, `*` etc.) to mean pointwise
//! arithmetic on the elements. When applied to `View`s with different
//! `Index` types, elements are replicated as necessary using a [`Broadcast`]
//! rule, similarly to [`NumPy`](https://numpy.org/).
//!
//! `Index`es are often tuples or nested tuples. Trait [`Isomorphic`]
//! defines conversions between different tuple structures. For example,
//! `((A,), B, C)` implements `Isomorphic<(A, (B, C))>`. Methods of `View`
//! exploit tuple isomorphism to ergonomically express which groups of
//! `Index`es to apply operations to. The main thing you can't do with
//! tuple isomorphism is to reorder the `Index`es; for this you need to
//! use [`View::transpose()`]. Either way, the code to manipulate the array
//! indices is all automatically generated at compile-time, and should mostly
//! be optimized away by the compiler.

#[inline(always)]
fn div_mod(n: usize, d: usize) -> (usize, usize) { (n / d, n % d) }

mod tuple;
pub use tuple::{NonTuple, Isomorphic};

mod coat;
pub use coat::{Coat, Coated};

mod index;
pub use index::{Size, Index, StaticIndex, All};

mod broadcast;
pub use broadcast::{Broadcast};

pub mod int;

pub mod ops;
use ops::{Binary};

pub mod view;
pub use view::{Push, NewView, View, ViewRef, ViewMut, Scalar, fn_view};

mod array;
pub use array::{Array};
